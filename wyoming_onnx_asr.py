#!/usr/bin/env python3
import argparse
import asyncio
import logging
import sys
from pathlib import Path

import numpy as np
import onnx_asr
import onnxruntime as ort
from wyoming.asr import Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import AsrModel, AsrProgram, Info, Attribution
from wyoming.server import AsyncEventHandler, AsyncServer

_LOGGER = logging.getLogger("wyoming_onnx")

# --- HARDCODED MODEL METADATA ---
MODEL_REGISTRY = {
    "istupakov/parakeet-tdt-0.6b-v2-onnx": {
        "description": "NVIDIA Parakeet TDT 0.6B V2 - Best for English. Ultra-fast TDT architecture.",
        "attribution": "NVIDIA / istupakov",
        "url": "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v2-onnx",
        "languages": ["en"],
        "version": "2025.5.0"
    },
    "istupakov/parakeet-tdt-0.6b-v3-onnx": {
        "description": "NVIDIA Parakeet TDT 0.6B V3 - Best Multilingual. Optimized for 25 European languages.",
        "attribution": "NVIDIA / istupakov",
        "url": "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx",
        "languages": ["bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de", "el", "hu", "it", "lv", "lt", "mt", "pl", "pt", "ro", "sk", "sl", "es", "sv", "ru", "uk"],
        "version": "2025.8.0"
    },
    "istupakov/canary-1b-v2-onnx": {
        "description": "NVIDIA Canary 1B V2 - Highly robust; handles heavy accents and noise best.",
        "attribution": "NVIDIA / istupakov",
        "url": "https://huggingface.co/istupakov/canary-1b-v2-onnx",
        "languages": ["bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de", "el", "hu", "it", "lv", "lt", "mt", "pl", "pt", "ro", "sk", "sl", "es", "sv", "ru", "uk"],
        "version": "2025.8.0"
    },
    "onnx-community/whisper-large-v3-turbo": {
        "description": "Whisper Large V3 Turbo - Broadest coverage (99+ languages). Autoregressive decoding.",
        "attribution": "OpenAI / onnx-community",
        "url": "https://huggingface.co/onnx-community/whisper-large-v3-turbo",
        "languages": ["en", "de", "es", "fr", "it", "pt", "nl", "ru", "zh", "ja", "ko"],
        "version": "2024.10.0"
    },
    "istupakov/canary-180m-flash-onnx": {
        "description": "Canary 180M Flash - Lowest latency for RPi or low-resource hardware.",
        "attribution": "NVIDIA / istupakov",
        "url": "https://huggingface.co/istupakov/canary-180m-flash-onnx",
        "languages": ["en"],
        "version": "2024.10.0"
    },
    "istupakov/whisper-base-onnx": {
        "description": "Whisper Base - Stable general-purpose fallback.",
        "attribution": "OpenAI / istupakov",
        "url": "https://huggingface.co/istupakov/whisper-base-onnx",
        "languages": ["en", "de", "es", "fr", "it", "pt"],
        "version": "2024.5.0"
    }
}

class OnnxAsrEventHandler(AsyncEventHandler):
    def __init__(self, server, model, reader, writer, model_id):
        super().__init__(reader, writer)
        self.server = server
        self.model = model
        self.model_id = model_id
        self.audio_data = bytearray()

    async def handle_event(self, event: Event) -> bool:
        _LOGGER.debug("Received event: %s", event.type)
        
        if event.type == "describe":
            # Get hardcoded info or use generic fallback
            meta = MODEL_REGISTRY.get(self.model_id, {
                "description": f"ONNX ASR Model: {self.model_id}",
                "attribution": "istupakov",
                "url": "https://github.com/istupakov/onnx-asr",
                "languages": ["en"],
                "version": "1.0.0"
            })

            model_info = AsrModel(
                name=self.model_id,
                languages=meta["languages"],
                attribution=Attribution(
                    name=meta["attribution"], 
                    url=meta["url"]
                ),
                installed=True,
                description=meta["description"],
                version=meta["version"]
            )

            info = Info(asr=[AsrProgram(
                name="onnx-asr",
                description="ONNX-based ASR server",
                attribution=Attribution(
                    name="istupakov", 
                    url="https://github.com/istupakov/onnx-asr"
                ),
                installed=True,
                version="1.1.0",
                models=[model_info]
            )])
            await self.write_event(info.event())
            _LOGGER.debug("Sent info response for %s", self.model_id)
            return True

        if AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)
            self.audio_data.extend(chunk.audio)
        
        elif AudioStop.is_type(event.type):
            _LOGGER.info("Processing audio buffer (length: %d bytes)", len(self.audio_data))
            audio_array = np.frombuffer(self.audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            try:
                results = self.model.recognize(audio_array)
                # Parse results from onnx-asr
                if isinstance(results, (list, tuple, filter, map)) or hasattr(results, '__iter__'):
                    text = " ".join([getattr(res, 'text', str(res)) for res in results if res]).strip()
                else:
                    text = getattr(results, 'text', str(results)).strip()

                _LOGGER.info("Transcript: %s", text)
                await self.write_event(Transcript(text=text).event())
            except Exception as e:
                _LOGGER.error("ASR Error: %s", e)

            self.audio_data.clear()
            return False  
        return True

async def main():
    parser = argparse.ArgumentParser()
    # Note: Ensure the --model string matches the keys in MODEL_REGISTRY
    parser.add_argument("--model", default="istupakov/parakeet-tdt-0.6b-v2-onnx")
    parser.add_argument("--model-dir", default="data/models")
    parser.add_argument("--uri", default="tcp://0.0.0.0:10300")
    parser.add_argument("--no-vad", action="store_false", dest="vad")
    parser.set_defaults(vad=True)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(message)s")

    # Path logic: joins model-dir with the model name
    # e.g., data/models/istupakov/parakeet-tdt-0.6b-v2-onnx
    target_path = Path(args.model_dir) / args.model
    
    providers = ["CPUExecutionProvider"]
    if not args.cpu:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    try:
        _LOGGER.info(f"Loading model: {args.model}")
        _LOGGER.info(f"Path: {target_path}")
        
        # Initialize model
        model = onnx_asr.load_model(args.model, str(target_path), providers=providers)
        
        # Verify hardware
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available and not args.cpu:
            _LOGGER.info("🚀 STATUS: Running on NVIDIA GPU (CUDA)")
        else:
            _LOGGER.info("🐌 STATUS: Running on CPU")

        if args.vad:
            _LOGGER.info("Applying Silero VAD...")
            model = model.with_vad(onnx_asr.load_vad("silero"))
            
    except Exception as e:
        _LOGGER.error(f"Failed to load model: {e}")
        sys.exit(1)

    server = AsyncServer.from_uri(args.uri)
    _LOGGER.info(f"ASR Server ready on {args.uri}")
    await server.run(lambda r, w: OnnxAsrEventHandler(server, model, r, w, args.model))

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass