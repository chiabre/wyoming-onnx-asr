#!/usr/bin/env python3
import argparse
import asyncio
import logging
import os
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

# --- VERIFIED MODEL REGISTRY (MAY 2026) ---
# Note: Use verified slug names that onnx-asr can resolve without 401 errors.
MODEL_REGISTRY = {
    "nemo-parakeet-tdt-0.6b-v3": {
        "description": "NVIDIA Parakeet TDT 0.6B V3 - Optimized Multilingual. 25+ languages.",
        "attribution": "NVIDIA / istupakov",
        "url": "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx",
        "languages": ["en", "de", "es", "fr", "it", "pt", "nl", "ru", "uk"],
        "version": "2026.2.0"
    },
    "nemo-parakeet-tdt-0.6b-v2": {
        "description": "NVIDIA Parakeet TDT 0.6B V2 - High-speed English optimized.",
        "attribution": "NVIDIA / istupakov",
        "url": "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v2-onnx",
        "languages": ["en"],
        "version": "2025.5.0"
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
        if event.type == "describe":
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
                attribution=Attribution(name=meta["attribution"], url=meta["url"]),
                installed=True,
                description=meta["description"],
                version=meta["version"]
            )

            info = Info(asr=[AsrProgram(
                name="onnx-asr",
                description="ONNX-based ASR server",
                attribution=Attribution(name="istupakov", url="https://github.com/istupakov/onnx-asr"),
                installed=True,
                version="0.11.0",
                models=[model_info]
            )])
            await self.write_event(info.event())
            return True

        if AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)
            self.audio_data.extend(chunk.audio)
        
        elif AudioStop.is_type(event.type):
            _LOGGER.debug("Processing audio buffer...")
            audio_array = np.frombuffer(self.audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            try:
                # 0.11.0 recognize returns a result object with a .text attribute
                result = self.model.recognize(audio_array)
                text = result.text if hasattr(result, "text") else str(result).strip()

                _LOGGER.info("Transcript: %s", text)
                await self.write_event(Transcript(text=text).event())
            except Exception as e:
                _LOGGER.error("ASR Error: %s", e)

            self.audio_data.clear()
            return False  
        return True

async def main():
    parser = argparse.ArgumentParser()
    # Using the verified slug name for the default
    parser.add_argument("--model", default="nemo-parakeet-tdt-0.6b-v3")
    parser.add_argument("--model-dir", default="/opt/wyoming-onnx-asr/data/models")
    parser.add_argument("--uri", default="tcp://0.0.0.0:10300")
    parser.add_argument("--no-vad", action="store_false", dest="vad")
    parser.set_defaults(vad=True)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(message)s")

    # 0.11.0 checks this env var for the local model cache
    os.environ["ONNX_ASR_MODELS_DIR"] = args.model_dir

    providers = ["CPUExecutionProvider"]
    if not args.cpu:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    try:
        _LOGGER.info(f"Loading model: {args.model}")
        
        # --- 0.11.0 API ---
        # No more 'hub_dir' or 'cache_dir' arguments. 
        # Pass the slug name and providers.
        model = onnx_asr.load_model(
            args.model, 
            providers=providers
        )
        
        if "CUDAExecutionProvider" in ort.get_available_providers() and not args.cpu:
            _LOGGER.info("🚀 STATUS: Running on NVIDIA GPU (CUDA)")
        else:
            _LOGGER.info("🐌 STATUS: Running on CPU")

        if args.vad:
            _LOGGER.info("Applying Silero VAD...")
            model = model.with_vad(onnx_asr.load_vad("silero"))
            
    except Exception as e:
        _LOGGER.error(f"Failed to load model: {e}")
        _LOGGER.info("Hint: Check if the model name is correct or try a verified slug like 'nemo-parakeet-tdt-0.6b-v3'")
        sys.exit(1)

    server = AsyncServer.from_uri(args.uri)
    _LOGGER.info(f"ASR Server ready on {args.uri}")
    await server.run(lambda r, w: OnnxAsrEventHandler(server, model, r, w, args.model))

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
