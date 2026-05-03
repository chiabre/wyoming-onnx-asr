import argparse
import asyncio
import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnx_asr import OnnxAsr
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.asr import Transcript

_LOGGER = logging.getLogger(__name__)

# --- 2026 UPDATES ---
# Using the V3 TDT model for better real-time performance and hallucination resistance
DEFAULT_MODEL = "istupakov/parakeet-tdt-0.6b-v3-onnx"

class OnnxAsrEventHandler(AsyncEventHandler):
    def __init__(self, wyoming_info: Event, model: OnnxAsr, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = model
        self.audio_buffer = bytearray()

    async def handle_event(self, event: Event) -> bool:
        if AudioStart.is_instance(event):
            self.audio_buffer.clear()
        elif AudioChunk.is_instance(event):
            chunk = AudioChunk.from_event(event)
            self.audio_buffer.extend(chunk.audio)
        elif AudioStop.is_instance(event):
            # NumPy 2.4 Fix: Use explicit float32 to avoid removal of np.float aliases
            audio_array = (
                np.frombuffer(self.audio_buffer, dtype=np.int16)
                .astype(np.float32) / 32768.0
            )
            
            # Inference
            _LOGGER.debug("Transcribing %s samples", len(audio_array))
            text = self.model.transcribe(audio_array)
            _LOGGER.info("Transcription: %s", text)
            
            await self.write_event(Transcript(text=text).event())
            return False

        return True

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--uri", default="tcp://0.0.0.0:10300")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    args = parser.parse_args()

    # Hardware Optimization for CUDA 13.0 and ORT 1.25.1
    providers = ["CPUExecutionProvider"]
    if not args.cpu and "CUDAExecutionProvider" in ort.get_available_providers():
        providers.insert(0, ("CUDAExecutionProvider", {
            "device_id": 0,
            "arena_extend_strategy": "kSameAsRequested", # Prevents fragmentation in LXC
            "gpu_mem_limit": 2 * 1024 * 1024 * 1024,      # 2GB Limit
        }))

    _LOGGER.info("Loading model: %s", args.model)
    # The onnx-asr library will handle the download/caching of the TDT v3 model
    model = OnnxAsr(args.model, providers=providers)

    server = AsyncServer.from_uri(args.uri)
    _LOGGER.info("Ready! Listening on %s", args.uri)
    
    await server.run(
        lambda: OnnxAsrEventHandler(None, model)
    )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
