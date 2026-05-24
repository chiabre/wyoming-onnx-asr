#!/usr/bin/env python3
import argparse
import asyncio
import logging
import os
import sys
import time
from functools import partial

import numpy as np
import onnx_asr
import onnxruntime as ort
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import AsrModel, AsrProgram, Attribution, Describe, Info
from wyoming.server import AsyncEventHandler, AsyncServer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
LOG = logging.getLogger("asr")

MAX_AUDIO_BYTES = 10 * 1024 * 1024  # 10MB protection guardrail
EXPECTED_SAMPLE_RATE = 16000
MIN_AUDIO_MS = 200

class SimpleAsrEventHandler(AsyncEventHandler):
    def __init__(self, wyoming_info: Info, model, model_lock: asyncio.Lock, reader, writer):
        super().__init__(reader, writer)
        self.wyoming_info_event = wyoming_info.event()
        self.model = model
        self.model_lock = model_lock
        self.audio_data = bytearray()
        self.sample_rate = EXPECTED_SAMPLE_RATE
        self.request_language = None

    async def handle_event(self, event: Event) -> bool:
        if Transcribe.is_type(event.type):
            self.request_language = Transcribe.from_event(event).language
            return True

        if AudioStart.is_type(event.type):
            self.audio_data.clear()
            self.sample_rate = getattr(event, "rate", None) or EXPECTED_SAMPLE_RATE
            return True

        if AudioChunk.is_type(event.type):
            self.audio_data.extend(AudioChunk.from_event(event).audio)
            if len(self.audio_data) > MAX_AUDIO_BYTES:
                self.audio_data.clear()
            return True

        if AudioStop.is_type(event.type):
            if not self.audio_data:
                await self.write_event(Transcript(text="").event())
                return False

            # Zero-copy memory buffer loading
            audio = np.frombuffer(self.audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            duration = len(audio) / self.sample_rate
            self.audio_data.clear()  

            if duration * 1000 < MIN_AUDIO_MS:
                await self.write_event(Transcript(text="").event())
                return False

            target_lang = self.request_language or "en"
            async with self.model_lock:
                try:
                    t0 = time.perf_counter()
                    results = self.model.recognize(audio, language=target_lang, sample_rate=self.sample_rate)
                    text = " ".join(getattr(s, "text", str(s)) for s in list(results)).strip() if results else ""
                    LOG.info("ASR RESULT | Text: '%s' | Latency: %.3fs", text, time.perf_counter() - t0)
                    await self.write_event(Transcript(text=text).event())
                except Exception as e:
                    LOG.exception("Inference processing loop exception")
                    await self.write_event(Transcript(text=f"ERROR: {e}").event())
            return False

        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            return True

        return True

async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model name identifier")
    parser.add_argument("--model-dir", default="/opt/wyoming-onnx-asr/data/models", help="Model storage directory")
    parser.add_argument("--uri", default="tcp://0.0.0.0:10300", help="Socket URI")
    parser.add_argument("--no-vad", action="store_true", help="Disable VAD")
    args = parser.parse_args()

    wyoming_info = Info(asr=[AsrProgram(
        name="onnx-asr", description="Streamlined Low-Latency ONNX Server",
        attribution=Attribution(name="NVIDIA / istupakov", url=""),
        installed=True, version="1.0.0",
        models=[AsrModel(name=args.model, description=args.model, attribution=Attribution(name="NVIDIA"), installed=True, languages=["en"], version="1.0")]
    )])

    # Hardware Session Profiles
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    cuda_options = {
        "device_id": "0",
        "arena_extend_strategy": "kNextPowerOfTwo",
        "cudnn_conv_algo_search": "HEURISTIC",
        "do_copy_in_default_stream": "1"
    }
    providers = [("CUDAExecutionProvider", cuda_options), "CPUExecutionProvider"]

    safe_name = args.model.replace("/", "_").replace(":", "_")
    model_path = os.path.join(args.model_dir, safe_name)

    LOG.info("Loading target model: %s", args.model)
    model = onnx_asr.load_model(model=args.model, path=model_path, providers=providers, sess_options=sess_options)
    if not args.no_vad:
        model = model.with_vad(onnx_asr.load_vad("silero"))

    server = AsyncServer.from_uri(args.uri)
    model_lock = asyncio.Lock()
    
    await server.run(partial(SimpleAsrEventHandler, wyoming_info, model, model_lock))

if __name__ == "__main__":
    asyncio.run(main())