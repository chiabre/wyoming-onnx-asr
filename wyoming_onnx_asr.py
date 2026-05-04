#!/usr/bin/env python3
import os
import sys

# --- ONNX RUNTIME CONFIG (single source of truth) ---
ORT_CONFIG = {
    "log_level": 3,              # 0=VERBOSE, 1=INFO, 2=WARNING, 3=ERROR, 4=FATAL
    "disable_affinity": True,
    "num_threads": 1,
    "execution_mode": "sequential",  # "sequential" or "parallel"
}

# --- APPLY ENV CONFIG BEFORE IMPORTS ---
os.environ["ORT_LOGGING_LEVEL"] = str(ORT_CONFIG["log_level"])
os.environ["ORT_DISABLE_THREAD_AFFINITY"] = "1" if ORT_CONFIG["disable_affinity"] else "0"
os.environ["ORT_NUM_THREADS"] = str(ORT_CONFIG["num_threads"])
os.environ["OMP_NUM_THREADS"] = str(ORT_CONFIG["num_threads"])

import argparse
import asyncio
import logging
import time
import numpy as np
import onnx_asr
import onnxruntime as ort

from wyoming.asr import Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import AsrModel, AsrProgram, Info, Attribution
from wyoming.server import AsyncEventHandler, AsyncServer

_LOGGER = logging.getLogger("wyoming_onnx")

# --- CONFIG ---
MAX_AUDIO_BYTES = 10 * 1024 * 1024
EXPECTED_SAMPLE_RATE = 16000

# --- MODEL REGISTRY ---
MODEL_REGISTRY = {
    "istupakov/parakeet-tdt-0.6b-v3-onnx": {
        "name": "Parakeet TDT 0.6B V3",
        "description": "Ultra-fast multilingual ASR",
        "attribution": "NVIDIA / istupakov",
        "url": "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx",
        "languages": ["en", "de", "es", "fr", "it", "nl", "ru"],
        "version": "2025.8.0",
        "notes": "Best overall: fast + multilingual"
    },
    "istupakov/parakeet-tdt-0.6b-v2-onnx": {
        "name": "Parakeet TDT 0.6B V2",
        "description": "Ultra-fast ASR optimized for English",
        "attribution": "NVIDIA / istupakov",
        "url": "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v2-onnx",
        "languages": ["en"],
        "version": "2025.2.0",
        "notes": "Best for English-only setups"
    },
    "istupakov/canary-1b-v2-onnx": {
        "name": "Canary 1B V2",
        "description": "High-accuracy multilingual Conformer-AED",
        "attribution": "NVIDIA / istupakov",
        "url": "https://huggingface.co/istupakov/canary-1b-v2-onnx",
        "languages": ["en", "de", "es", "fr"],
        "version": "2026.2.26",
        "notes": "Best accuracy, slower than Parakeet"
    }
}

# --- MODEL ALIASES ---
MODEL_ALIASES = {
    "parakeet-v3": "istupakov/parakeet-tdt-0.6b-v3-onnx",
    "parakeet-v2": "istupakov/parakeet-tdt-0.6b-v2-onnx",
    "canary": "istupakov/canary-1b-v2-onnx"
}


def resolve_model(model_arg: str) -> str:
    return MODEL_ALIASES.get(model_arg, model_arg)


class OnnxAsrEventHandler(AsyncEventHandler):
    def __init__(self, model, reader, writer, model_id):
        super().__init__(reader, writer)
        self.model = model
        self.model_id = model_id
        self.audio_data = bytearray()
        self.sample_rate = None

    async def handle_event(self, event: Event) -> bool:
        if event.type == "describe":
            meta = MODEL_REGISTRY.get(self.model_id, {
                "name": self.model_id,
                "description": "ASR Model",
                "languages": ["en"],
                "attribution": "Unknown",
                "url": "",
                "version": "1.0"
            })

            description = meta["description"]
            if "notes" in meta:
                description += " (%s)" % meta["notes"]

            model_info = AsrModel(
                name=meta["name"],
                languages=meta["languages"],
                attribution=Attribution(name=meta["attribution"], url=meta["url"]),
                installed=True,
                description=description,
                version=meta["version"]
            )

            info = Info(asr=[AsrProgram(
                name="onnx-asr",
                description="ONNX ASR Server",
                attribution=Attribution(name="istupakov", url=""),
                installed=True,
                version="0.12.0",
                models=[model_info]
            )])

            await self.write_event(info.event())
            return True

        elif AudioStart.is_type(event.type):
            self.audio_data.clear()
            self.sample_rate = getattr(event, "rate", None)

            if self.sample_rate and self.sample_rate != EXPECTED_SAMPLE_RATE:
                _LOGGER.warning(
                    "Unexpected sample rate: %s (expected %s)",
                    self.sample_rate, EXPECTED_SAMPLE_RATE
                )

        elif AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)

            if hasattr(chunk, "rate") and chunk.rate:
                self.sample_rate = chunk.rate

            self.audio_data.extend(chunk.audio)

            if len(self.audio_data) > MAX_AUDIO_BYTES:
                _LOGGER.warning("Audio buffer exceeded limit, resetting")
                self.audio_data.clear()

        elif AudioStop.is_type(event.type):
            if not self.audio_data:
                await self.write_event(Transcript(text="").event())
                return True

            audio_array = (
                np.frombuffer(self.audio_data, dtype=np.int16)
                .astype(np.float32) / 32768.0
            )

            try:
                start = time.perf_counter()
                results = self.model.recognize(audio_array)
                elapsed = time.perf_counter() - start

                text = results.text if hasattr(results, "text") else str(results).strip()

                _LOGGER.info("Transcript: %s", text)
                _LOGGER.info("Inference time: %.3fs", elapsed)

                await self.write_event(Transcript(text=text).event())

            except Exception as e:
                _LOGGER.error("ASR Error: %s", e)
                await self.write_event(Transcript(text="").event())

            finally:
                self.audio_data.clear()

            return True

        return True


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="parakeet-v3",
                        help="Model alias or HF id (parakeet-v3, parakeet-v2, canary)")
    parser.add_argument("--model-dir", default="/opt/wyoming-onnx-asr/data/models")
    parser.add_argument("--uri", default="tcp://0.0.0.0:10300")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--debug", action="store_true")

    # Optional overrides
    parser.add_argument("--threads", type=int, help="Override ORT thread count")
    parser.add_argument("--ort-log-level", type=int, choices=range(0, 5))

    args = parser.parse_args()

    # --- APPLY CLI OVERRIDES ---
    if args.threads:
        ORT_CONFIG["num_threads"] = args.threads

    if args.ort_log_level is not None:
        ORT_CONFIG["log_level"] = args.ort_log_level

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    model_id = resolve_model(args.model)

    # --- SESSION OPTIONS ---
    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = ORT_CONFIG["log_level"]
    sess_options.intra_op_num_threads = ORT_CONFIG["num_threads"]
    sess_options.inter_op_num_threads = ORT_CONFIG["num_threads"]

    if ORT_CONFIG["execution_mode"] == "sequential":
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    else:
        sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL

    sess_options.add_session_config_entry("session.intra_op.allow_spinning", "0")

    os.environ["ONNX_AS_MODELS_DIR"] = args.model_dir

    providers = (
        ["CPUExecutionProvider"]
        if args.cpu
        else ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

    try:
        _LOGGER.info("Loading Model: %s (%s)", args.model, model_id)

        try:
            model = onnx_asr.load_model(
                model_id,
                providers=providers,
                sess_options=sess_options
            )
        except Exception:
            _LOGGER.warning("Primary provider failed, falling back to CPU")
            model = onnx_asr.load_model(
                model_id,
                providers=["CPUExecutionProvider"],
                sess_options=sess_options
            )

        model = model.with_vad(onnx_asr.load_vad("silero"))

        _LOGGER.info(
            "Server Ready. Mode: %s",
            "CPU" if args.cpu else "GPU (CUDA)"
        )
        _LOGGER.info("Listening on: %s", args.uri)

    except Exception as e:
        _LOGGER.error("Failed to load model: %s", e)
        sys.exit(1)

    server = AsyncServer.from_uri(args.uri)

    await server.run(
        lambda r, w: OnnxAsrEventHandler(model, r, w, model_id)
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
