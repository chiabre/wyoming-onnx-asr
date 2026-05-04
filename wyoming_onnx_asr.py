#!/usr/bin/env python3
import os
import sys

# --- ONNX RUNTIME CONFIG (single source of truth) ---
ORT_CONFIG = {
    "log_level": 3,
    "disable_affinity": True,
    "num_threads": 1,
    "execution_mode": "sequential",
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

# ==========================================================
#                  FULL MODEL REGISTRY (RESTORED)
# ==========================================================
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

# ==========================================================
#                   MODEL ALIASES (UNCHANGED)
# ==========================================================
MODEL_ALIASES = {
    "parakeet-v3": "istupakov/parakeet-tdt-0.6b-v3-onnx",
    "parakeet-v2": "istupakov/parakeet-tdt-0.6b-v2-onnx",
    "canary": "istupakov/canary-1b-v2-onnx"
}


def resolve_model(model_arg: str) -> str:
    return MODEL_ALIASES.get(model_arg, model_arg)


# ==========================================================
#            SAFE TEXT EXTRACTION (GENERATOR FIX)
# ==========================================================
def extract_text(results):
    if results is None:
        return ""

    try:
        if hasattr(results, "__iter__") and not isinstance(results, (str, bytes)):
            segs = list(results)
            if not segs:
                return ""

            return " ".join(
                getattr(s, "text", str(s)) for s in segs
            ).strip()

        if hasattr(results, "text"):
            return results.text.strip()

        return str(results).strip()

    except Exception:
        return ""


# ==========================================================
#                     STREAMING HANDLER
# ==========================================================
class OnnxAsrEventHandler(AsyncEventHandler):

    def __init__(self, model, reader, writer, model_id, debug=False, stream_debug=False):
        super().__init__(reader, writer)
        self.model = model
        self.model_id = model_id

        self.debug = debug
        self.stream_debug = stream_debug

        self.audio_data = bytearray()
        self.sample_rate = EXPECTED_SAMPLE_RATE

        self.chunk_count = 0
        self.partial_text = ""

    # ---------------- INFERENCE ----------------
    def run_inference(self, audio_array):
        if audio_array is None or len(audio_array) == 0:
            return None
        return self.model.recognize(audio_array)

    # ---------------- EVENTS ----------------
    async def handle_event(self, event: Event) -> bool:

        # -------- DESCRIBE --------
        if event.type == "describe":
            meta = MODEL_REGISTRY.get(self.model_id, {
                "name": self.model_id,
                "languages": ["en"]
            })

            model_info = AsrModel(
                name=meta["name"],
                languages=meta["languages"],
                attribution=Attribution(
                    name=meta.get("attribution", ""),
                    url=meta.get("url", "")
                ),
                installed=True,
                description=meta.get("description", ""),
                version=meta.get("version", "1.0")
            )

            info = Info(asr=[AsrProgram(
                name="onnx-asr",
                description="ONNX Streaming ASR",
                attribution=Attribution(name="istupakov", url=""),
                installed=True,
                version="0.12.0",
                models=[model_info]
            )])

            await self.write_event(info.event())
            return True

        # -------- AUDIO START --------
        elif AudioStart.is_type(event.type):

            self.audio_data.clear()
            self.chunk_count = 0
            self.partial_text = ""

            incoming_rate = getattr(event, "rate", None)
            self.sample_rate = incoming_rate or EXPECTED_SAMPLE_RATE

            if self.stream_debug:
                _LOGGER.info("[STREAM] START rate=%s", self.sample_rate)

            return True

        # -------- AUDIO CHUNK --------
        elif AudioChunk.is_type(event.type):

            chunk = AudioChunk.from_event(event)

            self.chunk_count += 1
            self.audio_data.extend(chunk.audio)

            if len(self.audio_data) > MAX_AUDIO_BYTES:
                _LOGGER.warning("[STREAM] buffer overflow reset")
                self.audio_data.clear()

            if self.stream_debug:
                _LOGGER.info("[STREAM] chunk=%d bytes total_chunks=%d",
                             len(chunk.audio), self.chunk_count)

            return True

        # -------- AUDIO STOP --------
        elif AudioStop.is_type(event.type):

            if not self.audio_data:
                await self.write_event(Transcript(text="").event())
                return True

            audio_array = (
                np.frombuffer(self.audio_data, dtype=np.int16)
                .astype(np.float32) / 32768.0
            )

            duration = len(audio_array) / self.sample_rate

            _LOGGER.info(
                "[STREAM] FINAL buffer=%.2fs chunks=%d rate=%s",
                duration,
                self.chunk_count,
                self.sample_rate
            )

            try:
                t0 = time.perf_counter()

                results = self.run_inference(audio_array)

                t1 = time.perf_counter()

                text = extract_text(results)

                if self.debug:
                    _LOGGER.debug("Raw result type=%s", type(results))

                if self.stream_debug:
                    _LOGGER.info("[ASR FINAL] %s", text)
                    _LOGGER.info("[TIMING] inference=%.3fs", t1 - t0)

                await self.write_event(Transcript(text=text).event())

            except Exception:
                _LOGGER.exception("[STREAM] inference failure")
                await self.write_event(Transcript(text="").event())

            finally:
                self.audio_data.clear()

        return True


# ==========================================================
#                          MAIN
# ==========================================================
async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="parakeet-v3")
    parser.add_argument("--model-dir", default="/opt/wyoming-onnx-asr/data/models")
    parser.add_argument("--uri", default="tcp://0.0.0.0:10300")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--stream-debug", action="store_true")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--no-vad", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    model_id = resolve_model(args.model)

    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = ORT_CONFIG["log_level"]
    sess_options.intra_op_num_threads = args.threads or ORT_CONFIG["num_threads"]
    sess_options.inter_op_num_threads = args.threads or ORT_CONFIG["num_threads"]

    if ORT_CONFIG["execution_mode"] == "sequential":
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    os.environ["ONNX_AS_MODELS_DIR"] = args.model_dir

    providers = (
        ["CPUExecutionProvider"]
        if args.cpu
        else ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

    try:
        _LOGGER.info("Loading model: %s", model_id)

        model = onnx_asr.load_model(
            model_id,
            providers=providers,
            sess_options=sess_options
        )

        if not args.no_vad:
            model = model.with_vad(onnx_asr.load_vad("silero"))
        else:
            _LOGGER.warning("VAD disabled")

        _LOGGER.info("Server ready")

    except Exception as e:
        _LOGGER.error("Model load failed: %s", e)
        sys.exit(1)

    server = AsyncServer.from_uri(args.uri)

    await server.run(
        lambda r, w: OnnxAsrEventHandler(
            model,
            r,
            w,
            model_id,
            debug=args.debug,
            stream_debug=args.stream_debug
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
