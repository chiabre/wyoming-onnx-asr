#!/usr/bin/env python3
import os
import sys

# ==========================================================
# ONNX RUNTIME CONFIG
# ==========================================================
ORT_CONFIG = {
    "log_level": 3,
    "disable_affinity": True,
    "num_threads": 1,
    "execution_mode": "sequential",
}

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

# ==========================================================
# LOGGING
# ==========================================================
LOG = logging.getLogger("asr")
STREAM_LOG = logging.getLogger("asr.stream")
DEBUG_LOG = logging.getLogger("asr.debug")

MAX_AUDIO_BYTES = 10 * 1024 * 1024
EXPECTED_SAMPLE_RATE = 16000

# Endpointing tuning (HA critical)
DEFAULT_ENDPOINT_MS = 500  # silence threshold
MIN_AUDIO_MS = 200         # ignore ultra short bursts

# ==========================================================
# MODEL REGISTRY
# ==========================================================
MODEL_REGISTRY = {
    "istupakov/parakeet-tdt-0.6b-v3-onnx": {
        "name": "Parakeet TDT 0.6B V3",
        "description": "Ultra-fast multilingual ASR",
        "attribution": "NVIDIA / istupakov",
        "url": "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx",
        "languages": ["en", "de", "es", "fr", "it", "nl", "ru"],
        "version": "2025.8.0",
    },
    "istupakov/parakeet-tdt-0.6b-v2-onnx": {
        "name": "Parakeet TDT 0.6B V2",
        "description": "Ultra-fast ASR optimized for English",
        "attribution": "NVIDIA / istupakov",
        "url": "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v2-onnx",
        "languages": ["en"],
        "version": "2025.2.0",
    },
}

MODEL_ALIASES = {
    "parakeet-v3": "istupakov/parakeet-tdt-0.6b-v3-onnx",
    "parakeet-v2": "istupakov/parakeet-tdt-0.6b-v2-onnx",
}

def resolve_model(m):
    return MODEL_ALIASES.get(m, m)

# ==========================================================
# TEXT EXTRACTION
# ==========================================================
def extract_text(results):
    if results is None:
        return ""

    try:
        if hasattr(results, "__iter__") and not isinstance(results, (str, bytes)):
            segs = list(results)
            return " ".join(getattr(s, "text", str(s)) for s in segs).strip()

        if hasattr(results, "text"):
            return results.text.strip()

        return str(results).strip()

    except Exception:
        return ""

# ==========================================================
# EVENT HANDLER
# ==========================================================
class OnnxAsrEventHandler(AsyncEventHandler):

    def __init__(self, model, reader, writer, model_id,
                 debug=False, stream_debug=False,
                 endpoint_ms=DEFAULT_ENDPOINT_MS):

        super().__init__(reader, writer)

        self.model = model
        self.model_id = model_id

        self.debug = debug
        self.stream_debug = stream_debug

        self.audio_data = bytearray()
        self.sample_rate = EXPECTED_SAMPLE_RATE
        self.chunk_count = 0

        self.start_time = None
        self.last_audio_time = None

        self.endpoint_ms = endpoint_ms

    def run_inference(self, audio):
        if audio is None or len(audio) == 0:
            return None
        return self.model.recognize(audio)

    # ---------------- EVENTS ----------------
    async def handle_event(self, event: Event) -> bool:

        # ================= DESCRIBE =================
        if event.type == "describe":
            meta = MODEL_REGISTRY.get(self.model_id, {"name": self.model_id})

            model_info = AsrModel(
                name=meta["name"],
                languages=meta.get("languages", ["en"]),
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
                description="Streaming ASR (ONNX)",
                attribution=Attribution(name="system", url=""),
                installed=True,
                version="0.13.0",
                models=[model_info]
            )])

            await self.write_event(info.event())
            return True

        # ================= AUDIO START =================
        elif AudioStart.is_type(event.type):

            self.audio_data.clear()
            self.chunk_count = 0

            self.start_time = time.perf_counter()
            self.last_audio_time = self.start_time

            incoming_rate = getattr(event, "rate", None)
            self.sample_rate = incoming_rate or EXPECTED_SAMPLE_RATE

            STREAM_LOG.info("AUDIO START rate=%s", self.sample_rate)

            return True

        # ================= AUDIO CHUNK =================
        elif AudioChunk.is_type(event.type):

            chunk = AudioChunk.from_event(event)

            now = time.perf_counter()
            self.last_audio_time = now

            self.chunk_count += 1
            self.audio_data.extend(chunk.audio)

            if len(self.audio_data) > MAX_AUDIO_BYTES:
                STREAM_LOG.warning("buffer overflow reset")
                self.audio_data.clear()

            if self.stream_debug:
                STREAM_LOG.debug("chunk=%d size=%d", self.chunk_count, len(chunk.audio))

            return True

        # ================= AUDIO STOP =================
        elif AudioStop.is_type(event.type):

            if not self.audio_data:
                LOG.info("EMPTY AUDIO SEGMENT")
                await self.write_event(Transcript(text="").event())
                return True

            audio = (
                np.frombuffer(self.audio_data, dtype=np.int16)
                .astype(np.float32) / 32768.0
            )

            duration = len(audio) / self.sample_rate

            if duration * 1000 < MIN_AUDIO_MS:
                LOG.info("IGNORED SHORT AUDIO (%.2f ms)", duration * 1000)
                await self.write_event(Transcript(text="").event())
                self.audio_data.clear()
                return True

            LOG.info("PROCESSING | %.2fs | chunks=%d", duration, self.chunk_count)

            try:
                t0 = time.perf_counter()

                results = self.run_inference(audio)

                t1 = time.perf_counter()

                text = extract_text(results)

                LOG.info(
                    "ASR RESULT | %.2fs | %s | infer=%.3fs total=%.3fs",
                    duration,
                    text,
                    t1 - t0,
                    t1 - self.start_time if self.start_time else -1
                )

                await self.write_event(Transcript(text=text).event())

            except Exception:
                LOG.exception("ASR FAILURE")
                await self.write_event(Transcript(text="").event())

            finally:
                self.audio_data.clear()

        return True

# ==========================================================
# MAIN
# ==========================================================
async def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="parakeet-v2")
    parser.add_argument("--model-dir", default="/opt/wyoming-onnx-asr/data/models")
    parser.add_argument("--uri", default="tcp://0.0.0.0:10300")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--stream-debug", action="store_true")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--no-vad", action="store_true")
    parser.add_argument("--endpoint-ms", type=int, default=DEFAULT_ENDPOINT_MS)

    args = parser.parse_args()

    # Logging
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if args.debug else logging.INFO)

    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)

    if not root.handlers:
        root.addHandler(handler)

    LOG.setLevel(logging.INFO)
    STREAM_LOG.setLevel(logging.DEBUG if args.stream_debug else logging.INFO)
    DEBUG_LOG.setLevel(logging.DEBUG)

    model_id = resolve_model(args.model)

    # ONNX
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
        LOG.info("Loading model: %s", model_id)

        model = onnx_asr.load_model(
            model_id,
            providers=providers,
            sess_options=sess_options
        )

        if not args.no_vad:
            model = model.with_vad(onnx_asr.load_vad("silero"))

        LOG.info("Server ready (GPU=%s)", not args.cpu)
        LOG.info("Listening on: %s", args.uri)

    except Exception as e:
        LOG.error("Model load failed: %s", e)
        sys.exit(1)

    server = AsyncServer.from_uri(args.uri)

    await server.run(
        lambda r, w: OnnxAsrEventHandler(
            model, r, w, model_id,
            debug=args.debug,
            stream_debug=args.stream_debug,
            endpoint_ms=args.endpoint_ms
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
