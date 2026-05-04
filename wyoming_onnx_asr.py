#!/usr/bin/env python3
import os
import sys

# --- ONNX RUNTIME CONFIG ---
ORT_CONFIG = {
    "log_level": 3,
    "num_threads": 1,
    "execution_mode": "sequential",
}

# Apply before imports
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
        "languages": ["en", "de", "es", "fr", "it", "nl", "ru"],
    }
}

MODEL_ALIASES = {
    "parakeet-v3": "istupakov/parakeet-tdt-0.6b-v3-onnx",
}


def resolve_model(model_arg: str) -> str:
    return MODEL_ALIASES.get(model_arg, model_arg)


# --- HELPER: normalize results ---
def extract_text(results):
    """
    Normalize ONNX ASR outputs (generator / list / object) into text.
    """
    if results is None:
        return ""

    # Generator / iterable case
    if hasattr(results, "__iter__") and not isinstance(results, (str, bytes)):
        segments = list(results)

        if not segments:
            return ""

        texts = []
        for seg in segments:
            if hasattr(seg, "text"):
                texts.append(seg.text)
            else:
                texts.append(str(seg))

        return " ".join(texts).strip()

    # Direct object
    if hasattr(results, "text"):
        return results.text.strip()

    return str(results).strip()


class OnnxAsrEventHandler(AsyncEventHandler):
    def __init__(self, model, reader, writer, model_id, debug=False):
        super().__init__(reader, writer)
        self.model = model
        self.model_id = model_id
        self.audio_data = bytearray()
        self.sample_rate = None
        self.debug = debug

    async def handle_event(self, event: Event) -> bool:

        # --- DESCRIBE ---
        if event.type == "describe":
            meta = MODEL_REGISTRY.get(self.model_id, {
                "name": self.model_id,
                "languages": ["en"]
            })

            model_info = AsrModel(
                name=meta["name"],
                languages=meta["languages"],
                attribution=Attribution(name="istupakov", url=""),
                installed=True,
                description="ONNX ASR Model",
                version="1.0"
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

        # --- AUDIO START ---
        elif AudioStart.is_type(event.type):
            self.audio_data.clear()
            self.sample_rate = getattr(event, "rate", None)

            _LOGGER.debug("AudioStart received (rate=%s)", self.sample_rate)

        # --- AUDIO CHUNK ---
        elif AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)

            if hasattr(chunk, "rate") and chunk.rate:
                self.sample_rate = chunk.rate

            self.audio_data.extend(chunk.audio)

            if len(self.audio_data) > MAX_AUDIO_BYTES:
                _LOGGER.warning("Audio buffer exceeded limit, resetting")
                self.audio_data.clear()

        # --- AUDIO STOP ---
        elif AudioStop.is_type(event.type):

            if not self.audio_data:
                _LOGGER.warning("Empty audio received")
                await self.write_event(Transcript(text="").event())
                return True

            # Convert audio
            audio_array = (
                np.frombuffer(self.audio_data, dtype=np.int16)
                .astype(np.float32) / 32768.0
            )

            duration = len(audio_array) / EXPECTED_SAMPLE_RATE

            _LOGGER.info(
                "Audio received: %d samples (%.2fs, rate=%s)",
                len(audio_array),
                duration,
                self.sample_rate
            )

            if duration < 0.2:
                _LOGGER.warning("Audio too short, skipping inference")
                await self.write_event(Transcript(text="").event())
                return True

            try:
                start = time.perf_counter()

                results = self.model.recognize(audio_array)

                # Debug: inspect raw results
                if self.debug:
                    _LOGGER.debug("Raw results type: %s", type(results))

                text = extract_text(results)

                elapsed = time.perf_counter() - start

                _LOGGER.info("Transcript: %s", text)
                _LOGGER.info("Inference time: %.3fs", elapsed)

                if self.debug:
                    _LOGGER.debug("Chars: %d", len(text))

                await self.write_event(Transcript(text=text).event())

            except Exception as e:
                _LOGGER.exception("ASR Error")
                await self.write_event(Transcript(text="").event())

            finally:
                self.audio_data.clear()

            return True

        return True


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="parakeet-v3")
    parser.add_argument("--model-dir", default="/opt/wyoming-onnx-asr/data/models")
    parser.add_argument("--uri", default="tcp://0.0.0.0:10300")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--debug", action="store_true")

    # 🔥 KEY: allow disabling VAD
    parser.add_argument("--no-vad", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    model_id = resolve_model(args.model)

    # --- SESSION OPTIONS ---
    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = ORT_CONFIG["log_level"]
    sess_options.intra_op_num_threads = ORT_CONFIG["num_threads"]
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    os.environ["ONNX_AS_MODELS_DIR"] = args.model_dir

    providers = (
        ["CPUExecutionProvider"]
        if args.cpu
        else ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

    try:
        _LOGGER.info("Loading Model: %s (%s)", args.model, model_id)

        model = onnx_asr.load_model(
            model_id,
            providers=providers,
            sess_options=sess_options
        )

        # --- VAD ---
        if not args.no_vad:
            _LOGGER.info("Loading VAD (silero)")
            model = model.with_vad(onnx_asr.load_vad("silero"))
        else:
            _LOGGER.warning("VAD DISABLED")

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
        lambda r, w: OnnxAsrEventHandler(model, r, w, model_id, debug=args.debug)
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
