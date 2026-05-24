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

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
LOG = logging.getLogger("wyoming_onnx_asr")

MAX_AUDIO_BYTES = 10 * 1024 * 1024  # 10MB protective execution guardrail
EXPECTED_SAMPLE_RATE = 16000
MIN_AUDIO_MS = 200


def extract_model_info(model_str: str) -> dict:
    """Dynamically extracts Wyoming discovery parameters from the model identifier string or path."""
    raw_name = model_str.split("/")[-1].replace("-onnx", "")
    display_name = raw_name.replace("-", " ").replace("_", " ").title()
    display_name = display_name.replace("0.6B", "0.6B").replace("180M", "180M").replace("Ctc", "CTC").replace("Tdt", "TDT")

    lower_name = raw_name.lower()
    if any(token in lower_name for token in ["v3", "canary", "whisper", "multilingual"]):
        # All 25 languages natively supported by the underlying NeMo architecture
        languages = [
            "en", "de", "es", "fr", "it", "nl", "pt", "ru", "pl", "tr", "zh", "ja",
            "ca", "cs", "da", "el", "fi", "hu", "id", "ko", "no", "ro", "sv", "uk", "vi"
        ]
    elif "gigaam" in lower_name or "ru" in lower_name:
        languages = ["ru"]
    elif "ptbr" in lower_name or "portuguese" in lower_name:
        languages = ["pt"]
    else:
        languages = ["en"]

    return {
        "name": display_name,
        "description": f"Dynamically initialized ONNX ASR pipeline: {display_name}",
        "languages": languages
    }


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
                LOG.warning("Audio processing allocation guardrail reached. Clearing buffer.")
                self.audio_data.clear()
            return True

        if AudioStop.is_type(event.type):
            if not self.audio_data:
                await self.write_event(Transcript(text="").event())
                return False

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
                    
                    if isinstance(results, str):
                        text = results.strip()
                    elif hasattr(results, "text"):
                        text = results.text.strip()
                    elif isinstance(results, (list, tuple)) or hasattr(results, "__iter__"):
                        text_segments = []
                        for segment in results:
                            if isinstance(segment, str):
                                text_segments.append(segment)
                            elif hasattr(segment, "text"):
                                text_segments.append(segment.text)
                            else:
                                text_segments.append(str(segment))
                        text = " ".join(text_segments).strip()
                    else:
                        text = str(results).strip()

                    LOG.info("ASR RESULT | Text: '%s' | Latency: %.3fs", text, time.perf_counter() - t0)
                    await self.write_event(Transcript(text=text).event())
                except Exception as e:
                    LOG.exception("Inference engine pipeline error encountered")
                    await self.write_event(Transcript(text=f"ERROR: {e}").event())
            return False

        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            return True

        return True


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="istupakov/parakeet-tdt-0.6b-v3-onnx")
    parser.add_argument("--model-dir", default="data/models")
    parser.add_argument("--uri", default="tcp://0.0.0.0:10300")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-vad", action="store_true")

    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.getLogger().setLevel(log_level)
    LOG.setLevel(log_level)

    # =====================================================================
    # RESOLVE RELATIVE PATHS ROBUSTLY
    # =====================================================================
    if not os.path.isabs(args.model_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.model_dir = os.path.abspath(os.path.join(script_dir, args.model_dir))

    safe_name = args.model.replace("/", "_").replace(":", "_")
    model_path = os.path.join(args.model_dir, safe_name)
    os.environ["ONNX_AS_MODELS_DIR"] = args.model_dir

    # =====================================================================
    # AUTOMATED DYNAMIC DOWNLOAD & FLAT FILE EXTRACTION
    # =====================================================================
    if not os.path.isdir(model_path):
        LOG.info("Model directory not found. Initiating dynamic download for %s...", args.model)
        try:
            from huggingface_hub import snapshot_download
            
            # Modern huggingface_hub natively flattens files into local_dir by default
            snapshot_download(
                repo_id=args.model,
                local_dir=model_path
            )
            
            # Strict Post-Download Check: Defensive guardrail to guarantee no lingering symlinks
            import shutil
            for root, _, files in os.walk(model_path):
                for f in files:
                    f_path = os.path.join(root, f)
                    if os.path.islink(f_path):
                        real_src = os.path.realpath(f_path)
                        os.unlink(f_path)
                        shutil.copy2(real_src, f_path)
                        
            LOG.info("Successfully isolated and flattened model assets into: %s", model_path)
        except Exception as dl_err:
            LOG.error("Dynamic download sequencing failed: %s", dl_err)
            sys.exit(1)

    # =====================================================================
    # DYNAMIC CONFIGURATION EXTRACTION FROM ENVIRONMENT VARIABLES
    # =====================================================================
    ENV_THREADS = int(os.environ.get("ORT_NUM_THREADS", "1"))
    ENV_LOG_LEVEL = int(os.environ.get("ORT_LOGGING_LEVEL", "4"))

    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = ENV_LOG_LEVEL
    sess_options.intra_op_num_threads = ENV_THREADS
    sess_options.inter_op_num_threads = ENV_THREADS
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    
    # LXC Namespace Core-Affinity Settings
    sess_options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    sess_options.add_session_config_entry("session.use_per_session_threads", "1")

    # Monkey-Patch: Guarantees the hidden Silero VAD InferenceSession uses identical variables
    _original_InferenceSession_init = ort.InferenceSession.__init__

    def _patched_InferenceSession_init(self, *args, **kwargs):
        if "sess_options" not in kwargs or kwargs["sess_options"] is None:
            if len(args) < 2:  # Explicit positional validation safety check
                kwargs["sess_options"] = sess_options
        _original_InferenceSession_init(self, *args, **kwargs)

    ort.InferenceSession.__init__ = _patched_InferenceSession_init
    # =====================================================================

    providers = (
        ["CPUExecutionProvider"]
        if args.cpu
        else ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

    try:
        LOG.info("Initializing runtime connection for model: %s", args.model)

        model_kwargs = {
            "model": args.model,
            "providers": providers,
            "sess_options": sess_options
        }
        if os.path.isdir(model_path):
            model_kwargs["path"] = model_path

        model = onnx_asr.load_model(**model_kwargs)

        if not args.no_vad:
            LOG.info("Voice Activity Detection (VAD) enabled: loading silero VAD")
            model = model.with_vad(onnx_asr.load_vad("silero"))
        else:
            LOG.info("Voice Activity Detection (VAD) explicitly disabled via CLI parameters")

        # Deep/Recursive hardware verification lookup loop
        active_providers = []
        objects_to_check = [model]
        visited = set()
        
        while objects_to_check:
            obj = objects_to_check.pop(0)
            if id(obj) in visited:
                continue
            visited.add(id(obj))
            
            if isinstance(obj, ort.InferenceSession):
                active_providers = obj.get_providers()
                break
            
            for attr_name in dir(obj):
                if attr_name.startswith("__"):
                    continue
                try:
                    attr_val = getattr(obj, attr_name)
                    if isinstance(attr_val, ort.InferenceSession):
                        active_providers = attr_val.get_providers()
                        break
                    elif hasattr(attr_val, "__dict__") and id(attr_val) not in visited:
                        objects_to_check.append(attr_val)
                except AttributeError:
                    continue
            if active_providers:
                break

        if not active_providers and not args.cpu:
            active_providers = ort.get_available_providers()

        if "CUDAExecutionProvider" in active_providers and not args.cpu:
            LOG.info("Running on GPU (CUDA Accelerated)")
        else:
            LOG.info("Running on CPU")

        LOG.info("Wyoming server listening interface active on: %s", args.uri)

    except Exception as e:
        LOG.error("Model load sequencing initialization aborted: %s", e)
        sys.exit(1)

    meta = extract_model_info(args.model)

    wyoming_info = Info(
        asr=[
            AsrProgram(
                name="onnx-asr",
                description="Lightweight ONNX ASR Engine Container",
                attribution=Attribution(name="Ilya Stupakov", url="https://github.com/istupakov/onnx-asr"),
                installed=True,
                version="1.0.0",
                models=[
                    AsrModel(
                        name=meta["name"],
                        description=meta["description"],
                        attribution=Attribution(name="onnx-asr Runtime", url="https://github.com/istupakov/onnx-asr"),
                        installed=True,
                        languages=meta["languages"],
                        version="1.0.0"
                    )
                ]
            )
        ]
    )

    LOG.info("Wyoming Handshake Registered: %s %s", meta["name"], meta["languages"])

    server = AsyncServer.from_uri(args.uri)
    model_lock = asyncio.Lock()
    await server.run(partial(SimpleAsrEventHandler, wyoming_info, model, model_lock))


if __name__ == "__main__":
    asyncio.run(main())