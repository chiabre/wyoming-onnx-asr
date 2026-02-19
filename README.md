# 🎙️ Wyoming ONNX ASR (GPU/CPU)

[Wyoming Protocol](https://github.com/rhasspy/wyoming) server for high-speed local Speech-to-Text using [onnx-asr](https://github.com/istupakov/onnx-asr) optimized for Home Assistant voice pipelines.

## ⚡ Hardware Selection

This repository is hardware-intelligent and distro-aware. During setup, it detects your OS (Debian/Ubuntu) and GPU status to automate the CUDA 12.6 & cuDNN 9.x installation.

| Feature | 🟢 GPU Mode (Recommended) | 🔵 CPU Mode |
| :--- | :--- | :--- |
| **Performance** | Instant (sub-100ms response) | Fast (~500ms - 2s response) |
| **Requirements** | NVIDIA GPU + Driver 550+ | Any modern x86_64 / ARM64 CPU |

---

## ⚙️ Installation

### 1. Setup the Environment
The setup script prepares your system by installing dependencies and configuring the environment. 
*Note: Models are downloaded automatically on the first run of the service.*

```bash
git clone https://github.com/chiabre/wyoming-onnx-asr.git
cd wyoming-onnx-asr
chmod +x script/setup
./script/setup
```

> [!IMPORTANT]
> **GPU Users:** After the setup script finishes, you **must** run the command below (or restart your terminal) to activate the new CUDA paths in your current session:
> ```bash
> source ~/.bashrc
> ```

## 2. Choose Your Model
The default model is `istupakov/parakeet-tdt-0.6b-v2-onnx`. To switch models, use the `--model` parameter with the "Model Repo ID" from the table below.

### ⚙️ Supported Models & Performance

This repository supports 6 optimized ONNX models. Use the table below to select the `MODEL_ID` that matches your hardware and accuracy needs.

| Recommendation | Model Repo ID | Accuracy (WER) | Speed (RTFx) | RAM | Released | Description |
|----------------|---------------|----------------|--------------|-----|----------|-------------|
| 🏆 **Default** | `istupakov/parakeet-tdt-0.6b-v2-onnx` | ~6% | Very High | ~2GB | May 2025 |  Best for English. Ultra-fast TDT architecture. |
| 🌍 **Multi** | `istupakov/parakeet-tdt-0.6b-v3-onnx` | ~7% | Very High | ~2GB | Aug 2025 | Best Multilingual. Optimized for 25 European languages.|
| 🧠 **Robust** | `istupakov/canary-1b-v2-onnx` | ~6.5% | Medium | ~8GB | Aug 2025 | Highly robust; requires significant RAM/VRAM. |
| 🌍 **Universal** | `onnx-community/whisper-large-v3-turbo` | ~7–10% | High | ~6GB | Oct 2024 | Broadest coverage (99+ languages). Autoregressive decoding. |
| ⚡ **Edge** | `istupakov/canary-180m-flash-onnx` | ~9% | High | ~1GB | Oct 2024 | Lowest latency for RPi or low-resource hardware. |
| ⚖️ **Balanced** | `istupakov/whisper-base-onnx` | ~12% | Moderate | ~500MB | May 2024 | Stable general-purpose fallback. |

> [!WARNING]
> **Memory Usage:** The `Robust` (Canary 1B) model requires a minimum of **8GB RAM**. If your service crashes with a "Killed" message, switch to a `Default` or `Edge` model.

## 🚀 Running the Service

### Manual Execution
The run script uses the virtual environment and defaults to the model defined in `script/run`

```bash
# 1. Standard Startup (Recommended)
./script/run

# 2. Forced CPU Mode
./script/run --cpu

# 3. Development/Troubleshooting
./script/run --no-vad --debug

# 4. Custom Model or Port
./script/run --uri tcp://0.0.0.0:10305 --model istupakov/parakeet-tdt-0.6b-v3-onnx
```

#### ⚙️ Configuration Options

These parameters allow you to configure the Wyoming ONNX ASR server. You can pass them as command-line arguments when starting the service.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| **`--model`** | Optional | `istupakov/parakeet-tdt-0.6b-v2-onnx` | The Hugging Face model repo ID |
| **`--model-dir`** | Optional | `data/models` | The parent directory where your model folders are stored. |
| **`--uri`** | Optional | `tcp://0.0.0.0:10300` | The address and port for the Wyoming server to listen on. |
| **`--no-vad`** | Flag | `False` | **Disable** Silero VAD. VAD is ON by default to filter background noise. |
| **`--cpu`** | Flag | `False` | Force CPU inference even if an NVIDIA GPU is detected. |
| **`--debug`** | Flag | `False` | Enable verbose logging for troubleshooting. |

### Systemd Deployment
To run this as a persistent background service that starts with your machine:

```bash
chmod +x script/install-service
./script/install-service
```

To change the model after installation, edit `/etc/systemd/system/wyoming-onnx-asr.service` add  `--model YOUR_MODEL` in the `ExecStart` line, then run `sudo systemctl daemon-reload && sudo systemctl restart wyoming-onnx-asr`.

#### Manage the service:
- Logs: `journalctl -u wyoming-onnx-asr -f`
- Restart: `sudo systemctl restart wyoming-onnx-asr`
