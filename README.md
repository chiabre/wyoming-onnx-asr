# 🎙️ Wyoming ONNX ASR (GPU/CPU)

[Wyoming Protocol](https://github.com/rhasspy/wyoming) server for high-speed local Speech-to-Text using [onnx-asr](https://github.com/istupakov/onnx-asr), optimized for Home Assistant voice pipelines.

---

## ⚡ Hardware Selection

This repository is hardware-intelligent and distro-aware. During setup, it detects your OS (Debian/Ubuntu) and GPU status to automate CUDA & cuDNN installation.

| Feature | 🟢 GPU Mode (Recommended) | 🔵 CPU Mode |
| :--- | :--- | :--- |
| **Performance** | Near real-time (<100–300ms) | Fast (~500ms – 2s) |
| **Requirements** | NVIDIA GPU (CUDA) | Any modern CPU |

---

## ⚙️ Installation

```bash
git clone https://github.com/chiabre/wyoming-onnx-asr.git
cd wyoming-onnx-asr
chmod +x script/setup
./script/setup
```

> [!IMPORTANT]
> **GPU Users:** reload your environment after setup:
> 
> ```bash
> source ~/.bashrc
> ```

---

## 🧠 Model Selection
The default model is `istupakov/parakeet-tdt-0.6b-v2-onnx`. To switch models, use the `--model` parameter with the "Model Repo ID" from the table below.

### ⚙️ Supported Models & Performance

You can use short aliases instead of full Hugging Face IDs.

| Alias              | Model                                 | Best For               | Notes                       |
| ------------------ | ------------------------------------- | ---------------------- | --------------------------- |
| 🏆 `parakeet-v3`   | `istupakov/parakeet-tdt-0.6b-v3-onnx` | Default / multilingual | Fast + accurate             |
| 🇺🇸 `parakeet-v2` | `istupakov/parakeet-tdt-0.6b-v2-onnx` | English-only           | Slightly better EN accuracy |
| 🧠 `canary`        | `istupakov/canary-1b-v2-onnx`         | Accuracy-first         | Slower, higher RAM          |

---

## 🚀 Running the Service

### Basic Usage

```bash
./script/run
```

### Examples

```bash
# Default (recommended)
./script/run --model parakeet-v3

# English optimized
./script/run --model parakeet-v2

# Accuracy-focused
./script/run --model canary

# Force CPU
./script/run --cpu

# Debug mode
./script/run --debug

# Custom port
./script/run --uri tcp://0.0.0.0:10305
```

---

## ⚙️ Configuration Options

These parameters allow you to configure the Wyoming ONNX ASR server. You can pass them as command-line arguments when starting the service.

| Parameter         | Type     | Default               | Description                    |
| :---------------- | :------- | :-------------------- | :----------------------------- |
| `--model`         | Optional | `parakeet-v3`         | Model alias or full HF repo ID |
| `--model-dir`     | Optional | `data/models`         | Model storage directory        |
| `--uri`           | Optional | `tcp://0.0.0.0:10300` | Wyoming server address         |
| `--cpu`           | Flag     | `False`               | Force CPU inference            |
| `--debug`         | Flag     | `False`               | Enable verbose logging         |
| `--threads`       | Optional | `1`                   | Override ONNX thread count     |
| `--ort-log-level` | Optional | `3`                   | ONNX logging level (0–4)       |

---

## 🧠 ONNX Runtime Tuning
This version uses a centralized runtime configuration for:
- Threading
- Logging
- Execution mode

### Examples

```bash
# Increase CPU parallelism
./script/run --threads 2

# Silence ONNX logs
./script/run --ort-log-level 4
```
---

## 🧩 Systemd Deployment
To run this as a persistent background service that starts with your machine:

```bash
./script/install-service
```

To change the model:

```bash
nano /etc/systemd/system/wyoming-onnx-asr.service
```

Update:

```bash
--model parakeet-v3
```
Then:
```bash
systemctl daemon-reload
systemctl restart wyoming-onnx-asr
```
---

## 🔧 Service Management
- Logs:

```bash
journalctl -u wyoming-onnx-asr -f
```

- Restart:
  
```bash
systemctl restart wyoming-onnx-asr
```
---

## 🧠 Notes

- Use `parakeet-v3` for best overall performance
- Use `parakeet-v2` for English-only setups
- Use `canary` when accuracy matters more than latency
- Lower thread count may help on low-power CPUs
