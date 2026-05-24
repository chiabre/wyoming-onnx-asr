# 🎙️ Wyoming ONNX ASR (GPU/CPU)

[Wyoming Protocol](https://github.com/rhasspy/wyoming) server for high-speed local Speech-to-Text using [onnx-asr](https://github.com/istupakov/onnx-asr), optimized for Home Assistant voice pipelines.

---

## ⚡ Hardware 

The setip script detects your OS (Debian/Ubuntu) and NVIDIA GPU and driver status to automate CUDA 13.1 & cuDNN 9 installation.

| Feature | 🟢 GPU Mode (Recommended) | 🔵 CPU Mode |
| :--- | :--- | :--- |
| **Performance** | Near real-time (<100–300ms) | Fast (~500ms – 2s) |
| **Requirements** | NVIDIA GPU + NVIDIA Driver>=580.65.06 | Any modern CPU |

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
Models are dynamically downloaded directly from Hugging Face on first startup.

The default model is `istupakov/parakeet-tdt-0.6b-v2-onnx`. To switch models, use the `--model` parameter.

### ⚙️ Supported Models 

| Target Language | Hugging Face Repository ID (Use this for `--model`) | Purpose / Strengths |
| :--- | :--- | :--- |
| 🌐 **Multilingual** | `istupakov/parakeet-tdt-0.6b-v3-onnx` | Outstanding accuracy across 25 languages. Uses the modern Token-and-Duration Transducer structure. |
| 🇺🇸 **English Only** | `istupakov/parakeet-tdt-0.6b-v2-onnx` | Legacy English flagship. Superior accuracy if your smart home interactions are strictly in English. |
| 🇷🇺 **Russian** | `istupakov/gigaam-v3-onnx` | Sber GigaAM v3 foundation model. Exceptional tracking for Russian language voice intents. |

---

### 🔗 Full Ecosystem Directory
For the complete list of available variants (including FastConformer, Whisper exports, and specific Russian CTC/RNNT decoders), you can view the upstream model repository directly on the Hugging Face Hub:

👉 **[Browse All Available ONNX ASR Models on Hugging Face](https://huggingface.co/istupakov)**
---

## 🚀 Running the Service

### Basic Usage

```bash
./script/run
```

### Examples

```bash
# Run v3 Multilingual Model
./script/run --model istupakov/parakeet-tdt-0.6b-v3-onnx

# Run with custom local directory mapping
./script/run --model-dir data/models

# Run with custom environment overrides for threading & log silencing
ORT_NUM_THREADS=2 ORT_LOGGING_LEVEL=4 ./script/run --model istupakov/parakeet-tdt-0.6b-v3-onnx

# Force CPU Execution Provider
./script/run --cpu

# Explicitly disable Voice Activity Detection (VAD) parsing
./script/run --no-vad

# Custom Binding Port
./script/run --uri tcp://0.0.0.0:10305
```

---

## ⚙️ Configuration Options

These parameters allow you to configure the Wyoming ONNX ASR server. You can pass them as command-line arguments when starting the service.

| Parameter         | Type     | Default                                       | Description                    |
| :---------------- | :------- | :-------------------------------------------- | :----------------------------- |
| `--model`         | Optional | `istupakov/parakeet-tdt-0.6b-v2-onnx`         | Model alias or full HF repo ID |
| `--model-dir`     | Optional | `data/models`                                 | Model storage directory        |
| `--uri`           | Optional | `tcp://0.0.0.0:10300`                         | Wyoming server address         |
| `--cpu`           | Flag     | `False`                                       | Force CPU inference            |
| `--no-vad`        | Flag     | `False`                                       | Force CPU inference            |
| `--debug`         | Flag     | `False`                                       | Enable verbose logging         |

### Centralized Environment Tuning
To control internal ONNX Runtime concurrency allocations safely without compilation flags (vital for preventing CPU thread contention inside Proxmox LXC containers), expose these environment parameters to the process:

| Environment Variable | Default | Valid Range  | Functional Context                                                                                                                   |
| :------------------- | :------ | :----------- | :----------------------------------------------------------------------------------------------------------------------------------- |
| `ORT_NUM_THREADS`    | 1       | Integer (1+) | Maps absolute execution limits for intra_op and inter_op thread pools (Monkey-patched across both ASR and Silero VAD session states).|
| `ORT_LOGGING_LEVEL`  | 4       | 0 to 4       | Maps internal ORT logging severity (0 = Verbose, 4 = Silent). Keeps systemd journals clean.                                          |
---


## 🧩 Systemd Deployment
To run this as a persistent background service that starts with your machine:

```bash
./script/install-service
```

To switch models or tune processing footprints under systemd control, modify the runtime unit directly:

```bash
nano /etc/systemd/system/wyoming-onnx-asr.service
```

Ensure your service profile explicitly contains the targeted environment overrides and execution parameters:

```bash
[Service]
Type=simple
User=luca
Group=luca
WorkingDirectory=/opt/wyoming-onnx-asr
Environment="ORT_NUM_THREADS=1"
Environment="ORT_LOGGING_LEVEL=4"
ExecStart=/opt/wyoming-onnx-asr/.venv/bin/python3 wyoming_onnx_asr.py \
  --model istupakov/parakeet-tdt-0.6b-v3-onnx \
  --model-dir data/models \
  --uri tcp://0.0.0.0:10300
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

## 🧠 Operational Notes

### Cache Deletion Fallback Behaviour
If you manually wipe your local storage directory mapping (`data/models`), the code structure fallback layer redirects parsing execution arguments down to the default Hugging Face global storage contexts (`~/.cache/huggingface/hub`). 
* While **v2** can execute out of that shared global space natively because its weights are baked completely flat inside a single standalone `.onnx` binary file, **v3 will fail initialization** under recent ONNX runtimes unless kept inside our isolated, un-symlinked local folder layout.

### Performance Tuning Tips
* **LXC Resource Sharing:** Always match `ORT_NUM_THREADS` precisely to your container core quota maps. Leaving thread-spinning unchecked can lead to thread starvation loops across neighboring virtualization stacks.
* **Co-Location Bounds:** If running high-intensity Local LLM inference pools (e.g., Ollama running large Qwen profiles) concurrently with this service on a single GPU (like an RTX 3060 12GB), keep thread ceilings low to avoid scheduling collisions.

### Pipeline Reality (Important)
* Home Assistant voice pipelines are not fully streaming. End-of-speech detection optimization in your voice pipeline assistant configurations often has a more dramatic impact on perceived system latency than raw hardware inference speed.