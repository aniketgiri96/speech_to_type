# LFM Speech-to-Type

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![OS: macOS](https://img.shields.io/badge/OS-macOS-blue?logo=apple)](https://apple.com)
[![OS: Windows](https://img.shields.io/badge/OS-Windows-experimental?logo=windows&color=orange)](https://microsoft.com)

A local, privacy-first alternative to WhisperFlow — powered by Liquid AI's **LFM2.5-Audio-1.5B** model running entirely on your machine.

Press a hotkey → speak → text appears in whatever input field is focused. No cloud, no subscription, no data leaves your computer.

---

## 🚦 Status Indicators

| State | Tray Icon | Description |
| :--- | :--- | :--- |
| **Idle / Ready** | 🟢 Green mic | Waiting for hotkey |
| **Recording** | 🔴 Red mic | Capturing audio from microphone |
| **Transcribing** | 🟡 Yellow mic | Processing audio through LFM model |

---

## 🛠️ Platform Support

| Platform | Status | Preferred Method |
| :--- | :--- | :--- |
| **macOS** | ✅ Tested / Stable | Docker (Server) + Python (Client) |
| **Windows** | ⚠️ Experimental | Manual Build (`llama.cpp`) |

---

## 🚀 Quick Start (macOS)

macOS is the primary tested platform. We recommend using Docker for the heavy lifting.

### 1. Requirements
- **Docker Desktop** installed and running.
- **Python 3.11+** installed.
- **Microphone** access.

### 2. Setup
```bash
# Clone the repository
git clone https://github.com/aniketgiri96/speech_to_type
cd speech_to_type

# Install Python dependencies
pip install -r requirements.txt

# Start the LFM Audio Server
docker-compose up -d --build
```

### 3. Run
```bash
python speech_to_type.py --no-server
```

> [!IMPORTANT]
> The first time you use it, macOS will ask for **Accessibility permissions** (to simulate `Cmd+V`). Please grant them to your Terminal/IDE in *System Settings -> Privacy & Security -> Accessibility*.

---

## 🪟 Windows Setup (Experimental)

Windows support is currently in development. These steps require compilation of the inference engine.

### 1. Requirements
- **Visual Studio 2022** (Community edition) with C++ development tools.
- **CMake** (`pip install cmake`).
- **Python 3.11+**.

### 2. Building the Server
```bash
# Clone llama.cpp with LFM support
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
git fetch origin pull/18641/head:pr-18641
git checkout pr-18641

# Configure and Build
cmake -B build -G "Visual Studio 17 2022" -A x64 -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=OFF
cmake --build build --config Release --target llama-liquid-audio-server -j 8
```

### 3. Run
You can use the provided batch file once setup is complete:
```cmd
start.bat
```

---

## 📦 Model Files (Required)

The LFM model is gated. You must accept the license on HuggingFace to download.

1.  **Accept License**: [LiquidAI/LFM2.5-Audio-1.5B-GGUF](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-GGUF)
2.  **Download**:
```bash
python -c "
from huggingface_hub import hf_hub_download
from pathlib import Path

repo = 'LiquidAI/LFM2.5-Audio-1.5B-GGUF'
dest = Path('models')
dest.mkdir(exist_ok=True)

files = [
    'LFM2.5-Audio-1.5B-Q4_0.gguf',
    'mmproj-LFM2.5-Audio-1.5B-Q4_0.gguf',
    'vocoder-LFM2.5-Audio-1.5B-Q4_0.gguf',
    'tokenizer-LFM2.5-Audio-1.5B-Q4_0.gguf',
]
for f in files: hf_hub_download(repo, f, local_dir=dest)
"
```

---

## ⌨️ Configuration

Edit `config.ini` (auto-created on first run) to customize your experience:

```ini
[settings]
hotkey = <ctrl>+<alt>+<space>
port = 8142
threads = 4
```

### Available CLI Flags
- `--hotkey`: Override the hotkey (e.g., `--hotkey f9`)
- `--no-server`: Use this if the server is already running (e.g., in Docker)
- `--threads`: Number of CPU threads for inference

---

## 🧩 How it Works

1.  **Hotkey Pressed**: `pynput` catches the trigger and starts `sounddevice` recording.
2.  **Hotkey Released**: Audio is base64 encoded and sent to the local LFM server.
3.  **Inference**: The Liquid AI LFM model performs Speech-To-Text locally.
4.  **Typing**: The app restores your original clipboard and "pastes" the text via system simulation.

---

## 🛠️ Troubleshooting

- **Server Timeout**: The first load takes ~60s. Check `server.log` for details.
- **Transcription Accuracy**: Ensure your microphone is clear; the model works best at 16kHz.
- **Windows Build Errors**: Ensure you have the latest MSVC 14.39+ installed.

---

## 📜 Credits & License

- **Model**: [Liquid AI](https://liquid.ai) (LFM2.5-Audio-1.5B)
- **Engine**: [llama.cpp](https://github.com/ggml-org/llama.cpp)
- **License**: MIT
