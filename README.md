# LiveTranslate

**English** | [中文](README_zh.md)

Real-time audio translation for Windows and macOS. Captures system audio (WASAPI loopback on Windows, ScreenCaptureKit on macOS) plus optional microphone input, runs ASR, translates via LLM API, and displays results in a transparent overlay.

Works with any system audio — videos, livestreams, voice chat. No player modifications needed.

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)
![Platforms](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS-0078d4)
![License](https://img.shields.io/badge/License-MIT-green)

## Screenshot

![LiveTranslate](screenshot/en.png)

## Video

[![Install & Demo](https://img.shields.io/badge/Bilibili-Install%20%26%20Demo-00A1D6?logo=bilibili)](https://www.bilibili.com/video/BV1K2Awz6Euw)

## Features

- **Real-time pipeline**: System audio → VAD → ASR → LLM translation → overlay
- **Multiple ASR engines**: faster-whisper, SenseVoice, FunASR Nano, Anime-Whisper, GigaAM (Russian)
- **Remote ASR**: offload speech recognition to a GPU machine over HTTP — see [REMOTE_ASR.md](REMOTE_ASR.md)
- **Any OpenAI-compatible API**: DeepSeek, Grok, Qwen, GPT, Ollama, vLLM, etc.
- **Streaming translation display**: Real-time character-by-character translation output
- **Per-model settings**: Streaming, structured output (JSON), context history, disable thinking
- **Microphone mix-in**: Optionally mix microphone input with system audio for ASR
- **Low-latency VAD**: 32ms chunks + Silero VAD with adaptive silence detection
- **Transparent overlay**: Always-on-top, click-through, draggable, 14 color themes
- **Hardware acceleration**: CUDA on supported Windows setups; MPS on Apple Silicon for torch ASR, with CPU fallback
- **Auto model management**: Setup wizard, ModelScope / HuggingFace dual sources
- **Built-in benchmark**: Compare translation model speed and quality

### Local HY-MT1.5-7B on Apple Silicon

Apple Silicon Macs can prepare the official ModelScope weights as an MLX 4-bit local translation model:

```bash
./start.sh
```

The app adds **HY-MT1.5-7B (MLX 4-bit)** to the Translation settings. Click **Prepare Local Model** once to install the isolated MLX runtime and convert the temporary BF16 weights; the BF16 source is removed automatically. Then use **Start Local Service** or **Stop Local Service**. Selecting a model never starts a service or silently falls back to another model. The app stops its own MLX service on exit.

### GigaAM-v3 (Russian ASR)

LiveTranslate loads the official [`ai-sage/GigaAM-v3`](https://huggingface.co/ai-sage/GigaAM-v3)
checkpoint through Transformers with the `e2e_rnnt` revision. This is the official
end-to-end GigaAM-v3 ASR variant: it returns punctuated, normalized text and is
intended for Russian speech. The integration keeps the existing ASR worker contract,
accepts 16 kHz audio, and falls back from MPS to CPU if model loading or an operator
fails on Apple Silicon.

The current in-app path calls the model's short-audio `transcribe` API for each VAD
segment. Per the official project documentation, that API is intended for audio up to
25 seconds; long-form `transcribe_longform` with pyannote segmentation is not wired
into LiveTranslate yet. The model is downloaded from Hugging Face by the normal model
manager, so cloning the upstream repository is not required for the app.

Official resources: [GigaAM-v3 model](https://huggingface.co/ai-sage/GigaAM-v3) ·
[GigaAM project](https://github.com/salute-developers/GigaAM) ·
[upstream usage guide](https://github.com/salute-developers/GigaAM#model-inference)

## Changelog

See [English Changelog](i18n/CHANGELOG_en.md) | [中文更新日志](i18n/CHANGELOG_zh.md)

## Requirements

- **OS**: Windows 10/11, or macOS 13+ on Apple Silicon (arm64)
- **Python**: 3.10–3.12 (or use the portable build)
- **GPU** (recommended): NVIDIA + CUDA 12.6 on Windows (Blackwell GPUs like RTX 50xx require CUDA 12.8); Apple Silicon uses MPS for torch ASR and CPU for faster-whisper
- **Network**: Access to a translation API

## Quick Start

### Portable build (no Python required, recommended for non-developers)

Download `LiveTranslate-portable-*.zip` from [Releases](https://github.com/TheDeathDragon/LiveTranslate/releases), unzip, and double-click **`start.bat`**. The first run auto-downloads a portable Python 3.12 and installs GPU-aware dependencies — no Python installation needed.

### From source

```bash
git clone https://github.com/TheDeathDragon/LiveTranslate.git
cd LiveTranslate
```

Double-click **`start.bat`** to install and launch in one step. On the first run it will:
1. Detect Python 3.10–3.12 (auto-install via winget if missing)
2. Create a virtual environment
3. Auto-detect NVIDIA GPU and let you choose CUDA / CPU PyTorch
4. Install all dependencies

After that, double-click **`start.bat`** to launch directly. On macOS, use `./start.sh`.

You can configure an OpenAI-compatible translation endpoint without editing YAML:

```bash
export LIVETRANSLATE_API_BASE=http://127.0.0.1:1234
export LIVETRANSLATE_API_KEY=your-key
export LIVETRANSLATE_MODEL=hunyuan-mt-chimera-7b
# Optional: change the local HY-MT service port (default 8080)
export LIVETRANSLATE_MLX_PORT=8080
./start.sh
```

To update, double-click **`update.bat`** — it will pull the latest code and update dependencies (auto-installs Git via winget if missing).

<details>
<summary>Manual install</summary>

```bash
python -m venv .venv
.venv\Scripts\activate

# PyTorch (choose one)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126  # CUDA
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128  # CUDA (RTX 50xx)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu    # CPU only

# Dependencies
pip install -r requirements.txt

# Launch
.venv\Scripts\python.exe main.py
```

</details>

### macOS (Apple Silicon)

Use a native arm64 Python 3.10–3.12 environment. Rosetta/x86_64 Python is rejected by
the installer:

```bash
./install.sh
./start.sh
```

macOS system audio uses ScreenCaptureKit and requires Screen Recording permission;
microphone mixing requires Microphone permission. macOS may require restarting the
app after changing either permission. The SCK path captures the selected main display
and does not expose Windows-style loopback device names. Faster-whisper/CTranslate2
runs on CPU (int8); MPS is used by torch-based ASR engines when supported.

To launch it like a native Mac app instead of from a terminal:

```bash
./build_mac_app.sh --install   # builds LiveTranslate.app into /Applications
```

It then starts from Launchpad or Spotlight (`open -a LiveTranslate`). The bundle's
launcher is baked against the current project path — rerun the script after moving
the project. The first .app launch asks for Microphone permission again, because
macOS attributes the permission to the app bundle rather than to the terminal.

## First Launch

1. Setup wizard appears — choose download source (ModelScope / HuggingFace) and cache path
2. Silero VAD + SenseVoice models download automatically (~1GB)
3. Main UI appears when ready

## Translation API

Settings → Translation tab:

| Parameter | Example |
|-----------|---------|
| API Base | `https://api.deepseek.com/v1` |
| API Key | Your key |
| Model | `deepseek-chat` |
| Proxy | `none` / `system` / custom URL |

## Architecture

```
Audio (WASAPI/SCK, 32ms) → VAD (Silero) → ASR → LLM Translation → Overlay
         ↑ optional mic mix-in
```

```
main.py                 Entry point & pipeline
├── audio_capture.py    Platform audio dispatcher (WASAPI/SCK/CoreAudio)
├── vad_processor.py    Silero VAD
├── asr_engine.py       faster-whisper backend
├── asr_funasr.py       Unified FunASR model selector backend
├── asr_sensevoice.py   SenseVoice backend
├── asr_funasr_nano.py  FunASR Nano backend
├── asr_anime_whisper.py Anime-Whisper backend (ja anime/galgame)
├── asr_gigaam.py        GigaAM-v3 e2e_rnnt backend (Russian, MPS/CPU)
├── asr_remote.py        Remote Whisper client (→ asr_server.py, see REMOTE_ASR.md)
├── translator.py       OpenAI-compatible client (streaming, JSON schema, context)
├── model_manager.py    Model download & cache
├── subtitle_overlay.py PyQt6 overlay
├── control_panel.py    Settings UI (8 pages, incl. Meeting Records)
├── transcript_writer.py Meeting record: text + Markdown + metadata per session
├── dialogs.py          Wizard, download & model config dialogs
├── benchmark.py        Translation benchmark
└── debug_pipeline.py   Diagnostic: replay a file through the real pipeline
```

### Troubleshooting

If subtitles or translations stop appearing, replay an audio file through the
real pipeline instead of guessing:

```bash
.venv/bin/python debug_pipeline.py --audio sample.mp3              # both chains
.venv/bin/python debug_pipeline.py --audio sample.mp3 --no-translate  # ASR only
```

It uses your actual settings, ASR worker and translation model, prints what each
stage produced, and fails the run if a stage was asked to do work and produced
nothing. The full DEBUG log lands in `logs/diagnostic_*.log`, and its transcripts
go to `transcripts/diagnostic/` so they stay out of your meeting records.

## Acknowledgements

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — Whisper inference via CTranslate2
- [FunASR](https://github.com/modelscope/FunASR) — SenseVoice / Fun-ASR-Nano
- [Anime-Whisper](https://huggingface.co/litagin/anime-whisper) — Japanese anime/galgame ASR
- [Silero VAD](https://github.com/snakers4/silero-vad) — Voice activity detection

## Star History

<a href="https://www.star-history.com/?repos=TheDeathDragon%2FLiveTranslate&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=TheDeathDragon/LiveTranslate&type=date&theme=dark&legend=top-left&sealed_token=0t-kzcN9leqL-yaHdKcBdHLdLlE6NNHa48RpsLvUM3u2fOiZOYqWfjgplXxAtjk1ZJciSAbzI3gZ4PQqqHrOv4abM1CpOomUymVX6J1zPN-3Ygu0-Xr8Kpj3Xt8jWS05B4tTpuNSmoYqyHipPvKC7lxGfLcOF_zctjqvOka-j9gYWct0oQyJnGjdcZxY" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=TheDeathDragon/LiveTranslate&type=date&legend=top-left&sealed_token=0t-kzcN9leqL-yaHdKcBdHLdLlE6NNHa48RpsLvUM3u2fOiZOYqWfjgplXxAtjk1ZJciSAbzI3gZ4PQqqHrOv4abM1CpOomUymVX6J1zPN-3Ygu0-Xr8Kpj3Xt8jWS05B4tTpuNSmoYqyHipPvKC7lxGfLcOF_zctjqvOka-j9gYWct0oQyJnGjdcZxY" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=TheDeathDragon/LiveTranslate&type=date&legend=top-left&sealed_token=0t-kzcN9leqL-yaHdKcBdHLdLlE6NNHa48RpsLvUM3u2fOiZOYqWfjgplXxAtjk1ZJciSAbzI3gZ4PQqqHrOv4abM1CpOomUymVX6J1zPN-3Ygu0-Xr8Kpj3Xt8jWS05B4tTpuNSmoYqyHipPvKC7lxGfLcOF_zctjqvOka-j9gYWct0oQyJnGjdcZxY" />
 </picture>
</a>

## License

[MIT License](LICENSE)
