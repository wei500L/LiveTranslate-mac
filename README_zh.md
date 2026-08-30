# LiveTranslate

[English](README.md) | **中文**

Windows/macOS 实时音频翻译工具。Windows 使用 WASAPI loopback，macOS 使用 ScreenCaptureKit，并支持可选麦克风输入；语音识别后调用 LLM API 翻译，结果显示在透明悬浮字幕窗口上。

适用于看外语视频、直播、语音对话等场景——无需修改播放器，全局音频捕获即开即用。

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)
![平台](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS-0078d4)
![License](https://img.shields.io/badge/License-MIT-green)

## 截图

![LiveTranslate](screenshot/zh.png)

## 安装视频

[![安装演示](https://img.shields.io/badge/Bilibili-安装演示-00A1D6?logo=bilibili)](https://www.bilibili.com/video/BV1K2Awz6Euw) 适用于看外语视频、直播、ASMR等场景，也可以语音输入实时并行翻译多种语音

## 功能特性

- **实时翻译管线**：系统音频 → VAD → ASR → LLM 翻译 → 字幕显示
- **多 ASR 引擎**：faster-whisper、SenseVoice、FunASR Nano、Anime-Whisper、GigaAM（俄语）
- **远程 ASR**：通过 HTTP 把语音识别放到 GPU 机器上跑 —— 见 [REMOTE_ASR.md](REMOTE_ASR.md)
- **兼容任意 OpenAI 格式 API**：DeepSeek、Grok、Qwen、GPT、Ollama、vLLM 等
- **流式翻译显示**：翻译结果逐字实时显示
- **模型独立配置**：流式传输、结构化输出(JSON)、上下文历史、禁用思考
- **麦克风混音**：可选将麦克风输入混合到系统音频一起识别
- **低延迟 VAD**：32ms 音频块 + Silero VAD，自适应静音检测
- **透明悬浮窗**：始终置顶、鼠标穿透、可拖拽，14 种配色主题
- **硬件加速**：Windows 支持 CUDA；Apple Silicon 的 torch ASR 使用 MPS，并自动回退 CPU
- **模型自动管理**：首次启动向导，支持 ModelScope / HuggingFace 双源
- **内置基准测试**：对比翻译模型速度和质量

### Apple Silicon 本地 HY-MT1.5-7B

M5 Pro 等 Apple Silicon 设备可以使用 HY-MT1.5-7B 的 MLX 4-bit 版本进行本地实时翻译。应用会从 ModelScope 临时下载官方 BF16 权重并转换为 MLX 4-bit，转换完成后自动删除 BF16 源文件：

```bash
./start.sh
```

应用会在“翻译”设置中新增 **HY-MT1.5-7B (MLX 4-bit)**。首次使用时点击“准备本地模型”，应用会在隔离环境中安装 MLX、临时下载并转换 BF16 权重，完成后自动删除 BF16 源文件。之后可手动点击“启动本地服务”或“停止本地服务”；切换模型不会自动启动服务，也不会在失败后静默回退到其他模型。应用退出时会释放本程序启动的 MLX 服务。若 8080 端口被其他程序占用，不会强制结束该程序。

首次准备过程会临时下载约 16GB BF16 权重，转换后的 4-bit 模型约占 4GB；转换完成后 BF16 源权重不会保留。M5 Pro 48GB 统一内存适合这一配置。

### GigaAM-v3（俄语 ASR）

LiveTranslate 通过 Transformers 加载官方 [`ai-sage/GigaAM-v3`](https://huggingface.co/ai-sage/GigaAM-v3)
仓库中的 `e2e_rnnt` revision。这是官方 GigaAM-v3 端到端 ASR 变体，会直接输出带标点和规范化的文本，定位为俄语语音识别。当前集成保持既有 ASR worker 协议，接收 16 kHz 音频；在 Apple Silicon 上如果 MPS 加载或算子失败，会回退到 CPU。

当前应用对每个 VAD 分段调用短音频 `transcribe` 接口。根据官方项目说明，该接口适用于最长约 25 秒的音频；需要 pyannote 分段的长音频 `transcribe_longform` 尚未接入 LiveTranslate。模型由应用现有的模型管理器从 Hugging Face 下载，不需要单独 clone 官方项目。

官方资料：[GigaAM-v3 模型](https://huggingface.co/ai-sage/GigaAM-v3) ·
[GigaAM 项目主页](https://github.com/salute-developers/GigaAM) ·
[官方推理说明](https://github.com/salute-developers/GigaAM#model-inference)

## 更新日志

查看 [中文更新日志](i18n/CHANGELOG_zh.md) | [English Changelog](i18n/CHANGELOG_en.md)

## 系统要求

- **操作系统**：Windows 10/11，或 macOS 13+ Apple Silicon（arm64）
- **Python**：3.10–3.12（绿色版免装）
- **GPU**（推荐）：Windows 使用 NVIDIA + CUDA 12.6（RTX 50 系列等 Blackwell 架构需要 CUDA 12.8）；Apple Silicon 的 torch ASR 使用 MPS，faster-whisper 使用 CPU
- **网络**：需要访问翻译 API

## 快速开始

### 绿色版（免装 Python，推荐新手）

从 [Releases](https://github.com/TheDeathDragon/LiveTranslate/releases) 下载 `LiveTranslate-portable-*.zip`，解压后双击 **`start.bat`** 即可。首次运行会自动下载便携版 Python 3.12 并按显卡安装依赖，无需预装任何 Python。

### 从源码安装

```bash
git clone https://github.com/TheDeathDragon/LiveTranslate.git
cd LiveTranslate
```

双击 **`start.bat`** 即可一键安装并启动——首次运行会自动：
1. 检测 Python 3.10–3.12（未安装则通过 winget 自动安装）
2. 创建虚拟环境
3. 检测 NVIDIA 显卡，选择 CUDA / CPU 版 PyTorch
4. 安装全部依赖

之后再次双击 **`start.bat`** 直接启动。macOS 使用 `./start.sh`，行为相同。

翻译服务地址和密钥可通过环境变量一次配置，应用会自动补齐 OpenAI 兼容接口的 `/v1`：

```bash
export LIVETRANSLATE_API_BASE=http://127.0.0.1:1234
export LIVETRANSLATE_API_KEY=你的密钥
export LIVETRANSLATE_MODEL=hunyuan-mt-chimera-7b
# 可选：修改本地 HY-MT 服务端口（默认 8080）
export LIVETRANSLATE_MLX_PORT=8080
./start.sh
```

更新时双击 **`update.bat`**——自动拉取最新代码并更新依赖（未安装 Git 会通过 winget 自动安装）。

<details>
<summary>手动安装</summary>

```bash
python -m venv .venv
.venv\Scripts\activate

# PyTorch（三选一）
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126  # CUDA
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128  # CUDA（RTX 50 系列）
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu    # 仅 CPU

# 依赖
pip install -r requirements.txt

# 启动
.venv\Scripts\python.exe main.py
```

</details>

### macOS（Apple Silicon）

请使用原生 arm64 的 Python 3.10–3.12。安装脚本会拒绝 Rosetta/x86_64 Python：

```bash
./install.sh
./start.sh
```

macOS 系统音频通过 ScreenCaptureKit 捕获，需要授予“屏幕录制”权限；麦克风混音需要“麦克风”权限。修改权限后通常需要重启应用。SCK 捕获主显示器系统音频，不提供 Windows 风格的 WASAPI 设备名。faster-whisper/CTranslate2 在 CPU（int8）上运行，支持的 torch ASR 使用 MPS。

## 首次使用

1. 弹出设置向导——选择下载源（ModelScope 适合国内，HuggingFace 适合海外）和缓存路径
2. 自动下载 Silero VAD + SenseVoice 模型（约 1GB）
3. 下载完成后进入主界面

## 配置翻译 API

设置 → 翻译标签页：

| 参数 | 示例 |
|------|------|
| API Base | `https://api.deepseek.com/v1` |
| API Key | 你的密钥 |
| Model | `deepseek-chat` |
| 代理 | `none` / `system` / 自定义地址 |

## 架构

```
Audio (WASAPI/SCK，32ms) → VAD (Silero) → ASR → LLM Translation → Overlay
         ↑ 可选麦克风混音
```

```
main.py                 主入口，管线编排
├── audio_capture.py    平台音频分发（WASAPI/SCK/CoreAudio）
├── vad_processor.py    Silero VAD
├── asr_engine.py       faster-whisper 后端
├── asr_funasr.py       统一 FunASR 模型选择后端
├── asr_sensevoice.py   SenseVoice 后端
├── asr_funasr_nano.py  FunASR Nano 后端
├── asr_anime_whisper.py Anime-Whisper 后端 (日语动画/Galgame)
├── asr_gigaam.py        GigaAM-v3 e2e_rnnt 后端（俄语，MPS/CPU）
├── asr_remote.py        远程 Whisper 客户端 (→ asr_server.py, 见 REMOTE_ASR.md)
├── translator.py       OpenAI 兼容翻译客户端 (流式/JSON/上下文)
├── model_manager.py    模型下载与缓存管理
├── subtitle_overlay.py PyQt6 透明悬浮窗
├── control_panel.py    设置面板 UI (8 个页面，含会议记录)
├── transcript_writer.py 会议记录：每场生成文本 + Markdown + 元数据
├── dialogs.py          设置向导、下载、模型配置对话框
├── benchmark.py        翻译基准测试
└── debug_pipeline.py   诊断工具：把音频文件喂进真实管线
```

### 排查问题

字幕或翻译不出来时，不要靠猜 —— 用真实管线回放一个音频文件：

```bash
.venv/bin/python debug_pipeline.py --audio sample.mp3              # 两条链路
.venv/bin/python debug_pipeline.py --audio sample.mp3 --no-translate  # 只测识别
```

它用你实际的设置、ASR worker 和翻译模型，逐阶段打印产出；**某一环节被要求工作
却什么都没产出会判定为失败**，而不是安静通过。完整 DEBUG 日志在
`logs/diagnostic_*.log`，诊断转录写到 `transcripts/diagnostic/`，不会混进你的会议记录。

## 致谢

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — 基于 CTranslate2 的 Whisper 推理
- [FunASR](https://github.com/modelscope/FunASR) — SenseVoice / Fun-ASR-Nano
- [Anime-Whisper](https://huggingface.co/litagin/anime-whisper) — 日语动画/Galgame 专用 ASR
- [Silero VAD](https://github.com/snakers4/silero-vad) — 语音活动检测

## Star History

<a href="https://www.star-history.com/?repos=TheDeathDragon%2FLiveTranslate&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/image?repos=TheDeathDragon/LiveTranslate&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/image?repos=TheDeathDragon/LiveTranslate&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/image?repos=TheDeathDragon/LiveTranslate&type=date&legend=top-left" />
 </picture>
</a>

## 许可证

[MIT License](LICENSE)
