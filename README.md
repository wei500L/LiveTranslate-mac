# LiveTranslate for macOS

[English (upstream)](https://github.com/TheDeathDragon/LiveTranslate/blob/main/README.md) · 本仓库是 [TheDeathDragon/LiveTranslate](https://github.com/TheDeathDragon/LiveTranslate) 的 **macOS(Apple Silicon)分支**,本文档只覆盖 macOS 侧的安装、使用与排障。

实时音频翻译工具:通过 **ScreenCaptureKit** 捕获系统音频(可混入麦克风),经 VAD 切分、ASR 识别,再由任意 OpenAI 兼容 API(或本地 MLX 模型)翻译,结果展示在透明悬浮字幕上。适用于浏览器视频、直播、Zoom 会议、本地播放器——任何发出声音的东西,无需改动播放器本身。

![Python 3.10–3.12](https://img.shields.io/badge/Python-3.10–3.12-blue)
![macOS 13+](https://img.shields.io/badge/macOS-13%2B%20Apple%20Silicon-0078d4)
![License](https://img.shields.io/badge/License-MIT-green)

## 功能特性

- **实时流水线**:系统音频 → VAD(32ms 帧,Silero)→ ASR → LLM 翻译 → 悬浮字幕
- **系统音频采集**:ScreenCaptureKit 捕获所选主屏的系统声音;不需要时可在设置中切到"仅麦克风"
- **多种 ASR 引擎**:faster-whisper(CPU int8)、SenseVoice、FunASR Nano、Anime-Whisper(日语动画/galgame)、GigaAM-v3(俄语);torch 系引擎自动用 **MPS**
- **本地翻译模型**:Apple Silicon 上可一键准备 **HY-MT1.5-7B(MLX 4-bit)**,由应用托管独立 MLX 服务,完全离线翻译
- **任意 OpenAI 兼容 API**:DeepSeek、Grok、Qwen、GPT、Ollama、vLLM……流式输出、JSON 结构化输出、上下文历史、按模型关闭 thinking,全部可配
- **远程 ASR**:把识别负载放到有 GPU 的机器上,见 [REMOTE_ASR.md](REMOTE_ASR.md)
- **会议记录**:按会话保存原文/译文/全文 + Markdown 记录 + JSON 元数据,设置面板内直接回看
- **透明悬浮窗**:置顶、点击穿透、拖拽、14 套配色主题;另有供 OBS 采集的独立字幕窗
- **内置基准测试**:对比各翻译模型的速度与质量

## 环境要求

| 项目 | 要求 |
|---|---|
| 机型 | Apple Silicon(M1 及以上),**必须是原生 arm64** |
| 系统 | macOS 13+ |
| Python | 3.10–3.12 原生 arm64(Rosetta 下的 x86_64 Python 会被安装器直接拒绝) |
| 网络 | 能访问你配置的翻译 API |
| 磁盘 | 模型缓存默认在项目内 `models/`,不写系统盘 |

无需 CUDA/GPU:torch 系 ASR 走 MPS,faster-whisper/CTranslate2 走 CPU(int8)。

## 快速开始

```bash
git clone https://github.com/wei500L/LiveTranslate-mac.git
cd LiveTranslate-mac
./install.sh    # 创建 .venv 并安装 requirements-mac.txt(含 yasbd-lib)
./start.sh      # 启动;环境不完整时会先自动修复再启动
```

首次启动出现设置向导:选择下载源(ModelScope / HuggingFace)与缓存路径,随后自动下载 Silero VAD + SenseVoice(约 1GB),完成后进入主界面。

也可以用环境变量预置翻译端点,免改 YAML:

```bash
export LIVETRANSLATE_API_BASE=http://127.0.0.1:1234
export LIVETRANSLATE_API_KEY=your-key
export LIVETRANSLATE_MODEL=hunyuan-mt-chimera-7b
# 可选:本地 HY-MT 服务端口(默认 8080)
export LIVETRANSLATE_MLX_PORT=8080
./start.sh
```

更新:`./update.sh`(拉取代码并同步依赖)。

## 打包成 Mac 应用

不想从终端启动的话:

```bash
./build_mac_app.sh             # 构建 dist/LiveTranslate.app
./build_mac_app.sh --install   # 构建并安装到 /Applications
```

之后从启动台或 Spotlight(`open -a LiveTranslate`)启动。注意:

- 需要Xcode Command Line Tools(脚本用到 `iconutil` 和 `cc`);
- bundle 未签名,本机使用不受影响(Gatekeeper 只拦有隔离标记的下载文件);
- **启动器内固化了当前项目路径**——移动/重命名项目文件夹后需重新运行脚本;
- 首次以 .app 启动会**重新**请求麦克风/屏幕录制权限:macOS 把 TCC 权限记在 .app 头上,而不是你之前用的终端;
- 应用日志在 `logs/app_bundle.log`。

## macOS 权限(重要)

| 权限 | 何时需要 | 说明 |
|---|---|---|
| **屏幕录制(Screen Recording)** | 采集系统音频 | ScreenCaptureKit 的音频随屏幕捕获通道下发,不开就是永久静音 |
| **麦克风(Microphone)** | 麦克风混入 | 不开则只能采集系统音频 |

- 权限在 系统设置 → 隐私与安全性 中授予;**更改权限后通常需要重启应用**才生效。
- 音频模式存储在 `user_settings.json` 的 `audio_device`:`"__disabled__"` = 仅麦克风,`null` = ScreenCaptureKit 系统音频;在设置面板里切换时后端会热重建,新后端起不来会自动回退到旧的。
- SCK 捕获的是**当前主显示器**的系统声音,不会像 Windows WASAPI loopback 那样枚举出单个输出设备名。

## 本地翻译模型:HY-MT1.5-7B(MLX 4-bit)

Apple Silicon 专属。`./start.sh` 启动后,翻译设置里会出现 **HY-MT1.5-7B (MLX 4-bit)**:

1. 点 **准备本地模型(Prepare Local Model)**:安装隔离的 MLX 运行时(`.mlx-venv`,与主环境 PyTorch 栈互不干扰),将 ModelScope 官方权重转为 4-bit,临时的 BF16 源文件用完自动删除;
2. 用 **启动/停止本地服务** 控制服务;
3. 选中该模型即用本地翻译,选中本身**不会**悄悄起服务,也不会静默回退到别的模型;应用退出时停掉自己拉起的 MLX 服务。

## 翻译 API 配置

设置 → 翻译 页(自动保存,无需点保存按钮):

| 参数 | 示例 |
|---|---|
| API Base | `https://api.deepseek.com/v1` |
| API Key | 你的密钥 |
| 模型 | `deepseek-chat` |
| 代理 | `none`(绕过系统代理)/ `system` / 自定义 URL |

## 架构(macOS 视角)

```
系统音频 (ScreenCaptureKit) ─┐
                             ├→ 16kHz/mono → VAD (Silero, 32ms) → ASR 工作子进程 → LLM 翻译 → 悬浮字幕
麦克风 (PyAudio, 可选混入) ──┘
```

```
main.py                    入口 + 流水线编排
├── audio_capture_sck.py     macOS ScreenCaptureKit 系统音频(MacAudioCapture 门面)
├── audio_capture_pyaudio.py macOS 纯麦克风回退模式
├── vad_processor.py         Silero VAD,渐进静音 + 回溯切分
├── asr_worker.py            ASR 子进程入口(每个后端模型独占一个进程)
├── asr_engine.py            faster-whisper 后端(CPU int8)
├── asr_sensevoice.py 等     FunASR 系 / Anime-Whisper / GigaAM-v3 后端(MPS,失败回退 CPU)
├── asr_remote.py            远程 ASR 客户端(→ asr_server.py)
├── translator.py            OpenAI 兼容客户端(流式/JSON schema/上下文)
├── mlx_service.py           HY-MT MLX 服务的启停与探活
├── transcript_writer.py     按会话的会议记录
├── subtitle_overlay.py      悬浮字幕窗
├── control_panel.py         设置面板(8 页)
└── debug_pipeline.py        诊断:用真实流水线回放音频文件
```

线程/进程模型:主线程 Qt 事件循环;采集线程跑 VAD;ASR 队列线程把分段发给 ASR 子进程(Pipe IPC);跨线程 UI 更新一律走 Qt 信号。

## 排障

**字幕/翻译突然不出字了** —— 别猜,直接回放诊断:

```bash
.venv/bin/python debug_pipeline.py --audio sample.mp3                # 全链路(真实设置+真实 API)
.venv/bin/python debug_pipeline.py --audio sample.mp3 --no-translate  # 仅识别,不联网
```

它走的是真实 VAD、真实 ASR 子进程、真实翻译器,逐阶段打印产出;被要求干活却没产出会直接判失败。完整 DEBUG 日志在 `logs/diagnostic_*.log`,诊断转录单独放在 `transcripts/diagnostic/`,不污染正式会议记录。

**常见 macOS 问题**:

- **系统音频永久静音** → 检查屏幕录制权限是否授予、授权后是否重启了应用;
- **采集崩溃/切换音频模式失败** → 确认没有在 Rosetta 下运行;切换系统音频↔麦克风模式时应用会自动重建后端,失败会回退;
- **安装器报 "native arm64 Python" 错误** → 用了 x86_64 Python,装一个原生 arm64 的 3.10–3.12,或用 `PYTHON_BIN=/path/to/python3 ./install.sh` 指过去;
- **.app 双击没反应** → 看看是不是移动过项目文件夹(启动器路径已固化),重跑 `build_mac_app.sh --install`;日志在 `logs/app_bundle.log`。

**测试与检查**:

```bash
.venv/bin/python -m pytest -q    # 82 个离线测试,无网络、不下载模型
```

## 更新日志

见 [中文更新日志](i18n/CHANGELOG_zh.md) | [English Changelog](i18n/CHANGELOG_en.md)

## 致谢

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — CTranslate2 的 Whisper 推理
- [FunASR](https://github.com/modelscope/FunASR) — SenseVoice / Fun-ASR-Nano
- [Anime-Whisper](https://huggingface.co/litagin/anime-whisper) — 日语动画/ galgame ASR
- [GigaAM](https://github.com/salute-developers/GigaAM) — 俄语 ASR
- [Silero VAD](https://github.com/snakers4/silero-vad) — 语音活动检测

## License

[MIT License](LICENSE)
