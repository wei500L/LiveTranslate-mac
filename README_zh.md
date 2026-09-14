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
- **Soniox 云端实时（俄语→中文）**：连续音频直传云端、不经本地 VAD，安静远距离课堂语音也能识别，译文同流返回 —— 见下文专节
- **兼容任意 OpenAI 格式 API**：DeepSeek、Grok、Qwen、GPT、Ollama、vLLM 等
- **会议记录中心**：每场会议自动存档（原文/译文/Markdown/元数据），可搜索、筛选、重命名，正在记录/已暂停/正在保存/异常中断各有明确标注。一场记录可以随时「结束本次记录」收尾——后台等待最后的识别与翻译完成（最多 30 秒，没等到的按原文保存），不退出应用、不卸载模型；结束后应用停止监听（模型驻留内存），记录立即选中且不再被任何后台结果改动，可生成 AI 纪要或导出 PDF；「开始新记录」先建新会话再恢复监听，同秒连开也不会写串文件
- **AI 会议纪要**：从你已配置的模型中独立选择供应商，把整场记录生成为结构化 Markdown 纪要（会议/课堂双模板），可编辑、可标记过期后重新生成；长会议自动分块提取再逐级汇总。⚠️ 云端供应商会把完整记录发送给第三方，介意隐私请选本地模型（LM Studio、Ollama 等）
- **PDF 会议纪要导出**：A4 排版、含标题/元数据/页码的「仅纪要」或「纪要+完整双语记录」两种 PDF，文字可搜索复制
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

### Soniox 云端实时（俄语/英语 → 中文）

面向大学课堂、会议和远距离教师讲话的云端模式：音频**连续**上传 Soniox（`stt-rt-v5`，俄语），中文译文从同一条 WebSocket 流式返回，不再调用本地/远程翻译器。

**与本地模式的区别**：本地链路依赖本地 VAD 判断哪些音频是语音——老师声音小、离麦克风远时整句会被漏掉。Soniox 模式把连续音频（含安静片段）直接上传，由云端识别；VAD 阈值、最短/最长语音等设置对该模式不生效（设置页中会隐藏）。

**使用步骤**：

1. 在 [Soniox Console](https://console.soniox.com/) 注册并创建 API Key。
2. 配置 Key（二选一，环境变量优先）：`export SONIOX_API_KEY=你的Key`，或在 设置 → VAD/ASR 选择该引擎后在 API Key 输入框填写。Key 以**明文**保存在本机 `user_settings.json`（与翻译模型 Key 相同的保存方式），不会出现在日志中。
3. 说话即可。悬浮窗底部一条实时卡片：第一行俄语原文（较小）、第二行中文译文（较大），未定稿内容颜色较淡；到语义分段点自动固化为历史消息。
4. 可选：填写「课程主题 / 专业术语」——每行一个主题/人名/术语；写成 `俄语词 => 中文译名` 的行会强制按该译名翻译（如 `предел => 极限`）。

Soniox 使用同一条 WebSocket 流式识别和翻译，已结束的字幕没有可单独重新提交的请求，因此不提供“点击字幕重试”；网络层仍会自动重连。字幕词典只在本地确定性高亮原文和译文中的术语。

**分段模式**：准确率优先（默认）/ 平衡 / 低延迟。
**语言提示**：Soniox 设置中可分别开启「识别俄语」和「识别英语」。纯俄语课堂只开启俄语；教师夹带英语术语或英文发言时同时开启两项，英文术语会按原文保留，中文译文同样保留英文术语。至少需要开启一项。

**网络中断**：自动指数退避重连（悬浮窗显示「正在重连」）；连续失败达到上限后停止重试并提示，重新选择引擎即可恢复。 挂系统代理（HTTP/SOCKS）的机器可直接使用——依赖已包含代理支持；若代理软件未运行导致连不上，会显示「正在重连 / 连接失败」。

**费用提醒**：云端按音频流时长计费，选择该引擎即开始产生费用；不用时请切回本地引擎。

**可选的真实连通测试**（默认测试不访问网络）：

```bash
SONIOX_API_KEY=你的Key RUN_SONIOX_LIVE_TEST=1 python -m pytest tests/test_soniox_live.py -q
```

### 课堂术语高亮与录音

在 Soniox 设置中可维护本地字幕词典，每行一个 `俄语 => 中文` 词条，例如
`последовательность => 数列`。词典只在本地做确定性高亮，不会额外调用小模型；重音符号和大小写会自动归一化。

在“缓存/记录”设置中开启“随录音会话保存音频”后，点击“开始新记录”会持续录制当前配置的系统音频/麦克风混音；暂停字幕不会停止录音。
会话结束时先保留 WAV，再后台生成 MP3。若 MP3 编码失败，WAV 仍可在记录详情中打开。发布包可在 `ffmpeg/` 目录携带 FFmpeg，开发环境也会使用系统 PATH 中的 `ffmpeg`。

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
├── meeting_records.py  会议记录数据层：列表/解析/标题/纪要存取
├── meeting_records_page.py 会议记录中心页面（主从布局、搜索筛选）
├── meeting_records_widgets.py 记录中心的列表行/记录渲染/纪要 HTML 组件
├── ai_summary_service.py AI 会议纪要：模板、分块汇总、后台线程
├── pdf_exporter.py     PDF 纪要导出（QTextDocument + QPdfWriter）
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
