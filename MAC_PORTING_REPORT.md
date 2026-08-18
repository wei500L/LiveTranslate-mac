# LiveTranslate 项目全面解析 与 macOS 移植完整报告

> 本报告基于对仓库全部源码的逐文件深入阅读（main.py 2344 行、audio_capture.py、vad_processor.py、全部 asr_*.py、translator.py、model_manager.py、subtitle_overlay.py、subtitle_window.py、control_panel.py、dialogs.py、subtitle_settings.py、benchmark.py、log_window.py、i18n.py、transcript_writer.py、tests/、安装与发布脚本），目标是：**完整理解项目 + 在 macOS（Apple Silicon / M 系列芯片）上完整实现移植方案**。
>
> **既定技术选型**：目标机器为 **M 系列芯片（Apple Silicon，ARM64）**，系统音频捕获采用 **ScreenCaptureKit（SCK）系统音频捕获** 作为唯一主路径（macOS 13.0+ 全面可用），推理后端采用 **MPS + Apple Silicon CPU int8**。

---

## 目录

1. [项目概览](#1-项目概览)
2. [系统架构总览](#2-系统架构总览)
3. [线程 / 进程模型（精确描述）](#3-线程--进程模型精确描述)
4. [模块逐个详解](#4-模块逐个详解)
5. [核心数据流：一段语音的完整生命周期](#5-核心数据流一段语音的完整生命周期)
6. [增量 ASR 机制详解](#6-增量-asr-机制详解)
7. [配置与持久化体系](#7-配置与持久化体系)
8. [Windows 专属依赖完整清单](#8-windows-专属依赖完整清单)
9. [macOS 移植方案（逐项对照）](#9-macos-移植方案逐项对照)
10. [移植路线图（分阶段）](#10-移植路线图分阶段)
11. [风险与注意事项清单](#11-风险与注意事项清单)
12. [附录：术语表与文件索引](#12-附录术语表与文件索引)

---

## 1. 项目概览

**LiveTranslate** 是一个 Windows 平台的实时音频翻译系统（Phase 0 Python 原型）：

- 通过 **WASAPI loopback** 捕获系统音频（32ms 块），可选混入麦克风输入
- **Silero VAD** 检测语音并智能切分（渐进静默、自适应静默、回溯切分）
- 多引擎 **ASR**（faster-whisper / SenseVoice / FunASR Nano / Anime-Whisper / 远程 Whisper，新增 **GigaAM v2/v3 俄语专用**），ASR 运行在**独立子进程**中（spawn + Pipe IPC），崩溃自动重启、内存膨胀自动回收
- **LLM 翻译**（任意 OpenAI 兼容 API），支持流式输出、JSON 结构化输出、多思考模式禁用、上下文历史、逐模型参数覆盖
- **PyQt6 透明悬浮窗**（聊天式双语显示、点击穿透、置顶、14+ 主题）+ **独立字幕窗口**（OBS 采集用，描边文字、动画）
- 系统托盘完整菜单、控制面板（7 标签页、300ms 防抖自动保存）、基准测试、双语 i18n、转录持久化

**一句话架构**：`系统音频(32ms) → VAD → [增量ASR切句] → ASR子进程 → LLM翻译(流式) → 悬浮窗/字幕窗/转录文件`

### 目标平台（macOS 移植）

| 项 | 决策 |
|---|---|
| 芯片 | **M 系列（Apple Silicon，ARM64）**，不针对 Intel Mac 做优化（Intel 亦可用但非目标） |
| 系统音频捕获 | **ScreenCaptureKit 系统音频捕获**（主路径，唯一方案） |
| 麦克风捕获 | 普通 PyAudio（CoreAudio host） |
| 推理后端 | torch → **MPS**；faster-whisper/CTranslate2 → **Apple Silicon CPU int8** |
| 点击穿透 | `NSWindow.ignoresMouseEvents`（PyObjC） |
| 系统要求 | macOS 13.0+（M 系列所有机型均满足；SCK 音频捕获自 13.0 起可用） |
| Python | 3.10–3.12（**arm64 原生 wheel**，勿用 Rosetta x86_64） |

---

## 2. 系统架构总览

```
┌───────────────────────────── 主进程 (GUI) ─────────────────────────────┐
│                                                                        │
│  Qt 主线程: QApplication 事件循环                                      │
│    ├── SubtitleOverlay   悬浮窗 (subtitle_overlay.py)                  │
│    ├── SubtitleWindow    OBS字幕窗 (subtitle_window.py)                │
│    ├── ControlPanel      设置面板 (control_panel.py)                    │
│    ├── LogWindow / 托盘 / 对话框 (dialogs.py)                          │
│                                                                        │
│  LiveTranslateApp (main.py):                                           │
│    捕获线程 _capture_loop:                                             │
│      AudioCapture.audio_queue ─→ VADProcessor.process_chunk            │
│        ├─ 段完成 → _asr_queue ("vad_flush", segment)                   │
│        └─ 说话中且增量开启 → _asr_queue ("interim", None)               │
│    ASR 线程 _asr_loop:                                                 │
│      _asr_queue ─→ _run_asr() ─→ ASRClient.transcribe() ─(Pipe)─┐      │
│      含: 语言/padding 延迟应用、worker 自动重启、RSS 回收           │      │
│    翻译线程池 ThreadPoolExecutor(8+):                              │      │
│      _translate_async ─→ Translator.translate_iter (流式)          │      │
│      _translate_extra_langs (字幕窗多语言并行)                     │      │
│    跨线程 UI 更新一律走 Qt pyqtSignal                               │      │
└──────────────────────────────────────────────────────────────┬──────┘
                                                               │ multiprocessing
                                                    spawn + Pipe (duplex)
┌───────────────────────────── ASR Worker 进程 ────────────────▼──────┐
│  asr_worker.worker_main: 加载一个引擎, recv 循环处理                  │
│    transcribe / set_language / set_input_padding / shutdown          │
│  引擎: ASREngine(faster-whisper) / FunASREngine(→SenseVoice|Nano)    │
│        / AnimeWhisperEngine / GigaAMEngine(俄语)                       │
└───────────────────────────────────────────────────────────────────────┘

特殊: remote-whisper 引擎 = asr_remote.RemoteASREngine，进程内 httpx 客户端
      (无子进程, pid=None), 直接对接 asr_server.py (FastAPI, 可跑在GPU机器)
```

---

## 3. 线程 / 进程模型（精确描述）

| 执行单元 | 位置 | 职责 | 关键细节 |
|---|---|---|---|
| **Qt 主线程** | `main()` | 事件循环、全部 UI、托盘 | `setQuitOnLastWindowClosed(False)`；200ms 空 QTimer 保持 SIGINT 响应；`QTimer.singleShot(100)` 延迟初始化防止启动卡死 |
| **捕获线程** | `_capture_loop` | 读音频队列 → VAD → 入 ASR 队列 | 超时无音频时若正在说话，注入静音块强制推进 VAD 切分 |
| **ASR 队列线程** | `_asr_loop` | 消费 `(seg_type, segment)` | `vad_flush`→`_process_segment`/`_process_interim_final`；`interim`→先 `_drain_interim_duplicates()` 再 `_do_interim_asr()`；队列为空时做 worker RSS 回收检查 |
| **翻译线程池** | `ThreadPoolExecutor(max(8, n_langs+1))` | 异步翻译 + 流式 UI 推送 | 供主目标语言与字幕窗额外语言并行使用 |
| **ASR Worker 进程** | `asr_worker.py` | 拥有唯一 ASR 模型并推理 | spawn 上下文、daemon、Pipe 双向、uuid 请求 id 匹配 |
| **音频读取线程** | `AudioCapture._read_loop` | WASAPI loopback + mic 读取、混音 | 队列满时丢最旧；设备热切换自动重开流 |

**锁策略**（移植时必须保持）：
- `_asr_lock` (RLock)：保护引擎切换/重启状态机；**阻塞的跨进程 transcribe 不持有此锁**（否则慢 worker 会卡死 Qt 线程的设置变更）
- `_vad_lock`：VAD 状态的读/修剪互斥
- `_asr_pending_lock`：Qt 线程写入、ASR 线程在下一次 transcribe 前应用的"延迟设置"（语言/padding），避免 UI 阻塞在忙碌管道上
- `ASRClient._lock` (RLock)：串行化管道访问

**Worker 生命周期状态机**（`ASRClient.status`）：
`created → starting → loading → ready ⇄ busy → stopping → stopped`；异常路径 `failed` / `exited`（进程退出码非空）。

**自动恢复机制**：
- `_recover_asr_worker`：worker 死亡/超时 → 最多自动重启 3 次（`_asr_restart_max`）
- `_asr_generation` 代数守卫：后台慢加载完成后若已有更新的切换，废弃过期 worker
- `_maybe_recycle_asr_worker`：worker RSS 超基线 +2048MB → 空闲时优雅回收（防 FunASR/CTranslate2 原生侧泄漏）
- 引擎切换失败 → 用保存的旧配置恢复上一个 worker

---

## 4. 模块逐个详解

### 4.1 `audio_capture.py`（443 行）— ⚠️ 全文件 Windows 专属

- 依赖 **`pyaudiowpatch`**（PyAudio 的 WASAPI loopback 分支）
- `list_output_devices()` / `list_input_devices()`：枚举 WASAPI 宿主 API 下设备（排除 `isLoopbackDevice`）
- **loopback 查找**：取 WASAPI 默认输出设备名 → 在 loopback 设备中匹配 `target_name in dev["name"]`，找不到则回退任意 loopback
- 流参数：原生采样率/声道（通常 48kHz 立体声），`paFloat32`，native_chunk 按 chunk_duration
- `_resample_to_mono()`：多声道取均值 + 线性插值重采样到 16k 单声道 float32
- **麦克风混入**：独立 mic 流，可变长缓冲 `_mic_buf`，按 loopback 块长对齐后逐样本相加，输出 `(audio, mic_rms)` 元组
- **热切换**：`set_device()`/`set_mic_device()` 置 restart event；每 2 秒（`DEVICE_CHECK_INTERVAL`）新建 PA 实例查询当前默认输出，变化即重启流（仅 `device=None` 跟随系统默认时）
- `"__disabled__"` 哨兵值 = 关闭 loopback（纯麦克风模式，注入静音维持管线节拍）
- 读错误自动 `_restart_stream()`（设备拔出场景）

### 4.2 `vad_processor.py`（387 行）— 平台无关

- Silero VAD v5：优先 `silero-vad` PyPI 包（模型内置 wheel，零网络），否则 torch.hub（固定分支可离线）；`torch.set_num_threads(1)`
- 16kHz 时窗口 512 样本（正好 = 32ms chunk，`chunk_duration=0.032` 的由来）
- 三种模式：`silero` / `energy`（RMS/阈值） / `disabled`（恒 1.0）
- **pre-speech 环形缓冲**：3 块（~96ms）捕获 VAD 触发前的声母； onset 时以阈值置信度注入
- **渐进静默**（`_progressive_tiers`）：缓冲 <3s 用全额静默上限，3–6s 减半，6–10s 取 1/4 → 语速快的人更快出句
- **自适应静默**：记录近 50 次停顿时长，P75×1.2，钳制 0.3–2.0s
- **回溯切分**（`_split_at_best_pause`）：达到 max_speech_duration 时，对置信度历史做 5 块滑动平均平滑，在后 70% 区间找最低谷（要求真凹陷 dip_ratio<0.8 或低于阈值），前段出段、余段保留继续累积
- **语音密度过滤**：出段时若 <25% 块超阈值 → 整段丢弃（噪声）
- **短段不丢弃**：低于 min_speech_duration 只做软复位（`_is_speaking=False`）保留缓冲，与下次语音自然合并
- `_was_trimmed` 标志：增量 ASR 修剪过的缓冲，即使变短也 `force_flush()` 而非丢弃
- 增量接口：`peek_buffer()`（读不清空）、`trim_front(n_samples)`（按样本修剪）、`force_flush()`

### 4.3 `asr_client.py` / `asr_worker.py` — 平台无关

**ASRClient**（主进程代理）：
- spawn 上下文创建 `ASRWorker-<engine>` daemon 进程 + 双向 Pipe
- `wait_ready(180s)`：等待 worker 首条 ready 消息
- `_request()`：uuid id 匹配响应；超时 → terminate + `ASRWorkerTimeout`；管道断 → `ASRWorkerExited`
- `shutdown()`：发送 shutdown → join(5s) → 不退出则 terminate；`transcribe` 请求超时 60s（`request_timeout`，VAD 段只有几秒，60s 已很宽松）

**worker_main**：
- `_load_engine(config)` 按引擎类型构建后端；设备串解析 `"cuda:0 (RTX 4090)"` → `("cuda", 0)`；CPU 上 float16 自动降级 int8
- 命令循环：`transcribe`（inspect 签名决定是否传 word_timestamps）/ `set_language` / `set_input_padding` / `shutdown`
- 错误分类：加载失败 `recoverable=False`；命令失败 `recoverable=True`（主进程计次，≥3 次或不可恢复才废弃 worker）
- 退出清理：`unload()` + gc + `torch.cuda.empty_cache()`

### 4.4 ASR 引擎后端

| 文件 | 引擎 | 关键实现细节 |
|---|---|---|
| `asr_engine.py` | **Whisper**（faster-whisper/CTranslate2） | `device="cuda"`（不能 `cuda:0`，索引用 `device_index` 参数）；beam_size=5、vad_filter=False；**输入填充桶**（pad_seconds，默认 0.5s，量化到样本倍数，减少尾部分桶抖动）；unload 时逐属性删除释放 |
| `asr_funasr.py` | **FunASR 统一分发器** | 按 `model_key` 的 profile 分发到 sensevoice / funasr-nano 家族适配器；profile 声明 `supports_padding`/`supports_language` |
| `asr_sensevoice.py` | **SenseVoice** | FunASR `AutoModel(trust_remote_code=True)`；CUDA 时 fp16（`model.half()` + `torch.autocast`），CPU fp32；输出解析 `<|zh|>` 等语言标签 + 正则剥除 `<|HAPPY|>`/`<|BGM|>` 情绪事件标签；`disable_pbar=True` 必需（tqm 在 GUI 进程会崩） |
| `asr_funasr_nano.py` | **Fun-ASR-Nano / MLT-Nano** | 先把仓库内 `funasr_nano/` 加入 sys.path 并 `import model` 预注册模型类；**`os.chdir(model_dir)`** 后 AutoModel（让 config.yaml 里 `Qwen3-0.6B` 相对路径本地解析，避免 HF 网络请求），完成后 chdir 回去；音频写 16bit WAV 临时文件再 generate；语言未知时按 Unicode 区间猜（假名→ja、谚文→ko、汉字 30%→zh、否则 en） |
| `asr_anime_whisper.py` | **Anime-Whisper**（litagin，日语动漫/galgame特化） | transformers pipeline（非 ctranslate2）；fp16@cuda；generate_kwargs 固定日语、`no_repeat_ngram_size=5`；日语专用，忽略 set_language |
| `asr_gigaam.py`（**新增**） | **GigaAM v2 / v3**（Sber AI，俄语专用） | transformers 加载（`AutoModelForCTC`/`AutoModelForRNNT` 或 pipeline）；**纯俄语**，固定 `language="ru"` 忽略 set_language；16kHz 输入与项目一致；torch 架构 → MPS/CPU 皆可。详见下方接入要点 |
| `asr_remote.py` | **Remote-Whisper** | 进程内 httpx 客户端（`trust_env=False` 防止 localhost 走代理）；**自定义二进制协议**：`[uint32 lang_len][lang utf-8][float32 PCM]` POST → JSON `{text, language, elapsed}`；启动时 `/health` 探活，失败抛 ConnectionError（UI 认定为"预期错误"不弹栈）；实现 ASRClient 同形接口（status/pid=None/shutdown…）直接嵌入 `_switch_asr_engine` |
| `asr_server.py` | 远程服务端 | FastAPI + uvicorn；`_gpu_lock` 串行化 GPU；请求体严格校验（防恶意 body）；可部署在任意 GPU 机器，**本身跨平台** |

**GigaAM 接入要点（俄语专用，新增引擎）**

GigaAM 是 Sber AI 开源的俄语 ASR，与 Anime-Whisper 之于日语完全对称——单语言专用、torch 架构、可 MPS 加速，正好补齐"俄语 + M 芯片 GPU"的缺口：

| 模型 | HF 仓库 ID | 架构 | 说明 |
|---|---|---|---|
| GigaAM v2 | `salute-ai/GigaAM-v2-CTC`、`salute-ai/GigaAM-v2-RNNT` | Conformer | 较旧，参数小 |
| GigaAM v3 | `salute-ai/GigaAM-v3-CTC`、`salute-ai/GigaAM-v3-RNNT` | RNN-T | **推荐**，比 v2 更快更准（约 300M） |

接入实现（仿 `asr_anime_whisper.py`）：

```python
# asr_gigaam.py — 骨架
class GigaAMEngine:
    def __init__(self, device="mps", hub="hf"):
        import torch
        from transformers import pipeline
        if device == "mps" and not torch.backends.mps.is_available():
            device = "cpu"
        self.language = "ru"  # 俄语专用，忽略 set_language
        self._pipe = pipeline(
            "automatic-speech-recognition",
            model="salute-ai/GigaAM-v3-RNNT",   # v2/v3、CTC/RNNT 可配置
            device=device,
            torch_dtype=torch.float32,          # MPS 保守 fp32
        )
    def transcribe(self, audio):                # audio: float32 16kHz mono（已匹配）
        with torch.inference_mode():
            result = self._pipe(audio)
        text = (result or {}).get("text", "").strip()
        return {"text": text, "language": "ru", "language_name": "ru"} if text else None
```

配套改动（参照 Anime-Whisper 的接入方式）：
1. `model_manager.py`：`ASR_MODEL_IDS["gigaam"] = "salute-ai/GigaAM-v3-RNNT"`；`ASR_DISPLAY_NAMES` 加 `"gigaam": "GigaAM"`；`is_asr_cached`/`get_missing_models`/`download_asr` 加 gigaam 分支（HF-only）
2. `asr_worker.py` 的 `_load_engine` 加 `elif engine_type == "gigaam":` 分支
3. `control_panel.py` 的 ASR 引擎下拉框加 "GigaAM (ru)" 选项
4. 语言行为：固定 `ru`，`set_language` 忽略（与 Anime-Whisper 同款处理），因此 UI 的 `asr_language` 过滤对俄语不产生误判（对比 SenseVoice 的 `LANG_MAP` 缺 `ru` 问题）

**俄语选型结论**：俄语 ASR 优先级 = **GigaAM v3（MPS GPU，最快且准）** > faster-whisper（CPU int8，准但慢）> SenseVoice（俄语属长尾，准确率打折 + 需补 LANG_MAP）。

### 4.5 `translator.py`（501 行）— 平台无关

- `make_openai_client()`：**唯一**的代理感知 OpenAI 客户端工厂（translator + benchmark 共用）；`proxy="none"` → `httpx.Client(trust_env=False)` 绕过系统代理；custom URL → `httpx.Client(proxy=...)`；超时 `httpx.Timeout(t, connect=5.0)`
- **thinking_style 体系**（issue #38：思考模型把 max_tokens 全烧在推理上导致空输出）：
  - 6 种：`auto` / `deepseek`(`{"thinking":{"type":"disabled"}}`) / `qwen`(`{"enable_thinking":false}`) / `vllm`(`chat_template_kwargs`) / `openai`(`reasoning_effort:"none"`) / `off`(不发)
  - `auto` 启发式：模型名含 deepseek/glm 或端点含 deepseek/volces/api.z.ai/bigmodel → deepseek 式；官方 api.openai.com/api.x.ai/api.anthropic.com → off（拒绝未知参数）；否则 qwen 式
  - 空补全 + completion_tokens>0 → 记录 "thinking burned the budget" 警告
- `_build_request_kwargs()`：**唯一组装点**——注入 `overrides`（temperature/top_p/max_tokens/frequency_penalty/presence_penalty/seed，仅存在的键）、合并 `extra_body`（显式值覆盖自动 thinking 参数）、`json_response` 时盖 `response_format={"type":"json_schema", schema:{"t":"string"}}`
- 流式：`translate_iter()` 生成器逐块累积产出；总截止时间 deadline（防挂死）；`stream_options.include_usage` 失败自动降级重发；`translate()` 是阻塞等价物
- `RepetitionError`：≥40 字符且存在 8+ 长度前缀模式循环 → 上层显示用户可读警告
- 上下文：`context_turns` 内 (source, translation) 对，`{context}` 占位符或 multi-turn messages 两种注入方式；`no_system_role` 时合并进 user 消息（Qwen-MT 等拒收 system）
- `PROMPT_PRESETS`：daily / esports / anime / webid（视频 KYC 场景）+ DEFAULT_PROMPT
- `with_target_language()`：零拷贝式克隆共享 httpx client，供字幕窗多语言并行翻译

### 4.6 `model_manager.py`（749 行）— 平台无关

- 缓存根：`./models/`，`apply_cache_env()` 设 `MODELSCOPE_CACHE` / `HF_HOME` / `TORCH_HOME`（main.py 在 `import torch` **前**调用）
- 模型目录：SenseVoice（MS `iic/SenseVoiceSmall` ↔ HF `FunAudioLLM/SenseVoiceSmall` 命名空间不同）、Fun-ASR-Nano/MLT-Nano（内嵌 Qwen3-0.6B 权重需单独 `ensure_qwen_weights()` 拉取）、anime-whisper（仅 HF）、Whisper（`Systran/faster-whisper-*`）
- 缓存检测兼容 ModelScope 多代 SDK 目录布局（`{org}/{name}`、`models/{org}/{name}`、点号变 `___`、`{org}--{name}/snapshots/`）；HF 检测要求快照**完整**（无断链、总字节达标）——防止中断下载被误判为已缓存导致加载挂死
- `neutralize_funasr_requirements()`：重命名模型目录内 requirements.txt，跳过 FunASR trust_remote_code 触发的静默 `pip install -r`（慢网/被代理挡住会无限挂死）
- 代理上下文 `_proxy_env()`：覆盖 torch.hub(urllib)/HF/modelscope(requests) 的 *_PROXY 环境变量 + 显式 opener
- Silero 下载含 Python 3.13 `VERIFY_X509_STRICT` SSL 严格校验的降级重试路径
- 自定义 Whisper 模型：`model.bin + config.json` 目录判定为 CTranslate2 模型，`./models/` 递归扫描入选择器

### 4.7 `subtitle_overlay.py`（1286 行）

- **SubtitleOverlay**：`FramelessWindowHint | WindowStaysOnTopHint | Tool` + `WA_TranslucentBackground` + `WA_ShowWithoutActivating`；右下角默认位置；消息上限 50 条；位置/尺寸持久化（moveEvent/resizeEvent 500ms 防抖 → `overlay_x/y/w/h`）
- **DragHandle**（两行头）：
  - 行1：拖拽区标题 + Hide / Subtitle / Paused-Running / Clear / Full-Compact / Settings / Quit 按钮
  - 行2a：Click-through / Top-most / Auto-scroll / Taskbar 复选框
  - 行2b：模型选择、源语言、目标语言三个下拉框
- **MonitorBar**：MIC/RMS/VAD 三条进度条 + `CPU% RAM GPU | ASR数 TL数 Tok(↑↓) ¥/$成本` 富文本状态行；psutil 1s 轮询；成本符号随 UI 语言切换
- **ChatMessage**：头部（时间戳[语言]原文 ASR耗时）+ 翻译行（`> 翻译 TL耗时`）；流式更新 50ms QTimer 节流；右键菜单（复制/导出/清空）
- **点击穿透（Windows 核心）**：`WS_EX_TRANSPARENT`(0x20) 经 `ctypes.windll.user32.SetWindowLongW` 设置在**滚动区**；50ms 定时器 `_check_click_through` 按光标是否在头部区域动态开/关 → 头部永远可交互、正文永远穿透
- **Top-most**：切换 `WindowStaysOnTopHint`（需 `setWindowFlags + show()` 生效）
- **Taskbar**：`Qt.Tool` 标志切换（隐藏任务栏图标）
- **Compact 模式**：折叠 row2/monitor，200ms 尺寸动画；用 `frameGeometry()` 取实际窗口尺寸规避 Windows MINMAXINFO 不一致；高度差 <10px 跳过动画
- 样式系统：`DEFAULT_STYLE` + 14 个预设（Dracula/Nord/Monokai/Solarized/Gruvbox/Tokyo Night/Catppuccin/One Dark/Everforest/Kanagawa…）；原文/译文独立字体字段（`original_font_family` / `translation_font_family`），旧 `font_family` 自动迁移；`apply_style()` 重建全部消息 HTML

### 4.8 `subtitle_window.py`（934 行）

- OBS 采集专用独立透明窗：无边框、置顶、`WA_ShowWithoutActivating`、中键拖拽、固定宽（默认 1000）高度自适应
- `_SubtitleTextWidget`：
  - **QPainterPath 描边文字**：路径 stroke（描边色、2 倍宽、圆角连接）+ fill（文字色）
  - 文本渲染进缓存 QPixmap（含 devicePixelRatio），动画只 blit 缓存图
  - 自动换行：`QFontMetrics` 宽度测量 + 标点/空格断行偏好；`desired_height()` 永不为 0
  - 进出场动画：fade / slide_left/right/up/down（QPropertyAnimation 自定义 `content_opacity`/`slide_offset_*` 属性）
- 窗口高度动画 150ms OutCubic，**保持垂直中心不动**（y 随高度差移动一半）
- 自动隐藏超时 + 反向恢复动画；最小显示时间 1500ms（快速更新排队而非立即替换）
- 多行配置：每行可为 `original` 或 `translation`（指定语言）；`get_target_languages()` 返回启用翻译行的语言集合 → 驱动多语言并行翻译
- **点击穿透**：同 overlay 的 WS_EX_TRANSPARENT，500ms 定时器 + showEvent 重断言（Qt 在 show/flag 变化后会重置扩展样式）
- 线程安全：`update_text()` 信号序列化（dict→json 字符串传参）

### 4.9 `control_panel.py`（~1800 行）

- 7 个标签页：**VAD/ASR**（引擎、设备、模型、VAD 全参数、增量 ASR、音频/麦克风设备）· **Translation**（模型列表 CRUD + 活动模型、语言、超时、prompt 预设、上下文轮数、转录开关）· **Style** · **Subtitle**（字幕窗设置内嵌）· **Benchmark** · **Cache**（缓存扫描/清理）· **Changelog**（i18n/CHANGELOG_*.md → HTML）
- **自动保存**：全部控件 300ms 防抖 `_auto_save()`；滑块实时更新标签但 `sliderReleased` 才保存（键盘输入 `isSliderDown()` 立即）；prompt TextEdit 600ms 防抖
- 信号：`settings_changed` / `model_changed` / `models_list_changed` / `subtitle_settings_changed` / `reset_positions`
- 设置读取过滤 `models` 与 `system_prompt` 防止 API key 泄漏进日志
- 保存时全部 float `round(x, 2)` 防浮点漂移；**原子写**（`.tmp` + `os.replace`）

### 4.10 `dialogs.py` / `subtitle_settings.py` / 其他

- **dialogs.py**：`SetupWizardDialog`（首启：hub + 缓存路径 + 下载 Silero/SenseVoice）、`ModelDownloadDialog`（缺模型自动下载，含代理）、`ModelEditDialog`（API 配置 + 高级参数组：`[Override] 复选框 + 值控件` 模式，extra_body 多行 JSON 校验）、`_ModelLoadDialog`、changelog 渲染；**高 DPI 适配（issue #39）**：`available_screen_height()` + `make_scroll_area()`（绕过 QScrollArea 36x24 字高 sizeHint 上限），窗口高钳制到屏幕可用区域
- **subtitle_settings.py**：字幕窗设置网格布局；行列表（双击 `LineEditDialog` 编辑：类型/语言/字体/颜色/透明度/描边/对齐/背景图/动画）；增删移行
- **benchmark.py**：`BENCH_SENTENCES`（ja/en/zh/ko/fr/de 各 5 句）；`run_benchmark()` 后台线程逐模型测 TTFT/总耗时/输出，`make_openai_client` 共用
- **log_window.py**：启动即建隐藏，托盘 "Show Log" 呼出；`QLogHandler` 追加富文本
- **i18n.py**：zh/en YAML，`t(key)`；系统 locale 探测；`LANGUAGES` 30 项 + `COMMON_LANG_CODES` 托盘常用集
- **transcript_writer.py**：按会话写 3 个文件（original/translation/all），逐条即时落盘（行缓冲支持 tail -f），线程安全，msg_id 配对原文与译文
- **tests/**：分割逻辑、translator thinking 样式、requirements 一致性、启动脚本完整性（**校验 .ps1/.bat 内容，Windows 专属**）
- **安装/发布**：`install.ps1`（uv 管理便携 Python、winget 自动装 Python/Git、注册表桥接系统代理、全部缓存限项目盘）、`install.bat`/`update.bat`/`start.bat`、`build_release.ps1`（生成 bootstrap.ps1 便携包）、`.github/workflows/release.yml`（**windows-latest** 构建）

---

## 5. 核心数据流：一段语音的完整生命周期

```
1. AudioCapture._read_loop 读取 48kHz/2ch loopback 块 (+mic 混音)
   → _resample_to_mono → 16k mono float32 → audio_queue
2. _capture_loop: (chunk, mic_rms) → overlay.update_monitor(RMS/VAD/MIC)
   → VADProcessor.process_chunk(chunk)
   ├─ None（累积中）: 若增量开启且 buf≥interval 且 cooldown≥1s → 入队 ("interim", None)
   └─ segment（切分完成）→ ASR ready? → 入队 ("vad_flush", segment)
3. _asr_loop 取出:
   "vad_flush":
     非 interim_active → _process_segment:
       _run_asr → [应用延迟设置] → client.transcribe(audio)
       过滤: 空文本/纯标点 → 丢弃; ≥2s 且 ≤3 字符 → 噪声丢弃; 语言不匹配 → 丢弃
       overlay.add_message(msg_id, ts, 原文, 语言, 耗时)  (信号→Qt)
       transcript.write_original
       源==目标 → 免翻译直接显示; 否则线程池提交 _translate_async
     interim_active → _process_interim_final（余段识别 + 去回声 + 短句合并）
   "interim": → _do_interim_asr（见 §6）
4. _translate_async: Translator.translate_iter 流式
   每个 partial → overlay.update_streaming(节流 50ms)
   完成 → transcript.write_translation / overlay.update_translation + 统计/成本
   字幕窗可见 → 组装 {目标语言: 译文, 额外语言...} → subwin.update_text
   RepetitionError → "[检测到重复循环]" 用户提示
```

**ASR 错误路径**：`ASRWorkerExited/Timeout` → `_recover_asr_worker`（自动重启≤3次）；`ASRWorkerError` → 计次，≥3 或不可恢复 → `_mark_asr_unavailable`；每次 ASR 后记录 `MEM[asr#N]` 内存快照，组合 RSS 超 4096MB → 托盘告警一次。

---

## 6. 增量 ASR 机制详解（`incremental_asr` 设置）

目的：连续长语音不等说完，边说边出已完成的句子，降低延迟。

1. **触发**：说话中、缓冲 ≥ `interim_interval`（默认 2s）、距上次增量 ≥interval 且冷却 ≥1s
2. **识别**：`peek_buffer()` 全量识别（不 word_timestamps——重复代价太高）
3. **去回声**：`_strip_committed_overlap()`——上次已提交文本的尾部（末 50 字符）与本次识别前缀做后缀匹配，剥掉重叠部分
4. **切句**：`yasbd` 库 `pysbd_adapter.Segmenter`（规则式，40 语言含 ko，~0.2ms/调用）；不可切的长句回退逗号切分：CJK `、` 25 字符阈值 / 西文 `,，;；` 60 字符阈值；要求前段 >15 字符、后段 >3 字符防碎片
5. **提交**：除最后一句（还在说）外全部输出；短碎句（≤8 字母数字）进 `_interim_pending` 缓冲拼接到下一句
6. **音频修剪**：按提交文本字符占比比例修剪 + 0.3s 安全余量（减少回声），保底留 ≥0.5s；置 `_was_trimmed`
7. **收尾**：VAD 最终切分时走 `_process_interim_final`，`force_flush` 余段再次识别，同样去回声 + 短句合并后输出

---

## 7. 配置与持久化体系

| 文件 | 角色 | 要点 |
|---|---|---|
| `config.yaml` | 基础默认 | 音频（16k/32ms）、ASR（engine/model/device/compute/VAD/填充秒）、翻译（API/流式/prompt 模板）、字幕样式默认 |
| `user_settings.json` | 运行时持久化（**优先级更高**） | 全部 UI 设置、模型列表（含每模型 flags）、overlay/字幕窗位置、`cache_path`、`ui_lang`、`subtitle_mode` |
| `models/` | 模型缓存根 | `modelscope/`、`huggingface/hub/`、`torch/hub/` 子目录 |
| `transcripts/` | 转录输出 | 每会话 3 文件 |
| `logs/` | 滚动日志 | `livetrans_时间戳.log`（DEBUG 级） |

**每模型配置字段**：`name, api_base, api_key, model, proxy(none/system/URL), streaming, json_response, no_system_role, thinking_style, context_turns, input_price, output_price, overrides{temperature,top_p,max_tokens,frequency_penalty,presence_penalty,seed}, extra_body{...}`

---

## 8. Windows 专属依赖完整清单

| # | 位置 | 机制 | 性质 |
|---|---|---|---|
| W1 | `audio_capture.py` 全文件 | `pyaudiowpatch` WASAPI loopback（`isLoopbackDevice`、WASAPI host API 枚举、默认输出跟随） | **硬依赖，全平台无此 API** |
| W2 | `subtitle_overlay.py:26-27,1064-1094` | `ctypes.windll.user32.Get/SetWindowLongW` + `WS_EX_TRANSPARENT`(0x20) 点击穿透；50ms 光标轮询保头部可交互 | 硬依赖 |
| W3 | `subtitle_window.py:33-37,768-782` | 同上（500ms 定时器 + showEvent 重断言） | 硬依赖 |
| W4 | `main.py:40-41` | `import torch` 必须在 PyQt6 前（PyTorch 2.9 Windows DLL 冲突 pytorch#166628） | Windows 特有 bug 规避 |
| W5 | `main.py:1777-1779` | 钉 "Segoe UI" 字体（防 DirectWrite 解析 MS Sans Serif 位图字体失败） | Windows 特有 |
| W6 | `main.py:2341-2343` | `multiprocessing.freeze_support()`（Windows 打包 spawn 需要） | 兼容保留无害 |
| W7 | `config.yaml` / `DEFAULT_STYLE` / `DEFAULT_SUBTITLE_WIN_SETTINGS` / `subtitle_settings` | 默认字体 "Microsoft YaHei"、"Consolas"（UI/监控条） | 需按平台换字体栈 |
| W8 | `requirements.txt:12` | `PyAudioWPatch>=0.2.12`（仅 Windows wheel） | 替换 |
| W9 | CUDA 路径：`config.yaml device:cuda`、`asr_worker._parse_device`、SenseVoice fp16 逻辑、MonitorBar GPU 读取 `torch.cuda`、`_parse_device` 设备组合框（"cuda:0 (名称)"） | NVIDIA 专属 | macOS → MPS/CPU |
| W10 | `install.ps1` / `install.bat` / `update.bat` / `start.bat` / `build_release.ps1` + `tests/test_startup_environment.py`、`test_requirements.py`（断言 .ps1/.bat 内容） | winget、注册表代理、uv 便携 Python | 全部重写为 shell |
| W11 | `.github/workflows/release.yml` | `runs-on: windows-latest` | 增加 mac job |
| W12 | 各处日志/注释中的 WASAPI 措辞、README 平台徽章 | 文档 | 更新 |

其余部分——VAD、ASR worker 架构、全部 ASR 后端（faster-whisper CPU / FunASR / transformers / remote）、translator、PyQt6 UI 主体、i18n、transcript——**均为纯跨平台 Python**。

---

## 9. macOS 移植方案（逐项对照）

### 9.1 W1 系统音频捕获 —— ScreenCaptureKit（M 系列，既定方案）

macOS 没有进程级 loopback API；M 系列机器统一采用 **ScreenCaptureKit（SCK）系统音频捕获** 作为唯一主路径，BlackHole 仅作为极端降级回退（见末段）。

#### SCK 系统音频捕获原理

- `SCStreamConfiguration.capturesAudio = True` + `exclusivelyCapturesSystemAudio = True`（只捕系统混音输出，不含麦克风；与独立麦克风流分离，混音逻辑复用现有 `_resample_to_mono`）
- 用 `SCContentSharingPicker`（或直接 `SCShareableContent`）选一个**显示器**作为采集目标（SCK 的音频必须依附在某个 content 上，选主显示器即可，无画面需求可不渲染视频帧）
- 采集会话 `SCStream` 通过 delegate 持续投递 `CMSampleBuffer`；从 `sBufTypeID == 音频` 的 buffer 提取 `CMBlockBuffer` → `AudioBufferList` → float32 PCM

#### 关键绑定：PyObjC

```bash
pip install pyobjc-framework-ScreenCaptureKit \
            pyobjc-framework-CoreMedia \
            pyobjc-framework-AVFoundation \
            pyobjc-framework-AppKit
```

```python
# audio_capture_sck.py — 骨架（实际代码需完整 delegate 生命周期管理）
import objc
import ctypes
from Foundation import NSRunLoop, NSDate
import ScreenCaptureKit

class AudioDelegate(ScreenCaptureKit.NSObject):
    def init(self):
        self = objc.super(AudioDelegate, self).init()
        return self
    # stream:didOutputSampleBuffer:ofType:
    def stream_didOutputSampleBuffer_ofType_(self, stream, sampleBuffer, type_):
        if type_ != ScreenCaptureKit.SCStreamOutputTypeAudio:
            return
        # CMSampleBuffer → CMBlockBuffer → 交织 float32 PCM
        block = sampleBuffer.mediaData()
        # 拷贝字节 → numpy float32 → 按 512 样本重组 → audio_queue

conf = ScreenCaptureKit.SCStreamConfiguration.alloc().init()
conf.setCapturesAudio_(True)
conf.setExclusivelyCapturesSystemAudio_(True)
conf.setSampleRate_(48000)
conf.setChannelCount_(2)
stream = ScreenCaptureKit.SCStream.alloc().initWithFilter_configuration_delegate_(
    filter, conf, delegate)
stream.addStreamOutput_delegate_sampleHandlerQueue_(
    output, delegate, dispatch_get_global_queue(0, 0))
stream.startCaptureWithCompletionHandler_(handler)
```

#### M 系列专属要点

1. **无需设备跟随逻辑**：SCK 捕的是全局混音而非具体设备，Windows 版 `audio_capture.py` 的"每 2s 轮询默认输出设备并重开流"（`_query_current_default`/`_restart_stream`）在 mac 上**天然免疫**，可直接移除或保留为空操作。
2. **采样率对齐**：SCK 音频通常 48kHz 浮点交错，现有 `_resample_to_mono`（多声道均值 + 线性重采样到 16k 单声道）原样复用。
3. **块大小重组**：SCK 回调块大小不定，必须在 SCK 层按 **512 样本（32ms）** 重组后再投递 `audio_queue`，维持 Silero VAD 的 16k/512 窗口契约（对应 Windows 的 `chunk_duration=0.032`）。
4. **回调线程**：`sampleHandlerQueue` 用全局队列，delegate 只做拷贝入队，不阻塞 CoreMedia 投递。
5. **arm64 原生**：PyObjC 全程 arm64，无 Rosetta 兼容问题；`pyobjc-framework-ScreenCaptureKit` 从 macOS 13 SDK 起完整覆盖 SCK API。

#### TCC 权限（M 系列）

- 系统音频捕获归入 **"屏幕录制"（Screen Recording）** 权限类：`CGRequestScreenCaptureAccess()` 触发系统弹窗；拒绝后需引导用户到 系统设置 → 隐私与安全性 → 屏幕录制 手动开启
- 麦克风独立走 **"麦克风"** 权限（PyAudio CoreAudio 输入）
- 首次启动做预检（`CGPreflightScreenCaptureAccess()`）并在 UI 给出引导；注意权限变更需重启 app 才生效（macOS TCC 行为）

#### 回退方案（可选，不作为主路径）

- **BlackHole 2ch 虚拟设备**：用户装驱动 + 系统输出设为多输出设备，程序用普通 `pyaudio`（CoreAudio）从 BlackHole 录。改动小但体验差（延迟/音量同步问题），仅保留给极旧系统或 SCK 授权失败的降级。
- **CATap（macOS 14.4+）**：能力接近 WASAPI loopback，但 API 新、PyObjC/社区实践少，不采用。

**实现结构**：`audio_capture.py` 拆出平台层，保持上层 `AudioCapture` 接口不变（`audio_queue` / `get_audio()` / `set_device()` / `set_mic_device()` / `start()` / `stop()`）：

```
audio_capture_base.py       抽象基类 + 混音/重采样/队列公共逻辑
audio_capture_sck.py        M 系列主路径（ScreenCaptureKit）
audio_capture_blackhole.py  可选回退（CoreAudio PyAudio）
audio_capture.py            平台分发（sys.platform 选择实现）
```

### 9.2 W2/W3 点击穿透

macOS 等价物：`NSWindow.ignoresMouseEvents`（PyObjC）：

```python
# 获取 NSWindow: objc bridging
import objc
from AppKit import NSWindow
view = objc.objc_object(c_void_p=int(self.winId()))
ns_window = view.window()
ns_window.setIgnoresMouseEvents_(True)   # 等价 WS_EX_TRANSPARENT
```

- overlay 的"头部可交互、正文穿透"动态切换逻辑**原样保留**：50ms 定时器改为调 `setIgnoresMouseEvents_`（开销可忽略）
- 注意：`ignoresMouseEvents=True` 时窗口完全不吃鼠标，轮询光标位置用 `QCursor.pos()` 判断即可（与现实现一致）
- 封装为 `platform_clickthrough.py`，`sys.platform == "darwin"` 分支调用 PyObjC，Windows 分支保留 ctypes；`import ctypes.windll` 必须条件化（**在 mac 上 `ctypes.windll` 属性不存在，当前代码在 import 模块时不触发、仅在调用时触发，但清查所有调用点是移植必做项**）

### 9.3 W4 torch/PyQt6 导入顺序

Windows DLL 冲突在 macOS 不存在。但**保持现有顺序无害**（先 torch 后 PyQt6），建议保留统一行为、删除平台注释即可，避免两份 main.py 分叉。

### 9.4 W5/W7 字体

| Windows 默认 | macOS 替换 |
|---|---|
| Microsoft YaHei（中文正文） | `PingFang SC`（简）/`Hiragino Sans GB`；日文 `Hiragino Sans` |
| Segoe UI（应用默认） | `.AppleSystemUIFont`（或 SF Pro Text） |
| Consolas（等宽 UI/监控条） | `Menlo` / `SF Mono` |

实现：`platform_fonts.py` 提供 `default_ui_font()` / `default_mono_font()` / `default_cjk_font()`；`DEFAULT_STYLE`、`DEFAULT_SUBTITLE_WIN_SETTINGS`、`config.yaml` 的字体字段在加载时若无显式设置则填平台值。注意 macOS 系统字体是 TTC/系统集合，QFont 按族名可用。

### 9.5 W9 GPU / 推理后端（M 系列）

| 组件 | Windows CUDA | M 系列方案 |
|---|---|---|
| faster-whisper（CTranslate2） | `cuda` + float16/int8_float16 | CTranslate2 **无 MPS 后端** → Apple Silicon **CPU int8**（ARM NEON + Accelerate，M 系性能优秀，medium 实时率充足）；`_parse_device` 走 cpu 分支 |
| SenseVoice / FunASR（torch） | cuda + fp16 | `torch.backends.mps.is_available()` → `device="mps"`；**MPS 不支持全模型 `model.half()`**（部分算子缺失），统一保持 **fp32**（M 系统一内存带宽足够，fp32 也能实时）；`_is_cuda_device` 抽象为 `_device_supports_fp16(device)`：cuda→True、mps→False、cpu→False |
| Anime-Whisper（transformers） | cuda fp16 | MPS fp32（`torch_dtype=torch.float32`），不行则 CPU；M 系 CPU 跑 transformers 亦可接受 |
| GigaAM v2/v3（transformers） | —（原版面向俄语，无 Windows 预设） | **MPS fp32 主路径**（torch 架构，M 芯片 GPU 加速）；CPU 回退；俄语专用 |
| 设备选择 UI | "cuda:0 (RTX 4090)" | **"mps (Apple Silicon)" / "cpu"**；`_parse_device` 增加 mps 分支（注意 ctranslate2 的 Whisper 引擎只收 `device="cpu"`，mps 仅对 torch 引擎生效） |
| MonitorBar GPU 显示 | `torch.cuda.memory_allocated` | MPS 无显存查询 API → 显示 "MPS" 标识，或 `torch.mps.current_allocated_memory()`（PyTorch 2.4+ 可用则用，返回统一内存占用） |

**M 系列关键点**：

1. **统一内存（UMA）**：CPU/GPU 共享内存，无显存拷贝成本，MPS 下 tensor 传递开销低；内存监控的 `psutil` RSS 已经覆盖统一内存，`torch.cuda.empty_cache()` 等价物为 `torch.mps.empty_cache()`。
2. **默认设备即 mps**：`config.yaml` 的 `device` 默认值在 mac 加载时改写为 `mps`（存在 `torch.backends.mps.is_available()` 时），否则 cpu。
3. **fp16 保守禁用**：MPS 部分 attention 算子对 fp16 报错 → 统一 fp32 最稳；后续可对 SenseVoice 单独试验 `torch.autocast(device_type="mps", dtype=torch.float16)`，稳定后再放开。
4. **arm64 wheel**：torch/torchaudio 必须装 arm64 原生轮子；`uv`/`pip` 会自动选 arm64，但需确认无 x86_64 污染（`python -c "import platform; print(platform.machine())"` 应输出 `arm64`）。

**关键正确性点**：SenseVoice 的 `_set_precision`/`_apply_model_precision`/`_autocast_context` 三个函数与 `torch.cuda.is_available()` 强耦合，移植时抽象为 `_device_supports_fp16(device)`：cuda→True、mps→False（保守）、cpu→False。

### 9.6 W10/W11 安装、启动与发布（M 系列）

- `install.sh`：`uv venv` + `uv pip install`；mac 无 winget，可选 `brew` 提示（M 系列 brew 路径为 `/opt/homebrew/bin`）；代理直接读环境变量（mac 无注册表系统代理）
- `start.sh`：激活 venv 并 `python main.py`；保留 `.livetranslate-ready` 完整性标记机制
- `update.sh`：`git pull` + 依赖更新
- 发布：GitHub Actions 增加 `macos-latest`（arm64 runner）job；便携包两种思路：
  1. **uv 管理便携 Python（arm64）**（与 Windows bootstrap.ps1 同思路的 bootstrap.sh）——体积小、无签名问题
  2. **.app bundle**（pyinstaller，需 `--target-arch arm64`）——体验最好但需 codesign+notarize，否则 TCC 权限每次重置（见 §11）
- `tests/test_startup_environment.py`/`test_requirements.py` 改为平台条件断言（Windows 校验 .ps1/.bat，macOS 校验 .sh）

### 9.7 其他小项

- `locale.getdefaultlocale()`（i18n.py）在 3.12 已弃用但可用；macOS 返回 `zh_CN` 同样工作
- `signal.SIGINT` 处理、`multiprocessing spawn`（macOS 默认即 spawn）——零改动
- `QSystemTrayIcon` 在 macOS 是菜单栏图标，行为良好；托盘菜单层级较深时 macOS 显示正常
- `Qt.Tool` 窗口在 macOS 不进 Dock/Cmd-Tab（近似"任务栏隐藏"语义）；"Taskbar" 复选框可重命名或映射为 `app.setDockIconVisible`（PyObjC `NSApplication.setActivationPolicy_`）
- 悬浮窗 `WA_TranslucentBackground` + 无边框在 macOS 正常，但**圆角窗口在旧 macOS 有四角残影**问题——容器已用 border-radius QSS，必要时加 `setMask`；Retina 下 QPixmap 缓存已用 `devicePixelRatio` 处理（字幕窗代码已正确）
- `SubtitleWindow._is_pos_visible` 多屏判断跨平台 OK
- `test_audio.py` 简单回放测试需适配平台音频 API

---

## 10. 移植路线图（分阶段）

### Phase M0 — 骨架可跑（无系统音频，麦克风 + MPS/CPU）
1. 条件化三处 Win32 调用（W2/W3 → `platform_clickthrough.py`；`ctypes` import 收进函数）
2. 字体平台化（W5/W7）
3. `requirements.txt` 拆分：`requirements-mac.txt`（去 PyAudioWPatch，加 `pyaudio`、`pyobjc-framework-ScreenCaptureKit`、`pyobjc-framework-CoreMedia`、`pyobjc-framework-AVFoundation`、`pyobjc-framework-AppKit`；arm64 原生）
4. 设备层抽象：ASR 设备枚举加 mps/cpu；SenseVoice/Nano/AnimeWhisper 精度分支抽象（`_device_supports_fp16`）
5. `install.sh` / `start.sh`
**验收**：M 系列 mac 上启动 → MPS SenseVoice / CPU Whisper 跑通"麦克风→VAD→ASR→翻译→悬浮窗"全链路

### Phase M1 — 系统音频（核心价值，ScreenCaptureKit）
6. `audio_capture_sck.py`：SCK 系统音频流（PyObjC，`exclusivelyCapturesSystemAudio`），输出对齐现有 `audio_queue` 协议；按 512 样本重组维持 VAD 契约
7. TCC 权限引导：首次启动检测屏幕录制授权（`CGPreflightScreenCaptureAccess`/`CGRequestScreenCaptureAccess`），拒绝时弹指引 + i18n 文案
8. 麦克风走 CoreAudio PyAudio；混音逻辑复用
9. （可选）BlackHole 回退模式
**验收**：播放视频 → 悬浮窗出双语字幕；切默认输出设备不中断（SCK 天然免疫）

### Phase M2 — 平台体验对齐
10. 点击穿透 PyObjC 实现（`NSWindow.ignoresMouseEvents`）与动态头部交互
11. MPS 接入验证 + 设备选择 UI（mps/cpu）；MonitorBar 适配（"MPS" 标识 / `torch.mps.current_allocated_memory`）
12. 托盘/Dock 行为（LSUIElement 可选）、.app 打包（pyinstaller `--target-arch arm64`）+ ad-hoc 签名说明
13. i18n 文案补 macOS 权限提示串（zh/en yaml）

### Phase M3 — 工程化
14. `release.yml` 加 macos-latest（arm64）job（双平台产物）
15. tests 平台条件化 + 新增 SCK 捕获单测（mock CMSampleBuffer）
16. README/CHANGELOG 双平台说明

**工作量粗估**：M0 约 2–3 天；M1 是难点，SCK+PyObjC+权限 3–5 天（含调试）；M2 2–3 天；M3 1–2 天。总体 8–13 个工作日（熟悉 PyObjC 者取下限）。

---

## 11. 风险与注意事项清单

| 风险 | 等级 | 说明与对策 |
|---|---|---|
| **TCC 屏幕录制权限** | 高 | SCK 音频捕获需用户手动授权；未签名/未公证 app 的授权会随二进制变化重置 → 分发时用 .app + 固定签名，或文档明示源码运行方式 |
| PyObjC ScreenCaptureKit 绑定成熟度 | 中高 | `pyobjc-framework-ScreenCaptureKit` 存在但示例少；备选：用 `AVCaptureScreenInput`（旧、无音频）不行——必须 SCK；或用 Swift 小工具进程输出 PCM（子进程管道，架构上正好复用 ASR worker 的进程隔离思想） |
| MPS 算子覆盖 | 中 | SenseVoice/Nano 的 attention/conv 大多可用；个别算子 fallback 报错 → try/except 降 CPU；全模型 half() 不可用，保持 fp32 |
| faster-whisper 无 GPU 加速 | 中 | Apple Silicon CPU int8 性能尚可（medium 实时率充足）；文档管理预期；也可引导用 remote-whisper 把 GPU 机器放远端 |
| `ctypes.windll` 残留调用 | 高（崩溃级） | 移植后全局搜索 `windll`，确保全部包进平台分支，否则 mac 运行时 AttributeError |
| 静音注入逻辑依赖 32ms 节拍 | 低 | SCK 回调块大小不定 → 在 SCK 层按 512 样本重组再投递，保持 VAD 窗口契约 |
| 字体回退 | 低 | PingFang 等随系统存在；用户样式串里残留 "Microsoft YaHei" 时 QFont 自动回退，不崩溃 |
| 多输出设备回退方案体验 | 中 | 仅作为可选模式，不作为主路径 |
| Notarization | 中 | 仅影响 .app 分发；源码/uv 方式不受影响 |
| 32ms 块 = Silero 原生窗口 | 提示 | 音频平台层必须维持 16k/512 样本对齐（重采样后按块切分） |
| Worker 回收/重启逻辑平台无关 | 无 | spawn 在 mac 默认且工作良好；`freeze_support` 保留无害 |

---

## 12. 附录：术语表与文件索引

| 术语 | 含义 |
|---|---|
| WASAPI loopback | Windows 录制"正在播放的系统音频"的机制；macOS 对应 ScreenCaptureKit 系统音频捕获 |
| `WS_EX_TRANSPARENT` | Win32 扩展样式位，使窗口对鼠标穿透；macOS 对应 `NSWindow.ignoresMouseEvents` |
| Silero VAD | 轻量语音活动检测模型，16kHz/512 样本窗口 |
| 输入填充桶（pad bucket） | 把音频段补齐到固定秒数倍数，稳定 ASR 分桶行为 |
| 世代（generation） | 引擎切换代数计数，用于废弃过期后台加载 |
| 签名（signature） | `(engine, model, device, hub, compute)` 元组，相同则跳过重载 |
| 增量 ASR（interim） | 说话过程中周期性识别+提交完整句并修剪音频的机制 |
| TCC | macOS 隐私权限框架（屏幕录制/麦克风授权） |
| MPS | Apple Metal Performance Shaders，torch 的 Apple GPU 后端 |

**文件索引**（行数为当前版本）：

```
main.py(2344)          入口+管线+托盘+引擎切换状态机     audio_capture.py(443)  WASAPI 捕获 ⚠️
vad_processor.py(387)  Silero VAD 切分                   asr_client.py(231)     worker 代理
asr_worker.py(202)     worker 进程入口                   asr_engine.py(157)     faster-whisper
asr_funasr.py(64)      FunASR 分发器                     asr_sensevoice.py(213) SenseVoice
asr_funasr_nano.py(135) Nano(Qwen3 内嵌)                 asr_anime_whisper.py(101) 日语动漫
asr_gigaam.py(新增)     GigaAM v2/v3 俄语专用             asr_remote.py(111)/asr_server.py(137)
translator.py(501)     LLM 翻译                           model_manager.py(749)  模型缓存/下载
subtitle_overlay.py(1286) 悬浮窗 ⚠️Win32                   subtitle_window.py(934) OBS 字幕窗 ⚠️Win32
control_panel.py       设置面板(7页)                       dialogs.py(763-)       向导/下载/模型编辑
subtitle_settings.py   字幕窗设置                          benchmark.py(180)      翻译基准
log_window.py          日志窗                              i18n.py + i18n/*.yaml  双语
transcript_writer.py   转录落盘                             funasr_nano/           Nano 模型代码(内嵌)
tests/                 4 个测试文件                         install.ps1/.bat/update.bat/start.bat/build_release.ps1  ⚠️ Windows 脚本
.github/workflows/release.yml                            ⚠️ windows-latest
```

---

### 结论

代码库中 **~85% 是纯跨平台 Python**（VAD、worker 进程架构、全部 ASR 后端逻辑、翻译、UI 主体、i18n、持久化），且架构质量高（进程隔离、自动恢复、延迟设置、代数守卫等在 mac 上可原样受益）。真正的平台壁垒集中在三处：**① WASAPI loopback（核心功能，M 系列以 ScreenCaptureKit 系统音频捕获重写）、② Win32 点击穿透（PyObjC `ignoresMouseEvents` 直接映射）、③ CUDA→MPS 推理后端差异**。

M 系列（Apple Silicon）带来了两个**明确利好**：SCK 系统音频捕获自 macOS 13 起稳定可用、天然免疫设备热切换；统一内存让 MPS 推理零拷贝、fp32 即可实时。按 §10 的四阶段路线，优先打通 M1（ScreenCaptureKit 系统音频）即可获得完整产品体验。
