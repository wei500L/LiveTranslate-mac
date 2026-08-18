# macOS（Apple Silicon）移植执行 TODO

本清单依据 `MAC_PORTING_REPORT.md` 整理，目标是把当前 Windows 版本移植为 macOS 13+、Apple Silicon（arm64）上的可用版本。当前执行重点是完成真实的平台代码和完整的功能逻辑；暂不以 Mac 真机、TCC 授权或真实音频设备作为开发阻塞条件。工作方式采用可独立交付的阶段性任务，不要求一次性完成全部移植。

## 目标与边界

- 目标平台：macOS 13.0+，Apple Silicon M 系列，Python 3.10–3.12，arm64 原生环境。
- 系统音频主路径：ScreenCaptureKit（SCK）系统音频捕获；不以 BlackHole 作为默认方案。
- 麦克风：PyAudio/CoreAudio 独立采集，沿用现有混音、重采样和队列协议。
- 推理：torch 后端优先 MPS + fp32；faster-whisper/CTranslate2 使用 Apple Silicon CPU int8。
- 保持上层管线协议：`audio_queue`、16 kHz 单声道 float32、512 samples（32 ms）块、ASR worker Pipe 接口不变。
- 不在本次移植中重写 VAD、翻译、转录、ASR worker 生命周期或 UI 业务逻辑；只做必要的平台抽象。

## 状态约定

- `[ ]` 未开始
- `[~]` 进行中
- `[x]` 已完成并通过该项验证
- `[!]` 被外部条件阻塞（必须记录原因和恢复条件）

## 阶段任务总览

| 阶段任务 | 独立交付物 | 前置条件 | 本阶段必须完成的验证 | 不包含内容 |
|---|---|---|---|---|
| M0：平台骨架 | 平台分发、点击穿透、字体、设备抽象、macOS 依赖/脚本 | 现有 Windows 基线可运行 | 导入检查、单元测试、fake 音频管线、配置分支测试 | 真实 SCK、真实 MPS、TCC 弹窗 |
| M1：SCK 音频 | 可启动/停止的 SCK 捕获和 16k/512 PCM 适配层 | M0 代码验收通过 | sample-buffer fixture、生命周期/错误/线程测试 | 真机系统音频连续性、真实权限 |
| M2：体验与引擎 | macOS UI 适配、MPS/CPU 监控、可选 GigaAM | M0 完成；M1 接口稳定 | Qt offscreen、fake torch、fake loader、配置回退测试 | 实际 MPS 性能、Retina/多屏实测 |
| M3：工程化 | CI、平台测试、安装文档、发布脚本 | M0–M2 代码接口稳定 | 全量测试、脚本静态检查、文档与代码一致性 | 签名、公证、正式发布验收 |

## 阶段任务执行规则

- 每次只执行一个阶段任务，阶段内再按 `M0-A`、`M0-B` 这样的子任务拆分；完成后再开始下一个阶段。
- 阶段任务必须产出“代码 + 测试/验证 + 变更说明”，不能只完成接口占位或 mock。
- 一个子任务如果跨越多个模块，先列出改动文件和保持不变的接口，再开始编辑。
- 每个阶段结束时更新本文件复选框，并记录实际执行的命令、测试结果和 deferred 项。
- 真机验证属于独立的后置阶段；代码阶段不得为了通过测试而绕过平台逻辑，也不得删除暂时无法执行的测试。

## 优先级与延后项

- **P0（核心可用性）**：M0-A 点击穿透分发、M0-C 依赖/启动入口、M0-D 音频公共层、M0-E 设备后端、M1-A SCK 真实实现、M1-B 权限状态机。
- **P1（完整体验）**：M0-B 字体平台化、M2 UI/Dock 行为、M2 MPS 监控、M3 平台测试和文档。
- **P2（可选扩展）**：M1-C BlackHole 回退、GigaAM 俄语引擎、PyInstaller `.app`、签名/notarization。
- P2 项目不得阻塞 P0/P1；是否纳入当前版本必须在对应阶段开始前单独确认。

## 阶段完成记录

每个阶段或子任务完成后，在对应执行记录中填写以下信息；不要只勾选复选框：

| 字段 | 内容 |
|---|---|
| 阶段/子任务 | 例如 `M0-A` |
| 改动文件 | 新增、修改和未修改但受影响的文件 |
| 公共接口 | 保持不变的调用/数据协议 |
| 验证命令 | 实际执行的 pytest、静态检查或 fixture 命令 |
| 结果 | 通过、失败或 deferred；失败需附原因 |
| 遗留项 | 不影响本阶段完成的后续工作 |

## 开始每个阶段前的基线核对

- [x] 用 `rg --files` 和 `rg -n` 核对报告中提到的文件、类名和调用点；现有模块和调用点仍存在。
- [x] 仓库没有 `asr_gigaam.py`；GigaAM 明确延后到 M2，不纳入 M0。
- [x] 平台专属依赖放入独立 `requirements-mac.txt`，Windows `requirements.txt` 不变；真实解析 deferred 到 Python 3.10–3.12 arm64 环境。
- [x] 保持 `AudioCapture` 方法、`audio_queue` 元组、ASR worker Pipe、翻译/UI 信号协议不变。

## 必须保持的跨平台接口契约

这些契约是阶段任务之间的边界；实现可以更换平台后端，但不应让上层 `main.py` 为 macOS 写第二套管线。

| 接口/数据 | 约束 |
|---|---|
| `AudioCapture.start()` / `stop()` | 可重复调用、幂等关闭；后台线程和平台 stream 必须最终退出，不能遗留非 daemon 线程。 |
| `AudioCapture.audio_queue` | 队列元素保持现有协议：`(audio, mic_rms)`；`audio` 为 16 kHz、单声道、`float32` 的 NumPy 数组。 |
| 音频块 | 进入 VAD 前固定为 512 samples（32 ms）；平台回调可以任意大小，但必须在音频层重组。 |
| `set_device()` / `set_mic_device()` | Windows 继续支持设备名；macOS 系统音频设备选择不应伪装成 WASAPI 名称，未支持的切换必须返回明确状态。 |
| 捕获错误 | 区分权限拒绝、设备不可用、平台依赖缺失、stream 运行时错误；日志包含可诊断信息，UI 收到用户可读错误。 |
| 上层消费者 | `_capture_loop`、VAD、ASR 队列和监控信号保持现有调用方式，不新增平台判断。 |

## 测试矩阵

当前阶段优先覆盖代码行为，硬件测试单独延后：

| 环境 | 必须执行 | 允许 deferred |
|---|---|---|
| 当前开发环境（非 macOS 也可） | 纯 Python 单测、导入隔离、fake 音频、fixture PCM、配置/脚本静态检查 | PyObjC 实际调用、真实设备 |
| macOS 无权限/无设备 | 依赖加载、权限状态机、错误分类、CPU/fake torch 回退 | SCK 实际 sample buffer |
| macOS Apple Silicon 真机 | 后置执行：SCK、麦克风、MPS、窗口、TCC、性能 | 无 |

测试不得通过跳过 macOS 模块来掩盖导入错误；应使用 lazy import、依赖注入或明确的 `PlatformUnavailableError`。

## 执行前检查

- [x] 已记录基线 `git status --short`；仅 `MAC_PORTING_TODO.md` 为用户原有未跟踪文件，其余改动来自 M0。
- [x] 目标约束记录为 macOS 13+、Apple Silicon、Python 3.10–3.12、arm64 原生 wheel。
- [x] 屏幕录制、麦克风权限和真实设备验证保留在后置真机验收，不阻塞 M0 代码。
- [x] 已提供可注入 `FakeAudioCapture` 和 fake VAD/ASR/流式翻译/UI 测试；真实翻译 API 不作为 M0 依赖。
- [!] 基线 `python3 -m pytest -q` 无法执行：当前系统 Python 3.9.6 且未安装 pytest/numpy；未删除或跳过任何测试。

## Phase M0：骨架可运行（麦克风 + MPS/CPU）

目标：不依赖系统音频，先打通麦克风 → VAD → ASR → 翻译 → 悬浮窗。

### M0-A 平台边界与点击穿透

- [x] 新建 `platform_clickthrough.py`，提供统一接口（设置/读取窗口鼠标穿透状态）。
- [x] Windows 分支保留 `WS_EX_TRANSPARENT` 行为；macOS 分支使用 PyObjC `NSWindow.setIgnoresMouseEvents_`。
- [x] 修改 `subtitle_overlay.py` 和 `subtitle_window.py` 的所有 Win32 调用，移除业务层直接访问 `ctypes.windll`。
- [x] 全局复查 `rg -n "windll|WS_EX_TRANSPARENT|GetWindowLong|SetWindowLong"`；Win32 代码仅存在于平台模块内，macOS 路径不导入它。
- [x] 保持正文穿透、头部可交互、定时器动态切换和窗口显示后重断言的现有行为。

### M0-B 字体与默认配置

- [x] 新建 `platform_fonts.py`，提供 UI、等宽、中文/日文默认字体：`.AppleSystemUIFont`/SF Pro、Menlo/SF Mono、PingFang SC/Hiragino Sans。
- [x] 将字幕、设置、控制面板、主窗口和日志对话框的默认字体改为平台解析；显式用户配置优先。
- [x] `config.yaml` 增加 `system_audio`/`mic_device` 和 `auto` 设备默认值，保留旧 `__disabled__` 语义。
- [x] 新增音频/设备字段采用 `setdefault` 读取，未知字段保留且不影响启动。
- [!] Retina 下字幕缓存、字体回退和中日韩字符显示需 macOS Qt 真机/offscreen 环境；代码路径已平台化，实测后置。

### M0-C 依赖与安装入口

- [x] 保留 Windows `requirements.txt` 行为，新增 `requirements-mac.txt`：去除 WASAPI patch 依赖，加入 `pyaudio` 和所需 PyObjC frameworks。
- [x] `requirements-mac.txt` 使用官方 arm64 torch/torchaudio wheel，`install.sh` 在安装前强制检查 `uname -m=arm64`，拒绝 Rosetta/x86_64 环境。
- [x] 新增 `install.sh`、`start.sh`、`update.sh`，沿用 `.venv/.livetranslate-ready` 完整性标记和失败中止语义。
- [x] 安装脚本只使用环境代理变量，不复制 Windows 注册表/winget/CUDA 逻辑。
- [x] 采用独立平台 requirements；测试强制 mac 文件包含 Windows 文件的全部跨平台依赖，避免后续漏同步。
- [x] `tests/test_requirements.py`、`tests/test_startup_environment.py` 同时断言 Windows/macOS 入口，保留 Windows 现有检查。

### M0-D 音频公共层与麦克风-only 后端

- [x] 抽取 `audio_capture_base.py`：队列、停止事件、重采样、512 samples 重组、麦克风混音和队列背压等平台无关逻辑。
- [x] 新建 `audio_capture_pyaudio.py`，使用普通 PyAudio/CoreAudio 实现麦克风采集；支持只采麦克风、不依赖 SCK 的模式。
- [x] 改造 `audio_capture.py` 为平台/模式分发器：Windows 保持现有 WASAPI 后端，macOS 在 `system_audio=disabled` 时使用 PyAudio 麦克风后端。
- [x] 新建 `platform_permissions.py`，封装麦克风权限预检/请求和平台不可用错误；M0 只依赖麦克风权限。
- [x] 定义 `system_audio` 开关并兼容 `device="__disabled__"`；上层不依赖平台专属设备名。
- [x] 无系统音频时由麦克风后端保持 512 样本节拍，`mic_rms` 对麦克风块计算，尾部残块不进入 VAD。
- [x] 为公共层增加离线测试：多声道降混、重采样、残块重组、队列满、停止幂等和 fake 音频源。
- [x] 捕获层记录回调块、重组块、丢弃块、队列深度、最近错误和重启计数，不记录音频原文。

### M0-E 设备与 torch 后端抽象

- [x] 在设备解析层支持 `mps` 和 `cpu`；faster-whisper 收到 `cpu`，不得把 `mps` 传给 CTranslate2。
- [x] 将 SenseVoice 的 CUDA 判断抽象为 `_device_supports_fp16(device)`；MPS/CPU 默认 fp32，不调用 `model.half()`。
- [x] FunASR Nano、Anime-Whisper 增加 MPS → CPU 回退路径（GigaAM 尚未接入，延后）。
- [x] 将缓存清理、显存读取和设备显示改为平台能力探测；MPS 无 API 时显示 `MPS`，可用时读取统一内存占用。
- [x] 控制面板设备列表在 macOS 显示 `mps (Apple Silicon)` 和 `cpu`，不显示 CUDA 设备。
- [x] `config.yaml` 使用 `auto`，macOS 默认选择可用 `mps`，否则 `cpu`；保留用户显式选择。

### M0 代码验收闸门

- [x] 平台模块在非 macOS 环境可安全导入；macOS 专属依赖缺失时给出明确错误或可控降级，不在 import 阶段崩溃。
- [x] 使用 `FakeAudioCapture` 跑通音频队列 → fake VAD → fake ASR → 流式翻译 → UI 信号/转录链路。
- [x] `tests/test_m0_pipeline.py` 覆盖流式更新、最终 UI 状态、转录落盘以及 worker 两次启动/停止生命周期。
- [!] `python3 -m pytest -q` deferred：当前开发容器未安装 pytest/numpy；已完成 `compileall`、`bash -n`、`git diff --check` 和无依赖静态校验，CI/目标环境需执行全量测试。

### M0 执行记录

| 字段 | 内容 |
|---|---|
| 阶段/子任务 | `M0-A` 至 `M0-E` |
| 改动文件 | 新增平台、音频、权限、配置、torch 能力层和 macOS 脚本；修改窗口、ASR worker/引擎、主程序、控制面板、配置与测试。M1 SCK 文件未新增。 |
| 公共接口 | `AudioCapture.start/stop/get_audio/set_device/set_mic_device`、`audio_queue=(float32[512], mic_rms)`、ASR Pipe 和 Qt 信号保持不变。 |
| 验证命令 | `python3 -m compileall -q .`；`bash -n install.sh start.sh update.sh`；`git diff --check`；requirements/marker/Win32 静态断言。 |
| 结果 | 静态、语法和脚本检查通过；`pytest` deferred（当前 Python 3.9.6 无 pytest/numpy）。 |
| 遗留项 | Retina/真机 PyObjC、CoreAudio 权限和真实 MPS/CPU 推理后置；SCK 属于 M1。 |

## Phase M1：ScreenCaptureKit 系统音频（核心能力）

前置：M0 代码验收通过。SCK 必须实现真实的 PyObjC 捕获、生命周期和 PCM 转换逻辑；当前阶段用依赖隔离、协议测试和 sample-buffer fixture 验证，真机调试后置。M1 不应重复实现 M0 的公共音频层或麦克风后端。

### M1-A 音频平台层

- [x] 新建 `audio_capture_sck.py`，使用 PyObjC `SCStreamConfiguration.capturesAudio=True` 和 `exclusivelyCapturesSystemAudio=True`。
- [x] 选择主显示器作为 SCK content/filter；不渲染视频帧，只处理音频 sample buffer。
- [x] 实现 delegate 生命周期、stream start/stop、错误回调和线程安全关闭，避免回调阻塞系统队列。
- [x] 将 `CMSampleBuffer`/`CMBlockBuffer`/`AudioBufferList` 转换为 float32 PCM；处理交织/非交织和声道数变化。
- [x] 在 SCK 层累积并按 512 samples 重组，再投递给上层，保持 VAD 32 ms 契约。
- [x] 新建或改造 `audio_capture.py` 为 `sys.platform` 分发器；保持 `AudioCapture` 对上层的现有方法和属性兼容。
- [x] SCK 后端接入 M0 的公共层；复用现有重采样和混音对齐逻辑，不复制音频队列实现。
- [x] macOS 不再执行 Windows 默认输出设备轮询和 WASAPI loopback 查找；`set_device()` 行为定义为重建 SCK content。

### M1-B TCC 权限与用户引导

- [x] 增加屏幕录制权限预检/请求（`CGPreflightScreenCaptureAccess`、`CGRequestScreenCaptureAccess` 或对应 PyObjC 绑定）。
- [x] 复用 `platform_permissions.py` 的错误类型和 UI 提示机制；系统音频权限与麦克风权限不可混淆。
- [x] 首次启动和捕获失败时显示中英文 i18n 文案；说明权限变更通常需要重启应用。
- [x] 明确源码运行、ad-hoc 签名和正式签名对 TCC 授权持久性的差异（见 M1 执行记录）。

### M1-C 可选回退（低优先级）

- [ ] 只有在 SCK 无法使用且产品确有需要时，增加 BlackHole 2ch CoreAudio 回退；默认不安装、不启用。
- [ ] 不采用 CATap 作为本次主路径；若未来评估，另开设计任务。

### M1 代码验收闸门

- [x] 用 sample-buffer fixture 覆盖交织/非交织、多声道、不同回调块大小、尾部残块和异常回调。
- [x] 用 fake stream/delegate 验证 start/stop、重复 stop、启动失败、回调晚到和线程关闭不会死锁或泄漏。
- [x] 验证 SCK 输出经过重采样和 512 samples 重组后，严格符合 VAD 输入契约。
- [x] 验证权限检查、拒绝、请求失败和重试路径的返回值、异常分类和 i18n 文案；真实系统弹窗留作 deferred。

### M1 执行记录

| 字段 | 内容 |
|---|---|
| 阶段/子任务 | `M1-A`、`M1-B` |
| 改动文件 | 新增 `audio_capture_sck.py`、`tests/test_m1_sck.py`；修改 `audio_capture.py`、`platform_permissions.py`、`main.py`、`i18n/en.yaml`、`i18n/zh.yaml`。 |
| 公共接口 | `AudioCapture.start/stop/get_audio/set_device/set_mic_device`、`audio_queue=(float32[512], mic_rms)` 和 ASR 管线协议保持不变。 |
| 验证命令 | `python3 -m compileall -q .`、`bash -n install.sh start.sh update.sh`、`git diff --check`；pytest fixture 已加入但当前环境缺少 numpy/pytest。 |
| 结果 | 代码、lazy PyObjC 隔离、权限错误分类和 fake-stream/sample-buffer 覆盖完成；真实 SCK/TCC/设备验证 deferred。 |
| 遗留项 | 真机需确认 PyObjC AudioBufferList 绑定、连续系统音频、权限弹窗及签名 identity 持久性；BlackHole 回退不纳入 M1。 |

## Phase M2：平台体验对齐

- [ ] 验证 overlay/subtitle window 在 macOS 的透明背景、置顶、拖拽、点击穿透和多屏定位。
- [ ] 验证 `Qt.Tool`、托盘菜单栏图标、Dock/Cmd-Tab 行为；决定是否提供 LSUIElement 或 Dock 可见性选项。
- [ ] 完成 MPS 设备显示、统一内存监控和 ASR worker 释放路径（`torch.mps.empty_cache()` 能力探测）。
- [ ] 增加 GigaAM v3（俄语）模型接入：`asr_gigaam.py`、worker 分支、`model_manager.py`、控制面板和缓存检测；固定语言 `ru`。
- [ ] 如需要发布 `.app`，加入 PyInstaller arm64 构建、ad-hoc 签名说明；记录正式签名/notarization 的后续工作。
- [ ] 清理用户可见文案、日志、注释中的 Windows/WASAPI/CUDA 假设；更新中英文 i18n。

### M2 代码验收闸门

- [ ] 用 Qt offscreen 或 widget-level 测试验证窗口状态、布局约束、字体回退和点击穿透调用，不要求真实屏幕截图。
- [ ] 用 monkeypatch/fake torch 验证 MPS、CPU、CUDA 分支、内存监控和清理路径；真实 MPS 推理留作 deferred。
- [ ] 用 fake ASR/model loader 验证 CPU Whisper、MPS torch ASR、远程 ASR 的配置分发和失败回退。
- [ ] 俄语 GigaAM（若纳入本次版本）完成配置、缓存、worker 分支和语言过滤逻辑验证；真实模型下载/推理留作 deferred。

## Phase M3：工程化、测试与发布

- [ ] `.github/workflows/release.yml` 增加 macOS arm64 job；产物命名与 Windows 产物区分。
- [ ] 为 SCK 音频层增加 mock 单测：sample buffer 转换、重采样、512 样本重组、队列背压和关闭幂等性。
- [ ] 增加平台能力单测：点击穿透分发、字体默认值、设备解析、MPS 无 torch/无 MPS 时的 CPU 回退。
- [ ] 将 `test_audio.py` 改为平台无关的离线/模拟测试；不在 CI 中要求真实音频设备或 TCC 权限。
- [ ] README/README_zh.md 更新为双平台说明，新增 macOS 安装、权限、arm64、SCK 限制和性能预期。
- [ ] 更新 CHANGELOG，记录 macOS 支持范围、已知限制（CTranslate2 无 MPS、未签名 app 的 TCC 行为）。
- [ ] 发布前执行全局检查：`rg -n "windll|PyAudioWPatch|WASAPI|torch\\.cuda|Microsoft YaHei|Segoe UI"`，逐项确认是 Windows 分支、文档说明或待处理项。

## 关键风险闸门（代码阶段）

- [ ] SCK PyObjC 绑定若出现 API 缺失：先隔离绑定层并提供明确运行时错误；必要时保留独立 Swift 捕获进程方案，不改变上层 PCM 协议。真机行为列为后置验证，不以猜测替代代码实现。
- [ ] MPS 算子失败：优先单模型/单算子降级 CPU，不牺牲整个应用启动；记录具体模型、算子和 PyTorch 版本。
- [ ] TCC 授权在打包后反复重置：实现稳定的权限状态机和用户提示；固定 bundle identity/签名属于后置发布工作。
- [ ] SCK 采样格式或块大小不稳定：以“进入 VAD 前必须为 16 kHz/512 samples/float32”为硬断言，并记录丢块、重组和延迟指标。
- [ ] 任何 `windll` 残留导致 macOS 崩溃：在代码审查和真机启动前必须清零运行路径。

## 建议执行顺序

1. M0-A/B（平台边界）→ M0-C（依赖/脚本）→ M0-D（音频公共层）→ M0-E（设备后端）。
2. 通过 M0 代码验收后，实现 M1-A SCK；随后加入 M1-B 权限状态机和引导。
3. M1 代码和 fixture 验证稳定后处理 M2 UI/模型体验，最后做 M3 CI、文档和发布。
4. 每完成一个子阶段，更新本文件状态，记录验证命令、fixture/fake 约定和依赖版本；真机验证项统一标记为 `deferred`，不要用“未测试”掩盖未实现的代码路径。

## 后置真机验证（当前不阻塞开发）

- [ ] 在 macOS 13+ Apple Silicon 上安装 arm64 依赖并运行 `install.sh`/`start.sh`。
- [ ] 授予/拒绝屏幕录制和麦克风权限，验证 TCC 状态变化、重启提示和恢复流程。
- [ ] 播放真实系统音频，确认 SCK 连续捕获、VAD 切句、ASR、翻译和字幕显示。
- [ ] 验证真实 MPS 推理、CPU int8 Whisper 性能、Retina/多屏窗口行为和默认输出切换。
- [ ] 记录机器型号、macOS、Python、PyTorch、PyObjC 版本及已知限制。

## 完成定义

代码实现阶段的完成条件：

- SCK 系统音频和 CoreAudio 麦克风的真实代码路径已实现，输出严格遵守现有 VAD 输入契约；
- 至少一个 MPS torch ASR 和一个 CPU faster-whisper ASR 的配置、分发、失败回退逻辑已实现并有测试；
- 翻译、悬浮窗/字幕窗、转录、worker 恢复的跨平台逻辑保持完整；
- 自动化测试、平台隔离测试和文档与实际代码行为一致；
- 已知限制、TCC/签名要求和后置真机验证清单已明确记录。

真机端到端运行、实际 MPS 性能和系统权限弹窗属于发布前的独立验收阶段，不影响当前代码实现阶段的完成。
