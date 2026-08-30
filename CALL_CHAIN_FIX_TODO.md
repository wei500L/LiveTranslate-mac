# 调用链问题修复实施 TODO

依据：`CALL_CHAIN_AUDIT.md`（2026-08-23 第一轮）+ `STATIC_AUDIT_2026-08-29.md`（2026-08-29 第二轮，含三批发现）。

本文是纯实施方案。执行重点是：明确修改位置、复用现有入口、规定异常和状态如何流转，并通过代码审查、日志和实际操作确认结果。

**覆盖范围**：两轮审查共 **72 项**问题（第一轮 13 项全部未修，第二轮新增 59 项），实施阶段又新发现
**1 项**（N-P1-14），合计 **73 项**，归并为 **27 个任务**（D1 + A1~A10 + B1~B9 + C1~C7）。

**当前进度**：**27 个任务全部完成**。详见 §10 执行记录。测试 81 → 190 全绿（另有 1 项在未安装
FastAPI 的环境下 skip）。第二阶段（A3、A4、A7~A10、B1~B4、B6~B9、C1~C7 余项）在 `main` 上继续实施。

**归并原则**：多项问题落在同一函数上时合并为一个任务，避免同一处代码被两次修改且意图冲突。受影响的合并点：

| 代码位置 | 合并的问题 | 任务 |
|---|---|---|
| `LiveTranslateApp.stop()` 及退出入口 | P1-5、P1-6、N-P1-6、N-P2-3、N-P3-5 | A3 |
| `SCKAudioCapture.set_device()` | P1-3、N-P1-2 | A5 |
| `ASRClient` 锁与状态 | N-P1-4、N-P2-5 | A7 |
| `Translator` 结果收口 | P2-1、N-P2-2、N-P3-2 | B1 |
| 增量 ASR 提交循环 | N-P1-3、N-P3-6 | B6 |
| `ControlPanel` 模型列表 | P2-2、N-P3-7 | C1 |
| `mlx_service` 生命周期与状态查询 | N-P1-7、N-P1-8、N-P1-13、N-P2-6～9、N-P2-19、N-P3-9 | A8 |
| 下载对话框线程与全局状态 | N-P1-9、N-P2-10、N-P3-10～12 | A9 |
| `audio_capture._read_loop` | P1-4、N-P2-21、N-P3-20 | A4 |
| `benchmark._test_model` 请求循环 | P2-3、N-P2-11～13 | B2 |
| `subtitle_window` 渲染与重建 | N-P2-17、N-P3-16、N-P3-17 | B8 |
| `asr_server` 配置与暴露面 | P2-4、N-P2-14、N-P3-13 | C2 |
| 用户可见文案与 i18n 健壮性 | N-P2-4、N-P2-16、N-P3-4、N-P3-8、N-P3-15、N-P3-18 | C5 |
| CI 与测试基线 | N-P1-10、N-P1-11、N-P1-12、N-P1-14、N-P3-23 | D1 |
| 依赖声明与平台对称性 | N-P2-22、N-P2-23、N-P3-21、N-P3-22 | C7 |

## 1. 目标和不变项

### 1.1 目标

稳定以下主链路：

```text
AudioCapture -> _capture_loop -> VAD -> _asr_queue -> _asr_loop
-> ASRClient/RemoteASREngine -> _process_segment*
-> Translator.translate_iter -> Overlay/SubtitleWindow/TranscriptWriter
```

必须达到：

- 单个坏结果、远程服务一次故障或一次队列背压不会永久杀死采集/ASR 链路。
- 托盘退出、Overlay 退出、控制面板退出和缓存删除并退出使用同一清理路径，且不无限等待。
- 翻译、Benchmark、字幕窗口和模型选择使用一致的请求、结果和状态语义。
- 失败必须有日志和明确状态；合法空语音结果仍可安静丢弃。
- **跨线程共享状态只在其约定的锁下访问**，不存在「一部分调用点加锁、另一部分不加」的分裂。
- **一次设置变更只做实际发生变化的事**，不因发送全量设置字典而触发无关的重启或重载。

### 1.2 不变项

- 不修改 ASR worker Pipe **消息类型和字段**（`id`/`type`/`payload`/`ok`/`error`），不改 `/transcribe` 二进制协议。
  A7 允许重构 `ASRClient` 的**锁与状态字段**，但对端 `asr_worker.py` 的消息语义必须保持不变。
- 不修改 `(audio, mic_rms)`、16 kHz、单声道、512 samples 音频契约。
- 不修改 `Translator.translate_iter()` 生成器接口、Qt 信号和 transcript 文件格式。
- 不新增通用调度器、状态机框架、重试库、第二套 ASR 管线或第二套翻译请求构造器。
- 不重复 `MAC_PORTING_TODO.md` 中已经完成的平台移植工作。
- 不改变 VAD 的切分算法（progressive silence、backtrack split、密度阈值）；A6 只处理并发访问，不调参。

## 2. 共享契约

### 2.1 ASR 结果契约

在 `main.py` 增加一个轻量内部校验入口，三个结果消费点统一使用：

- `None`：合法空语音，直接跳过，不显示错误。
- 非 `dict`：协议错误，记录 `seg_type` 和实际类型，丢弃当前结果。
- `text`：必须是字符串；空字符串可作为合法空语音，缺失或错误类型是协议错误。
- `language`：必须是非空字符串；缺失或错误类型是协议错误。
- `language_name`、`words` 等仅为可选字段。

本契约同时适用于本地 worker 和 `RemoteASREngine`，不允许两个后端各自返回不同的半有效结构。

### 2.2 远程 ASR 异常契约

`RemoteASREngine` 只有三种出口：

1. 合法响应、空文本：返回 `None`。
2. 合法响应、非空文本：返回符合 2.1 的字典。
3. 网络、HTTP、JSON 或字段协议异常：抛出 `RemoteASRError`，保留阶段和状态码，不返回 `None`。

### 2.3 停止契约

`LiveTranslateApp.stop()` 是唯一 pipeline 清理实现，必须幂等、有界、完整：

- 未启动、已停止、重复调用都不抛异常，重复调用只补充残余清理。
- 不使用无限阻塞队列操作；线程 join 沿用现有超时。
- 音频、ASR、翻译 executor、transcript、监控定时器和 MLX service 都有清理出口。
- **任何一步的等待都必须有界**：包括等待 `ASRClient` 内部锁的时间（见 2.6）。
- **信号处理器不直接执行清理**：`SIGINT` 只投递退出请求，实际清理在 Qt 事件循环中执行一次。

### 2.4 统一状态流转

| 场景 | `_running` | `_asr_ready` | UI 状态 | 后续动作 |
|---|---:|---:|---|---|
| 正常运行 | `True` | 依实际情况 | Running + 引擎 | 持续采集/处理 |
| 用户暂停 | `True` | 保持 | Paused | 不处理暂停期间音频 |
| 延迟启动未执行 | `False` | 原值 | Paused/未启动 | 启动回调可取消 |
| 远程 ASR 故障 | `True` | `False` 或不可用态 | ASR unavailable | 使用现有恢复上限 |
| 翻译服务不可用 | `True` | 保持 | 每条消息显示明确错误 | ASR 继续，翻译显式失败 |
| SCK 切换失败但旧流恢复 | `True` | 保持 | 旧设备 + 错误提示 | 继续旧设备 |
| SCK 无法恢复 | `False` | 按清理结果 | Stopped/错误 | 统一 stop |
| 正在退出 | `False` | 清理中 | 不再启动 | 只做清理 |

`_running=True` 不等同于 ASR 已就绪；`_asr_ready=False` 也不能被当作合法空语音无限吞掉。
**翻译器为 `None` 不能被当作「执行器已关闭」静默跳过**——它必须在 UI 和 transcript 上落定为一次明确失败。

### 2.5 统一停止顺序

```text
_running=False
  -> 停止 AudioCapture，等待 capture thread（有界）
  -> 非阻塞投递 ASR 哨兵，等待 ASR thread（有界）
  -> 仅在 ASR 仍可用时 flush VAD
  -> 等待 translation executors（包含 flush 产生的翻译）
  -> 关闭 transcript
  -> 停止监控定时器
  -> shutdown ASR client/worker
  -> 停止 MLX service
```

某一步失败时记录错误并继续后续清理，不能因线程超时跳过 worker、文件或服务回收。

### 2.6 ASRClient 锁职责契约（新增）

`ASRClient` 内部区分两类临界区，**不得共用一把锁**：

- **IO 锁**：序列化 Pipe 上的请求/响应往返。只有 `transcribe`/`set_language`/`set_input_padding`/`wait_ready` 持有，持有时间可长达 `request_timeout`。
- **生命周期锁**：保护 `_process`、`_conn`、`_status` 字段本身。任何持有时间都必须是常数级。

规则：

- `shutdown()`/`terminate()` **禁止等待 IO 锁**。它们置取消标志、直接中止子进程以打断阻塞中的接收，然后只在生命周期锁下清理句柄。
- `_recv_response` 的轮询循环每轮检查取消标志，被取消时抛出 `ASRWorkerExited`。
- `status` 属性在任何时刻都不得报告一个已不存在的进程为 `"ready"`。

### 2.7 VAD 并发契约（新增）

`VADProcessor` **不是**线程安全类，其所有公有和内部状态访问都必须在 `LiveTranslateApp._vad_lock` 下进行。

- 约定写入 `vad_processor.py` 的类 docstring，避免后续调用点再次遗漏。
- `_reset()` 必须原子地替换全部相关状态；`_speech_buffer` 与 `_confidence_history` 的长度不变式在任何可观察时刻都必须成立。
- 只读探针（`last_confidence`、监控面板取值）可不加锁，但不得据此做切分决策。

### 2.8 设置变更契约（新增）

`ControlPanel._apply_settings()` 发出的是**全量**设置字典，因此 `_on_settings_changed` 的每个分支：

- **必须比较值**（`old != new`），不得用 `key in settings` 作为「用户改了这一项」的判据。
- 触发重启/重载类副作用（音频设备、ASR 引擎、翻译器）前必须先确认值确实变化。
- 被调用的 setter 自身也应幂等：传入与当前相同的值时不产生副作用。

### 2.9 后台任务与 UI 线程契约（新增）

任何**阻塞 IO 或进程派生**都不得在 Qt 主线程执行。具体包括：`urlopen`/`httpx` 请求、`subprocess.run`/`Popen`、`ps`、目录遍历（`rglob`/`dir_size`）、`shutil.rmtree`、线程 `join`。

- 需要这些结果的 UI 状态，一律由后台线程产出、经信号回填，UI 只读缓存值。
- 每个后台任务类型必须有**去重保护**（形如 `if task is not None and task.isRunning(): return`），不得每次触发都派生新线程。
- 长任务必须**可取消**，且取消标志的检查点不能依赖外部进程是否恰好产生输出。
- 项目已有正确样板：`_MLXHealthThread` + `request_mlx_health_check()`（`control_panel.py:119`、`1582`）。新增后台任务复用该模式，不再新造第三种。

### 2.10 全局解释器状态契约（新增）

`sys.stderr`、`sys.stdout`、`logging` root handler 属于进程级全局状态。任何对它们的替换必须：

- 成对安装与恢复，且恢复位于 `finally` 或上下文管理器中，不能只写在「正常完成」分支里。
- 在持有者（对话框/窗口）被关闭或销毁时也执行恢复。
- 替换对象必须实现被替换者的完整最小接口（`write`、`flush`、`isatty`、`fileno`、`encoding`、`errors`）。
- 替换期间跨线程回调不得直接引用可能已销毁的 Qt 对象。

### 2.11 依赖声明契约（新增）

- `requirements.txt` / `requirements-mac.txt` 必须能**独立**装出可运行环境；做不到时必须在文件内以注释写明缺什么、由谁安装（参照现有 torch 注释的写法）。
- 两个平台文件的差异必须是**有意且有注释**的（如 `PyAudioWPatch` 仅 Windows）。非平台特有的依赖不得只出现在一侧。
- 代码中未 import 的依赖不得声明，除非注明它是某个直接依赖的必需传递依赖及原因。
- 版本约束在两个平台上的上界策略应一致；一侧钉死、另一侧无上界属于需要修正的不对称。

### 2.12 测试与发布门禁契约（新增）

- 仓库任一提交上 `python -m pytest -q` 必须通过；测试失败即视为构建失败。
- CI 安装的依赖必须覆盖测试实际 import 的全部模块；若刻意不装，对应测试必须显式 skip 而非收集失败。
- **所有**产出发布物的作业都必须 `needs` 测试作业。平台之间的发布门禁不得不一致。
- 涉及跨文件文本契约的断言（如启动器提示文案），两个平台的实现必须使用同一措辞，或测试同时接受两种措辞。

## 3. 执行顺序

```text
阶段 D：测试与发布基线     D1                          <- 必须最先做
阶段 A：存活性与退出       A1 ~ A10
阶段 B：翻译和显示一致性   B1 ~ B9
阶段 C：配置、服务与依赖   C1 ~ C7
阶段 E：代码审查和交付核对
```

阶段内的任务编号不代表优先级。**实际动手顺序按下表**：

| 顺位 | 任务 | 状态 | 理由 |
|---:|---|---|---|
| **0** | **D1** | ✅ | **前置条件**。当时 `pytest -q` 就是失败的（`start.bat` 文案回归），CI 又装不全依赖、且发布不受测试门禁约束。没有可信的绿色基线，后面 26 个任务的「运行检查点」全都无法证伪。实施中还暴露出第 4 个 CI 缺陷（N-P1-14）。 |
| 1 | **A6、A5** | ✅ | 两者都由 `_on_settings_changed` 触发，是日常使用中最高频的故障路径，改动局部、风险低。 |
| 2 | **A1、A2、B5** | ✅ | 统一 ASR/翻译的结果契约与失败可见性，消除「静默无输出」这一最难排查的故障类别。 |
| **3** | **A7、A3** | ✅ | 一并重做停止/切换协议：锁分离 + 有界等待 + 单一退出入口。A7 先于 A3 完成。B3 因与 A3 共用 `main()` 的退出/启动闭包，一并处理。 |
| 4 | **A8** | ✅ | MLX 生命周期与状态查询。状态查询全部移入后台线程，取消在静默期同样生效，不再删除运行中的模型目录。 |
| 5 | **A4、A9、A10** | ✅ | 采集读循环、下载对话框、后端错误契约。A4 与 A10 共同确立 `get_audio()` 的三态契约。 |
| 6 | **B6** | ✅ | 增量 ASR 短句去重与密度丢弃后的状态复位。 |
| 7 | **B1 ~ B4、B7 ~ B9** | ✅ | 翻译请求语义、计数、启动状态与字幕渲染。 |
| 8 | **C1 ~ C4、C6**，及 C5/C7 余项 | ✅ | 配置边界、服务暴露面、文案与可维护性收尾。 |

依赖约束（均已满足，保留以记录顺序理由）：

- ~~**D1 先于一切**~~：已完成，基线已绿（190 项测试通过）。「CI 能收集并运行全部测试」一条现已**在等价环境中证实**——见 §10「CI 等价环境验证」。
- A1 先建立统一结果契约，A2 的错误分类依赖它。
- **A7 必须先于 A3 完成**：A3 要求 `stop()` 有界，而当前无界等待正来自 `ASRClient` 的单锁设计。
- **A8 的取消修复先于 N-P2-19 的验证**：设置面板卡死是取消无效的直接后果，不是独立缺陷。
- B5 依赖 A1 的结果契约入口（复用同一处失败落定逻辑）。
- C7 建议先于 D1 的最终验收再跑一次，确保 CI 依赖清单与 requirements 同源。
- 其余任务无强依赖；不要求为每个任务创建独立提交，但修改边界必须清楚。

## 3.5 阶段 D：测试与发布基线

### D1｜恢复绿色测试基线与发布门禁（N-P1-10、N-P1-11、N-P1-12、N-P3-23、N-P1-14）

**状态**：✅ 已完成


**涉及文件**：`start.bat`、`.github/workflows/release.yml`、`tests/test_requirements.py`；可能涉及 `translator.py`（惰性导入方案）。

**根因**：三个互相掩盖的问题。测试本身是红的，CI 收集不到一部分测试，而发布又不受测试结果约束——任何一个单独出现都会被发现，三个叠加就形成了「没人看见的红」。

**实施步骤**

*测试回归（N-P1-10）*

- [x] 统一两个启动器的提示文案。`start.sh:8` 为 `"Setup is incomplete; running the installer first..."`，`start.bat:12` 在 `cbd3a53` 中被改成 `"Environment is incomplete; ..."`。二选一并同步另一侧；若刻意保留分歧，`tests/test_startup_environment.py:15` 的断言必须同时接受两种措辞，并在测试中注明原因。
- [x] 确认 `python -m pytest -q` 返回 0。

*CI 依赖（N-P1-11）*

- [x] 二选一并记录理由：
  - **方案 A**：CI 的「Install test dependencies」补上 `httpx`、`openai`。简单，但 CI 依赖清单会与 `requirements.txt` 二次漂移。
  - **方案 B（推荐）**：把 `translator.py` 的 `import httpx` / `from openai import OpenAI` 改为函数内惰性导入，使 thinking style 解析、prompt 组装、请求 kwargs 这些纯逻辑测试无需第三方包即可运行。附带收益是启动更快。
- [x] 无论选哪个方案，都必须确保**不存在收集失败**：刻意不安装的依赖，对应测试要显式 `pytest.importorskip` 而不是报错。
- [ ] 用一个只装 CI 声明依赖的干净虚拟环境本地验证 `pytest -q` 通过。

*发布门禁（N-P1-12、N-P3-23）*

- [x] 给 `build` 作业加 `needs: test-macos-arm64`，与 `package-macos-arm64` 保持一致。
- [x] 在 `tests/test_requirements.py::test_release_workflow_has_arm64_test_and_distinct_macos_artifact` 中补断言：**每个产出发布物的作业都声明了 `needs`**。这条断言的缺失正是本问题未被捕获的原因。
- [x] 不改变现有的产物命名和 tag 触发条件。

*环境依赖型测试（N-P1-14，实施阶段新发现）*

审查阶段未捕获，实施 D1 时才暴露：`tests/test_m2_platform.py::test_gigaam_cache_is_hf_snapshot_only`
断言 `get_missing_models("gigaam", "", "hf") == []`，但 `get_missing_models` 会先检查 Silero VAD，
而 `is_silero_cached()` 走 `_has_silero_pkg()` → `importlib.util.find_spec("silero_vad")`。
CI 不安装 silero-vad，因此该调用返回 `['Silero VAD']`，断言必然失败。
**这意味着即使修完 N-P1-10/11/12，CI 仍然是红的。**

- [x] 让该测试不依赖「silero-vad 恰好已安装」——monkeypatch `_has_silero_pkg`，因为测试意图是
  GigaAM 缓存检测，Silero 的安装状态是无关噪声。
- [ ] 排查是否还有其他测试隐含依赖某个包恰好安装。本轮只处理了这一处，未做系统性扫描。

**运行检查点**：干净环境下 `pytest -q` 返回 0；在一个测试故意失败的分支上打 tag 或 workflow_dispatch，确认 `build` 作业被跳过而非产出发布包。

**完成标准**：仓库任一提交上测试可通过；CI 能收集并运行全部测试；测试失败时两个平台都不产出发布物。

## 4. 阶段 A：存活性与退出

### A1｜ASR 单项异常隔离与结果校验（P1-1）

**状态**：✅ 已完成


**涉及文件**：`main.py`。

**根因**：`_asr_loop` 只处理 `queue.Empty`；结果消费位置直接访问 `result["text"]`/`result["language"]`，坏结果或分句异常会退出 ASR 线程。

**实施步骤**

- [x] 增加统一结果校验函数，返回空结果、规范化结果或协议错误三种明确结果。
- [x] `_process_segment`、`_process_interim_final`、`_do_interim_asr` 在 `.strip()`、语言过滤和分句前调用该函数。
- [x] `_asr_loop` 对单个队列 item 建立外围 `try/except Exception`，日志包含 `seg_type`、异常类型和 segment 长度，不记录音频原文。
- [x] 协议错误只丢当前 item；worker 退出/超时继续由 `_run_asr` 的既有恢复路径处理。
- [x] 不捕获并吞掉 `SystemExit`/`KeyboardInterrupt`，不在该层重启应用。

**运行检查点**：日志中能区分空结果、协议错误和 worker 错误；坏结果之后 ASR loop 仍处于运行状态。

**完成标准**：坏结果不杀死 ASR 线程，合法空结果不产生故障提示，正常字幕和转录行为不变。

### A2｜远程 ASR 领域异常和恢复（P1-2）

**状态**：✅ 已完成


**涉及文件**：`asr_remote.py`、`main.py`；必要时更新 `REMOTE_ASR.md`。

**实施步骤**

- [x] 定义 `RemoteASRError`，保留原始异常作为 cause；消息包含请求阶段、主机或 HTTP 状态，不泄露凭据。
- [x] 分开处理 post、HTTP 状态、JSON 解析、响应类型和字段校验，移除宽泛的 `return None`。
- [x] 空文本只在合法响应和合法语言字段下返回 `None`；非字典 JSON、缺字段、类型错误全部抛异常。
- [x] `_run_asr` 捕获 `RemoteASRError`，沿用 `_recover_asr_worker()` 的重建次数和状态显示，不新增第二个重试循环。
- [ ] 特别确认 remote client 的 `pid=None`、`shutdown()`、旧 client 替换和 `_load_engine_client()` 重建顺序，避免旧请求覆盖新连接。
- [ ] 恢复失败显示 `ASR unavailable`；恢复成功更新 Overlay 引擎状态；合法空文本不触发恢复。

**运行检查点**：远程服务断开时日志有明确请求错误，Overlay 不再显示正常运行中的 ASR；服务恢复后重新产出字幕。

**完成标准**：网络故障不再静默吞掉，空语音仍保持静默。

### A3｜有界停止与统一退出入口（P1-5、P1-6、N-P1-6、N-P2-3、N-P3-5）

**状态**：✅ 已完成


**涉及文件**：`main.py`、`control_panel.py`。

**根因**：退出路径分散且无界——`stop()` 用阻塞 `put(None)`；缓存删除直接调 Qt `quit()` 绕过 `stop()`；`SIGINT` 在信号处理器里做多秒清理；`_enqueue_asr` 的丢弃逻辑会取出并解引用哨兵 `None`。

**实施步骤**

*停止本体*

- [ ] 为 `stop()` 增加幂等保护；重复调用只补充残余清理，不重复 flush/close。
- [ ] 将 `_asr_queue.put(None)` 改为 `put_nowait()`；队列满时有限次丢弃待处理项，再尝试投递哨兵；失败记录日志并依靠 `_running=False` 退出。
- [ ] 保留现有 capture/ASR join 超时；超时后继续关闭 executor、transcript、ASR client 和 MLX service。
- [ ] 停止音频和 capture thread 后再处理 ASR 哨兵；停止期间禁止新的 capture item 写入 ASR 队列。
- [ ] VAD flush 只在 ASR 仍可用且不会创建不可执行任务时进行；flush 产生的翻译完成后再关闭 translation executor。

*哨兵与队列（N-P2-3）*

- [ ] 优先改用独立的停止 `threading.Event` 表达退出，`_asr_loop` 以短超时轮询该事件；若保留队列哨兵，`_enqueue_asr` 的丢弃分支必须先判 `dropped is not None` 再取下标。
- [ ] `_enqueue_asr` 整体不得抛出未捕获异常到 `_capture_loop`。

*退出入口收敛（P1-5、N-P1-6）*

- [ ] `ControlPanel` 增加最小退出信号或回调，不在面板内直接调用 `QApplication.quit()`。
- [ ] 缓存删除确认后的顺序固定为：**先触发统一退出 → 等待 pipeline 清理完成（含 ASR worker 已退出）→ 再删除缓存目录 → 最后 quit**。删除前必须确认 worker 进程已不存在。
- [ ] `shutil.rmtree` 移出 Qt 主线程，执行期间显示进度或忙状态。
- [ ] 删除失败必须在 UI 上报错并列出失败路径，不能只写日志。
- [ ] 托盘、Overlay、控制面板关闭和 `aboutToQuit` 统一走一个 `on_quit`/stop-once 入口；`aboutToQuit` 仅作兜底。

*信号处理（N-P3-5）*

- [ ] `SIGINT` 处理器只设置退出标志并唤醒事件循环（如 `QTimer.singleShot(0, on_quit)`），不在处理器内执行 join、文件关闭或子进程回收。
- [ ] 第二次 `SIGINT` 不重复进入清理；依赖 `stop()` 的幂等保护。

**运行检查点**：日志按固定顺序出现停止、线程等待、transcript close、ASR shutdown；缓存删除并退出不再直接调用 Qt quit；Ctrl-C 两次不产生重复清理日志。

**完成标准**：满队列、ASR 线程已退出、重复退出、Ctrl-C 都能在有界时间内结束，不留下后台线程或 worker；缓存删除时没有进程仍持有被删目录。

### A4｜Windows 采集读循环：队列背压、重启失败与持锁阻塞（P1-4、N-P2-21、N-P3-20）

**状态**：✅ 已完成


**涉及文件**：`audio_capture.py`；必要时只在 `audio_capture_base.py` 增加最小共享 helper。

**根因**：三处缺陷都在同一个 `_read_loop` 函数内，必须一次改完，否则会互相干扰。

*队列背压（P1-4）*

- [ ] 使用 `AudioCaptureBase._enqueue()` 的丢最旧策略替换 Windows read loop 中直接 `get_nowait()`/`put_nowait()` 组合（`audio_capture.py:434-438`）。
- [ ] 注意实际缺陷比第一轮记录的更宽：**两个** `put_nowait` 都无保护。`get_nowait()` 可抛 `queue.Empty`，其后的 `put_nowait()` 可再抛 `queue.Full`，二者都会逃逸到无顶层保护的 `_read_loop`。基类实现（`audio_capture_base.py:119-127`）把两者一并包在 `except (queue.Empty, queue.Full): return` 中。
- [ ] 第二次 `queue.Empty`/`queue.Full` 只导致当前 block 丢弃并继续 loop；更新已有 metrics，不退出线程。

*重启失败静默丢弃（N-P2-21）*

- [ ] `audio_capture.py:308-321` 的重启分支当前先 `self._restart_event.clear()` 再尝试 `_restart_stream()`，失败时只 log + sleep + continue。改为：失败时**保留或重置事件以便重试**，并设置 `self._metrics.last_error`。
- [ ] 有限次重试后仍失败，必须落到明确终态——按 2.4 的状态表，要么继续旧流、要么停止采集并让主流程感知，不得让采集线程带着失败的流状态继续空转。
- [ ] 与 A5（macOS 侧同类问题）保持恢复语义一致：能恢复则回退设备，不能恢复则明确停止。

*持锁阻塞（N-P3-20）*

- [ ] `self._stream.read(...)`（`audio_capture.py:368-378`）在 `self._lock` 内阻塞执行，与 `_restart_stream()` 争用同一把锁。按 2.6 的同一原则收窄锁范围：锁只保护 `self._stream` 引用的读取，实际 `read()` 在锁外执行。
- [ ] 此处单次阻塞受 chunk 大小约束（32ms 量级），优先级低于 A7；若收窄锁引入新竞态，可保留现状并在注释中记录理由。

- [ ] 不改变采样率、块大小、设备重启触发条件和 mic 混音行为。

**运行检查点**：队列满时记录丢弃计数，后续仍能看到新的 audio block 和 read loop 日志；制造一次设备重启失败，确认有 `last_error`、有重试或明确停止，而不是静默无音频。

**完成标准**：并发消费窗口只丢当前块，不杀死采集线程；重启失败不会退化为「进程还在但永远没有音频」。

### A5｜macOS 设备切换：幂等 + 失败恢复（P1-3、N-P1-2）

**状态**：✅ 已完成


**涉及文件**：`audio_capture_sck.py`、`main.py`；必要时使用已有平台错误文案。

**根因**：`set_device()` 既缺相等性短路（同名设备也整流重启），失败时又不恢复旧流；调用方 `main.py` 用 `key in settings` 判断且忽略返回值。两个缺陷叠加在同一函数上，必须一次改完。

**实施步骤**

*幂等（N-P1-2）*

- [x] `SCKAudioCapture.set_device()` 开头补相等性短路：`if self._device_name == device_name: return True`，与同文件 `set_mic_device()` 的既有写法保持一致。
- [x] `main.py::_on_settings_changed` 改为比较值再调用：仅当 `old_device != settings["audio_device"]` 时才调 `set_device`，遵循 2.8 契约。
- [x] 同一函数内其余分支（ASR 引擎、翻译器、VAD）一并核对是否存在同类「键存在即执行」的判据；`_switch_asr_engine` 已有 signature 短路，确认无需改动即可，不要重复加判。

*失败恢复（P1-3）*

- [x] `set_device()` 保存旧设备和运行状态；停止旧 stream 后尝试新 stream。
- [x] 新 stream 失败时恢复旧设备名并尝试启动旧 stream；恢复成功返回 `False`，保留错误但继续旧设备采集。
- [x] 旧 stream 也失败时设置 stopped 和 `last_error`，返回 `False`，不得报告成功。
- [x] `main.py` 检查返回值；成功才 flush VAD，失败时恢复设置/控件并提示；backend 已停止则统一 stop。
- [x] 不新增 BlackHole/CATap 回退，不改变 SCK 内容重建模型。

**运行检查点**：macOS 上连续改动多项无关设置，日志中不再出现 SCK 停流/重建；切换失败时 UI 显示旧设备或 stopped 状态，不显示假运行。

**完成标准**：同名设备不触发任何重启；真实切换失败时旧流能恢复则只回退设备，不能恢复则明确停止。

### A6｜VAD 跨线程访问统一加锁（N-P1-1）

**状态**：✅ 已完成


**涉及文件**：`main.py`、`vad_processor.py`。

**根因**：`main.py` 中 7 处 VAD 访问持 `_vad_lock`，但 `_on_settings_changed`（384、415-416）与 `_switch_asr_engine`（811-812）这 3 处没有。`_reset()` 分两条语句重新绑定 `_speech_buffer` 与 `_confidence_history`，采集线程可能在中间向旧列表追加而向新列表追加，破坏两者的长度不变式。

**实施步骤**

- [x] `main.py:384`（`update_settings`）、`main.py:415-416`、`main.py:811-812` 三处补 `with self._vad_lock:`。
- [x] 删除 `flush()` 之后冗余的 `_reset()` 调用——`flush()`/`_flush_segment()` 内部已经 reset。若原意是「丢弃缓冲而不产出片段」，改为只调用 `_reset()`，并在注释中写明意图。
- [x] `VADProcessor._reset()` 改为原子替换全部状态，使 `len(_speech_buffer) == len(_confidence_history)` 在任何可观察时刻都成立。
- [x] 按 2.7 契约在 `VADProcessor` 类 docstring 写明「非线程安全，调用方必须持 `_vad_lock`」。
- [x] 复核 `_capture_loop` 中 `_is_speaking` / `_speech_samples` 的无锁读（1986、2010、2018-2019）：这些只作节流判据，保持无锁但**不得**用于切分决策；如已用于决策则一并纳入锁内。
- [x] 不调整 VAD 阈值、分层静音、backtrack 逻辑（见 1.2 不变项）。

**运行检查点**：说话过程中反复改动设置和切换 ASR 引擎，日志无异常，字幕不出现截断或错位；针对 `_reset()` 的不变式补一条断言或单测。

**完成标准**：VAD 的所有状态变更都在同一把锁下发生，两个列表的长度不变式不会被并发破坏。

### A7｜ASRClient 锁分离与状态保真（N-P1-4、N-P2-5）

**状态**：✅ 已完成


**涉及文件**：`asr_client.py`；调用侧核对 `main.py`。

**根因**：`_request`（最长 `request_timeout=120s`）与 `shutdown()`/`terminate()` 共用同一把 `RLock`，使 `_run_asr` 刻意释放 `_asr_lock` 的设计被完全抵消，Qt 线程仍会被阻塞。另外 `terminate()` 仅在进程仍存活时更新 `_status`，进程已自行退出时 `status` 会持续返回 `"ready"`。

**实施步骤**

*锁分离（N-P1-4）*

- [ ] 按 2.6 契约把单锁拆为 IO 锁与生命周期锁；生命周期锁的持有时间必须是常数级。
- [ ] 增加取消标志。`shutdown()`/`terminate()` 置标志后**不等待 IO 锁**，直接中止子进程以打断阻塞中的 `conn.poll/recv`，再在生命周期锁下关闭句柄。
- [ ] `_recv_response` 的轮询循环每轮检查取消标志，被取消时抛 `ASRWorkerExited` 并携带「已取消」原因。
- [ ] `wait_ready()` 同样受取消标志约束，加载中的 worker 必须能被立即中止（当前最长 180s）。
- [ ] 保持 Pipe 消息类型和字段不变（见 1.2）；`asr_worker.py` 不改。

*状态保真（N-P2-5）*

- [ ] `terminate()` **无条件**设置 `self._status = "failed"`，不再以 `is_alive()` 为前提。
- [ ] `status` 属性在 `_process is None` 且从未成功停止时不得返回 `"ready"`；`_close_handles()` 后的状态必须是终态之一（`stopped`/`failed`/`exited`）。
- [ ] 复核 `_switch_asr_engine`（`main.py:794`）依赖 `current_asr.status == "ready"` 的复用判断，确认死 client 不会被误判为可用。
- [ ] 复核 `_recv_response` 的响应 ID 失配分支：`ASRClientError` 不在 `_run_asr` 的恢复分支内，应改为归入可恢复错误或明确标记为致命并走 `_mark_asr_unavailable`，不能留下既不恢复也不报错的状态。

**运行检查点**：worker 处于长耗时 transcribe 时点击退出和切换引擎，UI 在 1~2 秒内响应；模型加载中途切换引擎能立即中止；杀死 worker 进程后 `status` 不再显示 `ready`。

**完成标准**：退出与切换的等待时间与 `request_timeout`/`ready_timeout` 解耦；`status` 永远不描述一个不存在的进程。

### A8｜MLX 服务生命周期与状态查询（N-P1-7、N-P1-8、N-P1-13、N-P2-6、N-P2-7、N-P2-8、N-P2-9、N-P2-19、N-P3-9）

**状态**：✅ 已完成


**涉及文件**：`mlx_service.py`、`control_panel.py`、`main.py`。

**根因**：MLX 相关的 9 项问题共享两个根因——**状态查询同步执行在 Qt 线程**，以及**取消标志的检查点依赖外部进程是否恰好产生输出**。分开修会反复触碰同一批函数，故合并。

**实施步骤**

*状态查询线程化（N-P1-13、N-P1-8）*

- [ ] `_update_mlx_controls`（`control_panel.py:1455`）不再直接调用 `is_model_ready()`/`is_environment_ready()`/`is_running()`。这三者分别派生 Python 子进程、`ps` 子进程和 1.5 秒 `urlopen`，而它绑定在 `currentRowChanged`（`control_panel.py:695`）上。
- [ ] 改为读取由 `_MLXHealthThread` 回填的缓存状态；缓存为空时显示「检查中」而不是阻塞等待。复用 `request_mlx_health_check()`（`control_panel.py:1582`）已有的去重与线程模式，符合 2.9。
- [ ] `_versions_are_compatible()` 的结果按 `.mlx-venv` 的 mtime 缓存，避免每次查询都派生解释器。
- [ ] `_on_mlx_probe_result`（`main.py:358`）中的 `_disable_translator()` 改为**仅在状态由可用变为不可用的边沿**执行一次，而非每次 5 秒探测都执行——当前实现每 5 秒自增一次 `_translator_generation` 并清空历史。
- [ ] 为 MLX 自动重启引入与 ASR 侧 `_asr_restart_count` 一致的**退避与次数上限**；达到上限后停止重试并在 UI 上明示，等待用户手动操作。

*取消语义（N-P2-7、N-P2-6、N-P2-19）*

- [ ] `_check_cancel` 不能只在 `for line in process.stdout` 循环体内被调用（`mlx_service.py:230-236`）。改为在等待子进程期间以固定周期检查（例如用带超时的轮询读取，或另起一个监视线程），使静默下载期间取消同样生效。
- [ ] 取消时必须终止子进程树，而不仅仅是抛出异常。
- [ ] `ensure_running` 的等待循环中 `_check_cancel` 抛出前必须先 `self.stop()`（`mlx_service.py:477-478`），与超时分支（496 行）行为一致；否则会留下仍在运行的服务进程和已写入的 pid 文件。
- [ ] 修好上述取消后，验证 `ControlPanel.closeEvent`（`control_panel.py:1562`）不再出现「面板被禁用且无法关闭」——该现象是取消无效的后果，不需要单独修改 closeEvent 逻辑。若仍可能长时间不响应，为其加一个兜底超时。

*进程与目录安全（N-P1-7、N-P2-9）*

- [ ] `prepare_model` 入口先检查 `is_running()`；服务运行中则先 `stop()` 或拒绝执行并提示用户。当前 `shutil.rmtree(self.model_dir, ignore_errors=True)` + `os.replace`（`mlx_service.py:343-345`）会删除正在被 `mlx_lm.server --model` 持有的目录。这与 A3 中「删除前先停进程」是同一条原则。
- [ ] `ignore_errors=True` 导致失败被完全吞掉，改为捕获并上报。
- [ ] `stop()`（`mlx_service.py:499`）中的 `os.killpg` / `signal.SIGKILL` 在 Windows 上不存在，而 `except` 只捕获 `ProcessLookupError`/`OSError`，`AttributeError` 会逃逸。该方法挂在 `app.aboutToQuit` 上且不分平台。加平台判断或把 `AttributeError` 纳入捕获。

*异常保真与环境隔离（N-P2-8、N-P3-9）*

- [ ] `_run_logged` 的异常路径中 `process.wait(timeout=10)`（`mlx_service.py:238-243`）自身会抛 `TimeoutExpired`，覆盖原始异常（包括用户取消）。用 try/except 包住并在超时后 `kill()`。
- [ ] `prepare_model` 用 `sys.executable -m pip install modelscope`（`mlx_service.py:296-302`）把依赖装进**应用自身正在运行的 venv**。改为装进 `.mlx-venv`，或把 `modelscope` 提升为声明式依赖（与 C7 一并处理）；如确需保留现状，在代码注释与用户提示中写明会修改主环境。

**运行检查点**：选中 HY-MT 模型时点击模型列表，UI 立即响应；HY-MT 环境未就绪时观察日志，5 秒周期内不再出现重复的 `_disable_translator` 与子进程派生；模型下载静默期间点击取消，任务在数秒内终止且无残留进程；服务运行时点击「准备本地模型」被拒绝或先自动停服。

**完成标准**：MLX 的所有状态查询都在后台线程完成；取消在任何阶段都有效且不留残留进程；不存在对运行中模型目录的删除。

### A9｜下载对话框：线程化、可取消与全局状态恢复（N-P1-9、N-P2-10、N-P3-10、N-P3-11、N-P3-12）

**状态**：✅ 已完成


**涉及文件**：`dialogs.py`。

**根因**：`SetupWizardDialog` 与 `ModelDownloadDialog` 使用「裸线程 + 200ms 轮询 `is_alive()`」模式（N-P3-10），而同文件的 `_ConnectionTestThread` 和 `control_panel.py` 的 `_MLXTaskThread` 都用 `QThread` + 信号。轮询模式正是无法取消、也无法在关闭时恢复全局状态的结构性原因。

**实施步骤**

- [ ] 把两个下载对话框改为 `QThread` 子类 + `progress`/`failed`/`succeeded`/`finished` 信号，与 `_MLXTaskThread` 保持同一形态（2.9）。删除 `_poll_timer` 轮询。
- [ ] 引入 `cancel_event`，在下载循环的模型之间以及可行的进度回调处检查；提供取消按钮，并允许通过窗口关闭触发取消。
- [ ] 按 2.10 处理全局状态：`sys.stderr` 与 root logger handler 的安装/恢复放进 `try/finally` 或上下文管理器，并在 `closeEvent`/`reject` 路径同样恢复。当前恢复只写在 `_check_done` 的成功分支上（`dialogs.py:352`、`487`），下载中退出会让 `sys.stderr` 永久指向已销毁 QDialog 的信号。
- [ ] `_StderrCapture`（`dialogs.py:98`）补齐 `fileno()`、`encoding`、`errors`，避免探测这些属性的第三方代码在下载路径上抛 `AttributeError`。
- [ ] 回调改为不直接持有 Qt 对象的弱引用形式，或在恢复时主动断开，确保对象销毁后不会再 emit。
- [ ] `SetupWizardDialog._check_done` 写入的硬编码设置字典（`dialogs.py:370-383`）中 `vad_threshold` 为 `0.3`，与 `config.yaml:32` 的 `0.5` 不一致。二者取其一并统一；更稳妥的做法是让向导复用 `config.yaml` 的默认值而不是另抄一份。
- [ ] 不改变下载的模型集合、hub 选择和代理处理逻辑。

**运行检查点**：下载进行中点击取消，任务在数秒内结束且 `sys.stderr` 已恢复；下载进行中退出应用，控制台能正常打印后续日志与异常栈；向导完成后写出的 `vad_threshold` 与 `config.yaml` 一致。

**完成标准**：下载可取消；任何退出路径都不残留被替换的解释器全局状态；三处后台任务使用同一种线程模式。

### A10｜采集后端错误契约统一（N-P2-15、N-P3-14）

**状态**：✅ 已完成


**涉及文件**：`audio_capture_pyaudio.py`；对照 `audio_capture_sck.py`、`audio_capture_base.py`。

**根因**：两个 macOS 后端对同一个终止条件的处理方式相反。`SCKAudioCapture.get_audio`（`audio_capture_sck.py:519-525`）会抛 `CaptureRuntimeError`，`_capture_loop`（`main.py:1976-1984`）据此停止管线；而 `PyAudioCapture.get_audio`（`audio_capture_pyaudio.py:208-212`）用 `except Exception: return None` 吞掉一切，同样的终止条件在这个后端上永远静默。

**实施步骤**

- [ ] `PyAudioCapture.get_audio` 只捕获 `queue.Empty` 并返回 `None`（与 `audio_capture_base.py:189` 一致），让 `CaptureRuntimeError` 等终止性异常向上传播。
- [ ] 若 PyAudio 后端存在需要区分的可恢复错误，定义为独立异常类型并在 `platform_permissions.py` 的既有异常层次中归位，不要与「无数据」共用 `None`。
- [ ] `__del__` 调用 `stop()`（`audio_capture_pyaudio.py:210-214`，含 3 秒 `join` 与 PyAudio `terminate()`）改为显式生命周期管理；如需保留 `__del__` 作为兜底，其中不得执行阻塞等待。
- [ ] 核对三个后端（Windows `AudioCapture`、`SCKAudioCapture`、`PyAudioCapture`）对 `get_audio` 返回值与异常的约定完全一致，并写入 `audio_capture_base.py` 的类 docstring。

**运行检查点**：在 PyAudio 后端上制造一次终止性采集失败，确认 `_capture_loop` 停止管线并给出错误，而不是静默无音频。

**完成标准**：三个采集后端对「无数据」与「终止性失败」的表达方式一致。

## 5. 阶段 B：翻译、启动和字幕一致性

### B1｜翻译结果收口统一（P2-1、N-P2-2、N-P3-2）

**状态**：✅ 已完成


**涉及文件**：`translator.py`。

**根因**：`translate()` 与 `translate_iter()` 的非流式分支各自实现收尾，后者跳过了 `_check_repetition` 和 `_warn_if_thinking_burned`；流式请求的 `stream_options` 回退用了宽泛的 `except Exception`；重复检测只覆盖从位置 0 开始的复读。三者都在同一组函数内，一并处理。

**实施步骤**

*收口统一（P2-1）*

- [ ] 提取内部最终结果收口方法，统一执行 `_warn_if_thinking_burned`、`_check_repetition`、`_append_history`。
- [ ] `translate()` 和 `translate_iter()` 非流式分支调用同一收口方法；流式循环结束只收口一次。
- [ ] JSON response 先提取 `t`，再执行重复检测和 history 提交；不改变现有 `RepetitionError` 由 `main.py` 处理的方式。

*流式回退收窄（N-P2-2）*

- [ ] `stream_options` 回退只捕获参数不兼容类异常（`BadRequestError`、`TypeError`、`UnprocessableEntityError`），连接/超时/鉴权错误直接向上抛。
- [ ] 把 `_translate_streaming` 与 `translate_iter` 中重复的流式循环收敛为一个私有生成器，两条路径共用，消除 25 行重复代码。
- [ ] 回退发生时记录一条 debug 日志，说明服务端不支持 `stream_options`。

*重复检测增强（N-P3-2）*

- [ ] `_check_repetition` 扩展为可检测从任意位置开始的复读（如对文本尾部窗口做周期性检测），保持 8 字符最小模式长度。
- [ ] 保持时间复杂度可接受：只对超过阈值长度的输出执行，避免在实时路径上引入明显开销。
- [ ] 不改变 `RepetitionError` 的抛出位置和调用方处理方式。

**运行检查点**：非流式重复结果进入现有 Overlay 错误提示；正常翻译 history 不重复增长；服务不可达时失败在一个 timeout 内返回而非两个。

**完成标准**：streaming 开关不再改变重复检测语义；网络错误不再被重试掩盖；尾部复读能被检出。

### B2｜Benchmark 与运行时请求等价，并消除其崩溃路径（P2-3、N-P2-11、N-P2-12、N-P2-13）

**状态**：✅ 已完成


**涉及文件**：`benchmark.py`；必要时 `translator.py`。

- [ ] 优先直接复用 `Translator._build_request_kwargs()`，不复制 thinking、JSON 和 overrides 推断规则。
- [ ] 传入模型的 `overrides`、`extra_body`、`thinking_style`、`no_system_role`、`json_response`、`context_turns`、proxy、timeout 和 streaming 设置。
- [ ] Benchmark 只计时和展示，不写运行时 history，不修改全局 Translator。
- [ ] 流式失败 fallback 只切换 `stream`，其余 kwargs 保持一致。
- [ ] 如果确实存在重复的模型配置组装，只提取一个窄 helper，不做 Translator/Benchmark 大重构。

*崩溃路径（N-P2-11、N-P2-12、N-P2-13）*

以下三处都是 `translator.py` 已经处理好、但 Benchmark 没有同步的防护。若上面的「复用 Translator 请求路径」彻底落实，三者会一并消失；若选择保留独立的请求循环，则必须逐条补齐：

- [ ] `benchmark.py:100` 的 `delta = chunk.choices[0].delta` 补 `if chunk.choices:` 判空，与 `translator.py:428` 一致。provider 发送 usage-only（空 choices）分片时当前会抛 `IndexError`，静默退化到非流式路径并污染延迟数据。
- [ ] `benchmark.py:118` 的 `resp.choices[0].message.content.strip()` 补 `or ""`，与 `translator.py:478` 一致。thinking 模型烧光预算时返回 `content=None`，当前会抛 `AttributeError`——正是 issue #38 的场景，而 Benchmark 恰恰是用户用来诊断该问题的工具。
- [ ] `benchmark.py:105` 流式失败回退的 `except Exception:` 收窄为参数不兼容类异常，与 B1 对 `translator.py` 的处理保持同一标准。当前连接错误会导致每个句子付出 2× timeout，`rounds` 轮下来等待成倍放大。

**运行检查点**：打印或调试查看请求时，Benchmark 和实时调用的角色、extra body、overrides、JSON 参数一致；对一个 thinking 未关闭的模型跑 Benchmark，得到的是有意义的诊断输出而不是 `AttributeError`；对不可达端点跑 Benchmark，总耗时约等于 `模型数 × 轮数 × timeout` 而非其两倍。

**完成标准**：Benchmark 与实时翻译的关键请求契约一致，且不存在实时路径已修复而 Benchmark 仍会崩溃的分支。

### B3｜延迟启动与暂停竞态（P2-6）

**状态**：✅ 已完成


**涉及文件**：`main.py`。

- [ ] 延迟启动期间 `_is_running=False`，Overlay/tray 显示未启动或暂停。
- [ ] 增加最小取消标志；`on_pause()`/`on_quit()` 设置取消，`on_start()` 入口先检查。
- [ ] 只有 `live_trans.start()` 成功后才设置 `_is_running=True` 并同步 UI。
- [ ] 不引入完整生命周期状态机。

**运行检查点**：启动后 500ms 内点击暂停或退出，延迟回调不再调用 start。

**完成标准**：延迟回调不能覆盖用户最后一次暂停/退出操作。

### B4｜字幕最短显示时间 FIFO（P2-5）

**状态**：✅ 已完成


**涉及文件**：`subtitle_window.py`。

- [ ] 将单 pending timer 改为最小 FIFO，元素保持 `(original, translations)`。
- [ ] 只保留一个队首 timer；timer 触发时先移除队首，再插入并安排下一项。
- [ ] 新消息不再无条件取消 pending；clear、新会话和销毁时才清空队列和 timer。
- [ ] 保持最大句数、自动隐藏和刷新逻辑。

**运行检查点**：短时间连续收到多条最终字幕时，窗口按到达顺序显示所有句子；clear 后旧 timer 不再写入。

**完成标准**：字幕窗口不丢中间句。

### B5｜翻译不可用的可见失败（N-P1-5）

**状态**：✅ 已完成


**涉及文件**：`main.py`；必要时 `i18n/*.yaml`。

**根因**：`_snapshot_translation_request` 在 `_translator is None` 时抛 `RuntimeError`，与 executor 关闭抛出的 `RuntimeError` 无法区分，被同一个 `except RuntimeError` 吞掉。此前已写入的 overlay 消息和 `TranscriptWriter._pending` 条目因此永不落定。

**实施步骤**

- [x] 定义独立异常（如 `TranslationUnavailable`），与 executor 关闭的 `RuntimeError` 区分开。
- [x] `_process_segment` 和 `_process_segment_text` 两处捕获点分别处理：
  - 翻译服务不可用 → `self._transcript.finalize_no_translation(msg_id)` + Overlay 显示明确错误文案（i18n 键）。
  - Executor 已关闭（退出中）→ 保持现有静默跳过，但同样调用 `finalize_no_translation` 以免 `_pending` 泄漏。
- [x] 两处捕获逻辑抽为一个私有方法，避免 `_process_segment` 与 `_process_segment_text` 再次分叉（这两个函数本就有大量重复，见 C 阶段备注）。
- [x] 修正日志文案：不可用时不再输出 "Translation executor shut down"。
- [x] 复核 `TranscriptWriter._pending`：确认所有写入 `write_original()` 的路径最终都有 `write_translation()` 或 `finalize_no_translation()` 与之配对。
- [ ] 复核 `_translate_extra_langs` 中 `self._translator.fork_for_request(...)` 的 `None` 解引用，走同一异常路径。

**运行检查点**：选中 HY-MT 但不启动 MLX 服务，说话后 Overlay 每条消息都落定为明确错误；`all` transcript 中该条目存在；长时间运行后 `_pending` 不增长。

**完成标准**：翻译服务不可用是一次可见、可落定、可解释的失败，不产生悬挂消息或内存泄漏。

### B6｜增量 ASR 短句缓冲去重（N-P1-3、N-P3-6）

**状态**：✅ 已完成


**涉及文件**：`main.py`；关联 `vad_processor.py` 的密度丢弃路径。

**根因**：`_do_interim_asr` 中所有句子都是短句时 `actually_committed` 为 `False`，函数在 trim 与 `_interim_committed_tail` 更新之前返回。缓冲未裁剪 + tail 未更新 → 下一轮对同一段音频重新识别出同样的短句 → `_interim_pending += text` 重复累加。

**实施步骤**

*重复累加（N-P1-3）*

- [ ] 短句被缓冲时同样推进状态：要么执行对应的 trim 并更新 `_interim_committed_tail`，要么在追加前对 `_interim_pending` 做尾部去重，二选一并在注释中写明取舍。
- [ ] 为 `_interim_pending` 设长度上限（建议 200 字符），超限时丢弃最旧部分并记 debug 日志。
- [ ] 复核 `_process_interim_final` 中 `_interim_pending` 被前置后又因空文本/噪声过滤 early-return 的分支，确保缓冲内容不会被静默丢弃且不残留。

*状态泄漏（N-P3-6）*

- [ ] `_flush_segment` 因密度 < 25% 丢弃片段时不产生 `vad_flush` 事件，导致 `_asr_loop` 的 `_interim_active`/`_interim_pending` 清理不执行。为该路径补一条明确的状态复位通道（例如让 VAD 返回一个「已丢弃」信号，或由 `_capture_loop` 在检测到 `_is_speaking` 由真变假且无片段产出时复位）。
- [ ] 不改变密度阈值本身（见 1.2 不变项）。

**运行检查点**：连续说「はい。…（长句）」，字幕中「はい。」只出现一次；密度过低的噪声段之后，下一段正常语音不携带上一段的 `_interim_pending` 残留。

**完成标准**：同一段音频不会被重复提交为文本；interim 状态在所有片段结束路径上都会复位。

### B7｜翻译并发计数与线程数收口（N-P2-1、N-P3-1）

**状态**：✅ 已完成


**涉及文件**：`main.py`。

**实施步骤**

*计数（N-P2-1）*

- [ ] `_commit_translation_result` 的 generation 不匹配分支**不得**触碰新一代的 `_translation_pending`——该计数器已被 `_disable_translator()`/`_on_model_changed()` 归零。
- [ ] 同一分支中 `_translation_order.remove(msg_id)` 与 `_translation_results.pop(msg_id)` 同样只应作用于本代数据；确认 `msg_id` 单调递增不会与新一代冲突后，可直接 `return False`。
- [ ] 确认 `_record_latency` 打印的 `translation_pending` 在模型切换后仍能反映真实在途数。

*线程数（N-P3-1）*

- [ ] `_on_model_changed` 改为调用 `_set_translation_workers(...)`，不再直接给 `self._translation_workers` 赋值，以复用 `max(4, min(16, ...))` 钳制。
- [ ] 确认在 `_running=False` 时该调用只更新目标值、不创建 executor，与现有行为一致。

**运行检查点**：切换模型后触发若干翻译，PERF 日志中的 `translation_pending` 数值合理；把 `user_settings.json` 的 `translation_workers` 改成 100，启动后线程池上限仍为 16。

**完成标准**：并发计数不再被过期回调污染；线程数在所有入口上都受同一钳制。

### B8｜字幕窗口渲染与重建（N-P2-17、N-P3-16、N-P3-17）

**状态**：✅ 已完成


**涉及文件**：`subtitle_window.py`。

**实施步骤**

*换行性能（N-P2-17）*

- [ ] `_SubtitleTextWidget.split_text`（`subtitle_window.py:359-368`）当前对每个字符位置测量整个前缀：`for i in range(1, len(text)+1): fm.horizontalAdvance(text[:i])`。由于 `horizontalAdvance` 本身与前缀长度成正比，单段换行即为 O(n²) 字形测量。
- [ ] 改为二分查找定位断点，或直接改用 `QTextLayout` 做换行。
- [ ] 该函数经 `_rewrap()` 在每条字幕更新和每次 `resizeEvent` 时于 Qt 主线程执行，属于实时路径。
- [ ] 保持现有断点偏好规则（在 ` ,，。、!！?？;；:：.` 处优先断行）与返回值形态不变。

*重复实现（N-P3-16）*

- [ ] `apply_settings`（`subtitle_window.py:670-683`）内联复制了 `_rebuild_text_widgets`（585-598）的 12 行——移除旧 widget、`deleteLater`、按 `lines` 重建、连 `height_changed`、加入布局。`_rebuild_text_widgets` 在 577 行确有调用，因此这是并存的两份实现而非死代码。
- [ ] 让 `apply_settings` 直接调用 `_rebuild_text_widgets()`，删除内联副本。

*空值分支（N-P3-17）*

- [ ] `_refresh_display`（`subtitle_window.py:879-891`）取译文的四个分支里，只有 `elif lang and lang in tl_dict: texts.append(tl_dict[lang])` 不检查值是否为空，其余三个都有 `if v` / `and tl_dict[""]` 判断。补齐该分支，避免空译文进入 `" | ".join(texts)` 产生前导分隔符。

**运行检查点**：长句字幕（>100 字符）更新与窗口缩放时无可感卡顿；修改字幕行配置后 widget 正确重建；某个目标语言译文为空时不出现 `" | 正文"` 这样的前导分隔符。

**完成标准**：换行不再是实时路径上的 O(n²) 操作；widget 重建只有一份实现；四个取值分支的空值处理一致。

### B9｜字幕设置的状态隔离（N-P2-20、N-P3-19）

**状态**：✅ 已完成


**涉及文件**：`subtitle_settings.py`。

**实施步骤**

- [ ] `_emit_settings`（`subtitle_settings.py:628-629`）当前发射内部字典本体 `self.settings_changed.emit(self._settings)`，且 `s["lines"]` 与 `self._settings["lines"]` 是同一个 list 对象。改为发射深拷贝（至少 `lines` 需要独立复制），与 `ControlPanel._apply_settings`（`control_panel.py:1864`）发 `dict(self._current_settings)` 的做法对齐。
- [ ] 接收方（`main.py` → `subwin.apply_settings()`）不得与设置控件共享可变状态；核对 `_merge_settings` 是否也需要复制。
- [ ] `get_settings()`（`subtitle_settings.py:631-633`）内部调用 `_emit_settings()`，使这个 getter 会发射信号并触发一次完整的设置传播（在 `main.py` 中会走到 widget 重建）。拆分为纯读取的 `get_settings()` 与显式的 `emit_settings()`，调用方按需选择。
- [ ] 核对拆分后 `SubtitleSettingsDialog.get_settings()`（`subtitle_settings.py:653`）的调用方仍能拿到最新控件值。

**运行检查点**：修改字幕设置后，控件内部状态与已发出的设置对象互不影响；调用 `get_settings()` 不再触发字幕窗口重建。

**完成标准**：设置对象跨模块传递时不共享可变状态；读取操作没有副作用。

## 6. 阶段 C：配置、服务边界与体验

### C1｜active_model 删除和运行时切换（P2-2、N-P3-7）

**状态**：✅ 已完成


**涉及文件**：`control_panel.py`，联动检查 `main.py` 的 `model_changed` 接收路径。

*索引维护（P2-2）*

- [ ] 删除前保存 `old_active` 和旧活动模型。
- [ ] 删除 active 选择相邻项；删除 active 前方项则 `active -= 1`；删除后方项不变；最终索引 clamp 合法。
- [ ] 活动模型对象/索引变化时发 `model_changed(new_model)`；列表变化仍发 `models_list_changed`。
- [ ] 保持列表刷新、当前行、持久化和 Overlay 更新顺序，确保运行时 Translator 切换到新配置。

*列表引用（N-P3-7）*

- [ ] `_dup_model` / `_remove_model` 不再依赖 `self._current_settings.get("models", [])` 的临时默认空列表；键缺失时先写回一个真实列表再操作，确保 `append`/`pop` 作用于持久化对象。
- [ ] 复核同文件其他 `get(..., [])` / `get(..., {})` 后立即修改返回值的位置，采用同一写法。

**运行检查点**：删除模型后同时查看设置文件、控制面板当前行、Overlay 菜单和运行日志，四者必须指向同一个模型；在无 `models` 键的空配置上执行复制/删除，设置文件内容与 UI 一致。

**完成标准**：active 索引不会漂移，运行中的 Translator 不再持有已删除模型，列表修改总能写回。

### C2｜远程 ASR 服务：配置入口与暴露面（P2-4、N-P2-14、N-P3-13）

**状态**：✅ 已完成


**涉及文件**：`asr_server.py`，必要时 `REMOTE_ASR.md`。

*配置入口（P2-4）*

- [ ] 集中默认 host/port/model/device/compute_type；`__main__` 只解析参数并覆盖默认值。
- [ ] 模块导入时提供默认 `app.state.args`，或提供明确的 `create_app(args)` 注入；startup 和 health 使用同一配置源。
- [ ] 不改变 `/transcribe` 二进制协议、GPU lock 和错误响应。
- [ ] 文档记录 `python asr_server.py` 与 `uvicorn asr_server:app` 的实际参数来源和启动方式。

*暴露面收敛（N-P2-14）*

- [ ] 默认 `--host` 由 `0.0.0.0`（`asr_server.py:129`）改为 `127.0.0.1`。需要跨机访问时由用户显式指定绑定地址——这是本地优先的安全默认值，不是功能削减。
- [ ] 给 `/transcribe` 加请求体大小上限（按最长可接受音频时长换算，16 kHz float32 下 5 分钟约 19 MB），超限返回 413，而不是先 `await request.body()` 全量读入内存再处理。
- [ ] 若确需保留跨机使用场景，增加一个可选的共享密钥 header 校验；不要求实现完整鉴权体系。
- [ ] 在 `REMOTE_ASR.md` 中写明：该服务无内建鉴权，仅应部署在受信网络，并说明新的默认绑定地址。

*废弃 API（N-P3-13）*

- [ ] `@app.on_event("startup")`（`asr_server.py:42`）自 FastAPI 0.93 起废弃，迁移到 `lifespan`。该迁移与 `create_app(args)` 的重构天然契合，一并完成。

**运行检查点**：分别用两种入口启动，startup 能加载配置且 `/health` 返回模型名和 `status=ok`；从另一台机器访问默认启动的服务应被拒绝；发送超限 body 得到 413 而非 OOM。

**完成标准**：ASGI 导入不再因缺少 `app.state.args` 失败；默认配置不对外网暴露 GPU 推理；无废弃 API 警告。

### C3｜FunASR-Nano 原始错误保真（P3-1）

**状态**：✅ 已完成


**涉及文件**：`funasr_nano/model.py`。

- [ ] `load_audio_text_image_video()` 失败立即抛出带输入引用的异常，并使用 `raise ... from e`。
- [ ] 失败路径不得继续调用依赖 `data_src` 的 `extract_fbank()`。
- [ ] 保持 `asr_worker.py` 错误包装字段和 recoverable 语义，正常推理张量流程不变。
- [ ] 顺带修正同函数的可变默认参数（`data_load_speech()` 的 `list`/`dict` 默认值），改为 `None` 哨兵。

**运行检查点**：错误日志显示音频引用和原始解码原因，不再出现 `UnboundLocalError`。

**完成标准**：上层看到真实音频加载/解码失败原因。

### C4｜ASR worker 请求转发契约（N-P3-3）

**状态**：✅ 已完成


**涉及文件**：`asr_worker.py`。

- [ ] `_transcribe` 明确定义会转发哪些 payload 键；对无法转发的键要么显式忽略并记 debug 日志，要么按契约报错，不再静默丢弃。
- [ ] 把 `inspect.signature(engine.transcribe)` 的结果按 engine 实例缓存，避免每次实时调用都做签名反射。
- [ ] 保持 Pipe 消息字段与 `recoverable` 语义不变（见 1.2）。

**运行检查点**：日志中能看到被忽略的 kwargs（若有）；连续调用时不再出现重复的签名解析开销。

**完成标准**：客户端传入的参数要么被转发，要么被明确记录，不存在无声丢弃。

### C5｜用户可见文案、i18n 健壮性与导出完整性（N-P2-4、N-P2-16、N-P3-4、N-P3-8、N-P3-15、N-P3-18）

**状态**：✅ 已完成


**涉及文件**：`main.py`、`mlx_service.py`、`control_panel.py`、`i18n.py`、`i18n/zh.yaml`、`i18n/en.yaml`、`subtitle_overlay.py`。

*硬编码文案（N-P2-4、N-P3-8、N-P3-18）*

三处绕过 i18n 的文案，一并处理：

- [ ] `main.py:589-593` 的 HY-MT 硬编码中文提示改为 `t()` 调用，新增对应 i18n 键（如 `error_mlx_not_running`），中英文各补一条。
- [ ] `mlx_service.py` 全模块的用户可见文案改为通过回调传出 i18n 键或已本地化文本，包括 `ensure_running` 的两处中英混排错误信息、等待提示 `"正在等待 HY-MT 服务加载模型..."`，以及 `prepare_model` 的全部进度文案。该模块目前不依赖 `i18n`，可让调用方 `control_panel` 负责本地化，避免给它新增 UI 依赖。
- [ ] `control_panel.py:1742-1757` 的 `_on_ui_lang_changed` 提示框硬编码中英双语字面量，而该函数刚刚调用过 `set_lang(lang)`——`t()` 本可正常工作。同时它直接 `_save_settings`，绕过 `_apply_settings()`，改为与其余设置项一致走 `_auto_save()`。
- [ ] 全局搜索其余用户可见的硬编码中英文字符串（`QMessageBox`、`setText`、`showMessage` 的字面量参数），一并纳入或明确记录为后续项。

*i18n 健壮性（N-P2-16、N-P3-15）*

- [x] `i18n.set_lang()`（`i18n.py:21-27`）的 `yaml.safe_load(f.read_text())` 加异常保护。该函数在**模块导入期**执行（`i18n.py:38`），YAML 损坏会在任何 UI 出现之前终止进程且无提示；失败时应退回内置英文或空字典并记录错误。
- [x] `locale.getdefaultlocale()`（`i18n.py:13`）自 Python 3.11 废弃、计划 3.15 移除，改用 `locale.getlocale()` 或环境变量探测。
- [x] `t()`（`i18n.py:34`）在 key 缺失时静默返回 key 本身；改为同时记录一条 debug 日志，使拼写错误在开发期可见，UI 上仍返回 key 以免抛异常。
- [x] 补一个测试：`i18n/zh.yaml` 与 `i18n/en.yaml` 的键集合一致。

*导出（N-P3-4）*

- [ ] `export_messages` 在 `_messages` 已因 `_max_messages=50` 轮转而不完整时，于导出结果和 UI 上明确提示只包含最近 N 条，并指向对应的 transcript 文件路径（`TranscriptWriter.session_paths()` 已提供）。
- [ ] 不提高 `_max_messages`（内存与渲染成本是有意的），不把 transcript 读取逻辑塞进 Overlay。

**运行检查点**：英文界面下触发 HY-MT 未启动提示，显示英文；产生 60 条以上消息后导出，UI 提示截断并给出 transcript 路径。

**完成标准**：不存在绕过 i18n 的用户可见文案；导出的不完整性对用户可见。

### C6｜控制面板后台任务去重（N-P2-18）

**状态**：✅ 已完成


**涉及文件**：`control_panel.py`。

- [ ] `_refresh_cache`（`control_panel.py:1241-1253`）当前无条件 `threading.Thread(target=_scan, daemon=True).start()`，而 `_on_tab_changed`（1237）每次切到缓存页都调用它。反复切换标签页会并发派生多个遍历多 GB 模型目录的扫描线程。
- [ ] 按 2.9 补去重保护，复用同文件 `request_mlx_health_check`（1583）的 `isRunning()` 模式；建议一并改为 `QThread` 以统一形态。
- [ ] 扫描进行中再次进入缓存页应复用结果或显示「扫描中」，不派生新线程。
- [ ] 不改变缓存条目的计算方式与展示格式。

**运行检查点**：快速反复切换到缓存页，确认同一时刻只有一个扫描线程在运行。

**完成标准**：控制面板的后台任务与项目内已有的线程模式一致，且不会因 UI 操作被无限派生。

### C7｜依赖声明与平台对称性（N-P2-22、N-P2-23、N-P3-21、N-P3-22）

**状态**：✅ 已完成


**涉及文件**：`requirements.txt`、`requirements-mac.txt`、`tests/test_requirements.py`；可能涉及安装脚本注释。

**根因**：requirements 文件既不自足、也不对称，而现有测试只做单向校验（windows ⊆ mac），因此这类问题不会被发现。

**实施步骤**

*自足性（N-P2-22）*

- [x] `yasbd-lib` 被 5 个入口脚本安装（`install.ps1:283`、`install.sh:26`、`update.bat:73`、`update.sh:14`、`build_release.ps1:168`）且被 `tests/test_requirements.py` 逐个断言，但两个 requirements 文件对它只字未提。按 2.11，二选一：
  - **方案 A（推荐）**：把 `yasbd-lib>=0.15,<1.0` 直接写入两个 requirements 文件，脚本中的独立安装步骤保留为幂等冗余或删除。
  - **方案 B**：保留现状，但在两个 requirements 文件中加注释说明它由安装脚本单独安装及原因（参照现有 torch 注释的写法）。
- [x] 无论哪个方案，都要在 `main.py` 的 `_get_segmenter` 处理 `ImportError`：该 import 位于 `_do_interim_asr` 中**没有 try/except 覆盖**的路径上，缺包会直接杀死 ASR 线程。这是 A1 的一个具体触发器，应与 A1 的结果契约一并验证。降级行为：记录一次明确错误并禁用增量 ASR，而不是让线程退出。

*平台对称性（N-P2-23、N-P3-21）*

- [x] `socksio>=1.0.0` 目前只在 `requirements-mac.txt`。模型配置的 `proxy` 字段可填任意 URL，Windows 用户填 `socks5://...` 时 httpx 抛 `ImportError`。补入 `requirements.txt`，或在代理设置 UI 上明确说明 Windows 不支持 SOCKS。
- [x] `transformers>=4.40.0`（Windows，无上界）vs `transformers==4.57.1`（macOS，钉死）。`mlx_service._versions_are_compatible`（`mlx_service.py:186`）断言 `transformers` 主版本 `< 5`，因此 Windows 侧至少应补 `<5` 上界。
- [x] 逐条核对两个文件的其余差异，确认每一处都是有意的平台差异并带注释（`PyAudioWPatch` 仅 Windows、pyobjc 系列仅 macOS 已符合要求）。

*未使用依赖（N-P3-22）*

- [ ] `pyannote-audio>=4.0,<5` 与 `torchcodec>=0.7` 在 `requirements-mac.txt` 中声明，但全代码库无任何 import。确认它们是否为 GigaAM/torchaudio 的必需传递依赖：
  - 是：加注释写明是谁的传递依赖及为何需要显式钉版本。
  - 否：同时删除依赖与 `tests/test_requirements.py:81-82` 中对应的断言。当前测试把未使用的重依赖（`pyannote-audio` 会拉入 lightning 等）固化进了契约。

*测试补强*

- [ ] `tests/test_requirements.py::test_mac_requirements_contain_every_cross_platform_dependency` 增加**反向**校验：mac 独有的依赖必须出现在一个显式的平台专属白名单中，否则判定为不对称。
- [ ] 补一条断言：requirements 中出现的每个包，要么在代码中被 import，要么在文件内有注释说明理由。

**运行检查点**：在一个只执行 `pip install -r requirements.txt` 的干净环境中启动应用并开启增量 ASR，确认要么正常工作、要么给出明确的降级提示，而不是 ASR 线程静默退出。

**完成标准**：requirements 自足或明确记录例外；平台差异全部有意且有注释；测试能捕获双向不对称。

### C 阶段备注：已识别但本轮不处理的重复代码

以下重复在审查中已确认，**不在本轮任务范围内**，记录以避免后续重复发现：

- `_process_segment`（`main.py:1597`）与 `_process_segment_text`（`main.py:1870`）有约 40 行近乎逐行重复的逻辑（语言过滤、计数、overlay/transcript 写入、extra_langs、同语言分支）。B5 只抽取其中的失败处理部分；完整合并应在存活性问题全部修复后单独进行。
- `subtitle_overlay.py` 与 `subtitle_window.py` 的样式/动画代码存在结构性重复，与本轮问题无关。

## 7. 代码审查和交付核对

以下是修复完成后的实际运行核对：

*基础（D1 是其余全部核对的前置条件）*

- [x] **`python -m pytest -q` 返回 0**。这是第一个要达成、也是每次改动后都要复查的条件；在它变绿之前，下面所有运行检查点都无法证伪。
- [x] `python -m compileall -q .`，确认没有语法错误。
- [ ] 若环境可用，安装并运行 `ruff check --select F,E,W --ignore E501,E402 *.py`；不可用时记录环境限制（本轮实施环境仍缺少该工具，已回退到 compileall + AST 扫描，见 §10）。
- [x] 为 A6 的 VAD 不变式、B6 的短句去重、C5 的 i18n 键一致性各补一条单测。
- [x] 在只装 CI 声明依赖的干净环境中确认 `pytest -q` 无收集失败（用符号链接构造的等价 site-packages，见 §10）。
- [ ] 在只执行 `pip install -r requirements.txt` 的干净环境中启动应用并开启增量 ASR，确认不会静默杀死 ASR 线程。

*存活性与退出*

- [x] 代码审查确认所有退出入口都经过统一 stop，且没有无限阻塞 `put()`。
- [ ] 实际运行中制造一次 ASR 坏结果，确认后续音频仍出字幕。
- [ ] 停止远程 ASR 服务，确认 UI 显示不可用；恢复服务后确认重新产出。
- [ ] 在队列拥塞时观察 Windows 采集线程仍继续运行。
- [ ] worker 处于长耗时 transcribe 时点击退出，确认 UI 在数秒内响应而非等满 `request_timeout`。
- [ ] 模型加载中途切换 ASR 引擎，确认加载能被立即中止。
- [ ] 外部杀死 ASR worker 进程，确认 `status` 不再报告 `ready` 且能触发恢复。
- [ ] 托盘、Overlay、控制面板缓存退出各执行一次，确认没有残留 ASR worker、音频线程或未关闭转录文件。
- [ ] 连按两次 Ctrl-C，确认不出现重复清理日志或异常栈。

*macOS 与 VAD*

- [ ] macOS 上连续改动 10 项无关设置，确认日志中没有 SCK 停流/重建记录，音频不中断。
- [ ] 在 macOS 设备切换失败时确认旧 stream 恢复或 pipeline 明确停止。
- [ ] 说话过程中反复改设置和切换 ASR 引擎，确认字幕无截断、无错位。

*翻译与字幕*

- [ ] 在非流式模式返回重复译文，确认现有重复错误提示生效。
- [ ] 构造尾部复读的译文，确认新的重复检测能识别。
- [ ] 断开翻译服务，确认失败在一个 timeout 内返回（不是两个）。
- [ ] 选中 HY-MT 但不启动 MLX 服务，确认每条消息都落定为明确错误、transcript 完整、内存不增长。
- [ ] 连续说短应答 + 长句，确认字幕中短句只出现一次。
- [ ] 快速连续输入字幕，确认 SubtitleWindow 不丢中间句。
- [ ] 启动后 500ms 内点击暂停，确认不会被延迟回调覆盖。

*MLX 与下载*

- [ ] 选中 HY-MT 模型时点击模型列表，UI 立即响应（当前约 2 秒冻结）。
- [ ] HY-MT 环境未就绪时观察 5 分钟日志，确认没有每 5 秒一次的 `_disable_translator` 与子进程派生。
- [ ] 模型下载静默期间点击取消，任务在数秒内终止，无残留进程，设置面板可正常关闭。
- [ ] MLX 服务运行时点击「准备本地模型」，确认被拒绝或先自动停服。
- [ ] 下载进行中退出应用，确认控制台仍能正常打印后续日志与异常栈（`sys.stderr` 已恢复）。

*配置与服务*

- [ ] 删除 active 前方模型，确认设置、Overlay 和 Translator 同步。
- [ ] 分别使用两种 ASGI 入口启动服务并访问 `/health`；从另一台机器访问默认启动的服务应被拒绝。
- [ ] 触发 FunASR-Nano 音频加载失败，确认保留原始错误。
- [ ] 英文界面下触发 HY-MT 未启动提示与 MLX 进度文案，确认全部显示英文。
- [ ] 产生 60 条以上消息后导出，确认有截断提示。
- [ ] 快速反复切换到缓存页，确认同一时刻只有一个扫描线程。
- [ ] 长句字幕更新与窗口缩放无可感卡顿。
- [ ] 对一个 thinking 未关闭的模型跑 Benchmark，得到诊断输出而非 `AttributeError`。
- [ ] 在 PyAudio 后端上制造终止性采集失败，确认管线停止并报错而非静默无音频。
- [ ] 在一个测试故意失败的分支上触发发布流程，确认 `build` 与 `package-macos-arm64` 都被跳过。

## 8. 文件和任务映射

| 任务 | 覆盖问题 | 主要文件 | 允许联动文件 | 禁止扩展范围 |
|---|---|---|---|---|
| A1 | P1-1 | `main.py` | `asr_client.py` 仅复用类型 | 不改 Pipe/VAD |
| A2 | P1-2 | `asr_remote.py`, `main.py` | `REMOTE_ASR.md` | 不改二进制协议 |
| A3 | P1-5, P1-6, N-P1-6, N-P2-3, N-P3-5 | `main.py`, `control_panel.py` | `i18n/*.yaml` | 不把删除逻辑移入模型管理器 |
| A4 | P1-4, N-P2-21, N-P3-20 | `audio_capture.py` | `audio_capture_base.py` | 不改采样率/块大小/重启触发条件 |
| A5 | P1-3, N-P1-2 | `audio_capture_sck.py`, `main.py` | `i18n/*.yaml` | 不新增音频回退 |
| A6 | N-P1-1 | `main.py`, `vad_processor.py` | `tests/` | 不调 VAD 阈值和切分算法 |
| A7 | N-P1-4, N-P2-5 | `asr_client.py` | `main.py` 仅调用侧核对 | 不改 Pipe 消息语义，不改 `asr_worker.py` |
| B1 | P2-1, N-P2-2, N-P3-2 | `translator.py` | `main.py` | 不改 provider 推断 |
| B2 | P2-3, N-P2-11, N-P2-12, N-P2-13 | `benchmark.py` | `translator.py` | 不维护独立请求规则 |
| B3 | P2-6 | `main.py` | 无 | 不引入完整状态机 |
| B4 | P2-5 | `subtitle_window.py` | 无 | 不改最大句数/自动隐藏 |
| B5 | N-P1-5 | `main.py` | `i18n/*.yaml`, `transcript_writer.py` 仅核对 | 不合并 `_process_segment*`（见 C 阶段备注） |
| B6 | N-P1-3, N-P3-6 | `main.py` | `vad_processor.py` | 不改密度阈值和分句规则 |
| B7 | N-P2-1, N-P3-1 | `main.py` | 无 | 不改 generation 机制本身 |
| A8 | N-P1-7, N-P1-8, N-P1-13, N-P2-6, N-P2-7, N-P2-8, N-P2-9, N-P2-19, N-P3-9 | `mlx_service.py`, `control_panel.py`, `main.py` | `i18n/*.yaml` | 不改 HY-MT 模型转换流程与 `/v1` 协议 |
| A9 | N-P1-9, N-P2-10, N-P3-10, N-P3-11, N-P3-12 | `dialogs.py` | `config.yaml` 仅读默认值 | 不改下载的模型集合与 hub 选择 |
| A10 | N-P2-15, N-P3-14 | `audio_capture_pyaudio.py` | `audio_capture_base.py`, `platform_permissions.py` | 不改 SCK 后端行为 |
| B8 | N-P2-17, N-P3-16, N-P3-17 | `subtitle_window.py` | 无 | 不改断行偏好规则与动画 |
| B9 | N-P2-20, N-P3-19 | `subtitle_settings.py` | `main.py` 仅调用侧核对 | 不改设置字段格式 |
| C1 | P2-2, N-P3-7 | `control_panel.py` | `main.py` | 不改模型字段格式 |
| C2 | P2-4, N-P2-14, N-P3-13 | `asr_server.py` | `REMOTE_ASR.md` | 不改 `/transcribe` 二进制协议与 GPU lock |
| C3 | P3-1 | `funasr_nano/model.py` | 无 | 不改正常推理流程 |
| C4 | N-P3-3 | `asr_worker.py` | 无 | 不改 Pipe 字段和 recoverable 语义 |
| C5 | N-P2-4, N-P2-16, N-P3-4, N-P3-8, N-P3-15, N-P3-18 | `main.py`, `mlx_service.py`, `control_panel.py`, `i18n.py`, `i18n/*.yaml`, `subtitle_overlay.py` | `tests/` | 不提高 `_max_messages`；不给 `mlx_service` 新增 UI 依赖 |
| C6 | N-P2-18 | `control_panel.py` | 无 | 不改缓存条目计算与展示格式 |
| C7 | N-P2-22, N-P2-23, N-P3-21, N-P3-22 | `requirements.txt`, `requirements-mac.txt`, `tests/test_requirements.py` | `main.py`（ImportError 降级）、安装脚本注释 | 不改安装脚本的执行顺序 |
| D1 | N-P1-10, N-P1-11, N-P1-12, N-P1-14, N-P3-23 | `start.bat`, `.github/workflows/release.yml`, `tests/test_requirements.py` | `translator.py`（惰性导入方案） | 不改产物命名与 tag 触发条件 |

## 9. 回滚和失败处理

- 发现实现方向错误时，只回退对应任务的行为代码，不回退已确认的共享契约。
- 外部服务、真实 SCK、真实模型或 GPU 无法使用时记录环境限制，不把功能静默禁用当作完成。
- 每个恢复失败分支必须落到明确终态：继续旧资源、标记不可用或完成退出，不能保留半初始化对象。
- 日志只记录状态转换和资源清理结果，不记录原始音频、API key、完整请求头或凭据。
- D1 若因方案 B（惰性导入）引入启动期行为变化，回退到方案 A（补 CI 依赖）；但「发布必须 needs 测试」这一条不得回退。
- A7 的锁重构若在实测中引入新的竞态，优先回退到「单锁 + `shutdown` 前先 `terminate` 子进程」的折中方案，而不是恢复原状——原状的无界阻塞是必须消除的。

## 10. 执行记录

第一阶段基线 `cbd3a53`，实施分支 `fix/static-audit-2026-08`（已并入 `main`）。
第二阶段直接在 `main` 上继续。测试数 81 → 104 → **190**，全部通过。

### 已完成（27/27）

| 任务 | 覆盖问题 | 改动文件 | 复用入口 / 保持不变的契约 | 运行核对结果 |
|---|---|---|---|---|
| D1 | N-P1-10, N-P1-11, N-P1-12, N-P1-14, N-P3-23 | `start.bat`, `translator.py`, `.github/workflows/release.yml`, `tests/test_requirements.py` | 采用方案 B（惰性导入）；产物命名与 tag 触发条件未变 | `pytest -q` 由 1 failed 转为全绿；临时移除 `needs` 后新断言确实失败、恢复后通过；CI 等价环境验证见下 |
| A1 | P1-1 | `main.py` | 新增 `validate_asr_result` 实现 §2.1；Pipe 协议与 VAD 未动 | 契约测试 16 项通过；`result["text"]`/`result["language"]` 直接下标已全部消除 |
| A2 | P1-2 | `asr_remote.py`, `main.py` | `RemoteASRError` 并入既有 `_recover_asr_worker` 路径；二进制协议未动 | HTTP 500 → 抛 `RemoteASRError`；合法空文本 → 仍返回 `None`；补做：`unload()` 后 `status` 不再报 `ready`（否则 `_switch_asr_engine` 会复用已关闭的 client） |
| A3 | P1-5, P1-6, N-P1-6, N-P2-3, N-P3-5 | `main.py`, `control_panel.py`, `i18n/*.yaml` | 停止顺序照 §2.5；`ControlPanel` 不再自行 `QApplication.quit()`（该 import 已移除） | 9 项新测试：顺序、幂等、满队列不阻塞、单步失败不跳过后续、哨兵不被解引用；`aboutToQuit` 降为兜底 |
| A4 | P1-4, N-P2-21, N-P3-20 | `audio_capture.py`, `audio_capture_base.py` | 新增共享 `enqueue_latest()`（Windows 后端不继承 `AudioCaptureBase`）；采样率/块大小/重启触发条件未动 | 队列竞态测试通过；重启失败改为重试 + 有限次后 `fail_terminally()`；`read()` 移出锁 |
| A5 | P1-3, N-P1-2 | `audio_capture_sck.py`, `main.py` | 与同文件 `set_mic_device` 的相等性短路写法一致；未新增音频回退 | 3 项测试：同名不重启、失败恢复旧设备、恢复也失败则明确 stopped |
| A6 | N-P1-1 | `main.py`, `vad_processor.py` | 锁仍归 `LiveTranslateApp._vad_lock`；阈值与切分算法未动 | `_reset()` 不变式测试通过；`main.py` 中 VAD 写操作已全部在锁内 |
| A7 | N-P1-4, N-P2-5 | `asr_client.py`, `asr_remote.py` | 按 §2.6 拆 `_io_lock` / `_state_lock` + `_cancelled`；Pipe 消息语义与 `asr_worker.py` 未动 | 10 项新测试：退出不等待在途请求、忙时直接 terminate、取消中断轮询、句柄关闭竞态、id 失配归入可恢复错误、`terminate()` 无条件置 `failed` |
| A8 | N-P1-7, N-P1-8, N-P1-13, N-P2-6～9, N-P2-19, N-P3-9 | `mlx_service.py`, `control_panel.py`, `main.py`, `i18n/*.yaml` | 复用既有 `_MLXHealthThread` 模式（§2.9）；HY-MT 转换流程与 `/v1` 协议未动 | 6 项新测试：版本探测按 venv mtime 缓存、服务运行时拒绝替换模型目录、静默子进程可取消、无 `killpg` 平台不抛 `AttributeError`、文案本地化 |
| A9 | N-P1-9, N-P2-10, N-P3-10～12 | `dialogs.py`, `i18n/*.yaml` | 改用 `QThread` + 信号，与 `_ConnectionTestThread`/`_MLXTaskThread` 同形；下载模型集合与 hub 选择未动 | 10 项新测试：`fileno`/`encoding`/`errors` 齐备、异常与重复恢复、不覆盖他人替换的 stderr、向导默认值取自 `config.yaml` |
| A10 | N-P2-15, N-P3-14 | `audio_capture_pyaudio.py`, `audio_capture_base.py`, `audio_capture.py` | `get_audio()` 三态契约写入基类 docstring | 8 项新测试；`__del__` 不再做阻塞 join |
| B1 | P2-1, N-P2-2, N-P3-2 | `translator.py` | 新增 `_finalize()` 单一收口 + `_stream_chunks()` 单一流式循环；生成器接口未变 | 流式/非流式重复检测一致；`stream_options` 回退只捕获参数类异常；尾部复读可检出，实测 0.002ms/次 |
| B2 | P2-3, N-P2-11～13 | `benchmark.py` | 直接复用 `Translator._build_request_kwargs()` | 6 项新测试：与运行时请求逐字段相等、HY-MT 托管配置、不写 history、回退只针对参数错误 |
| B3 | P2-6 | `main.py` | 最小取消标志，未引入状态机 | `_is_running` 初值改为 `False`；`on_pause`/`on_quit` 置取消；`start()` 成功后若已被取消则回滚 |
| B4 | P2-5 | `subtitle_window.py` | 单 timer + FIFO；最大句数与自动隐藏未动 | 4 项新测试：突发按序显示、同时只有一个 timer、clear 后旧 timer 不写入 |
| B5 | N-P1-5 | `main.py`, `i18n/*.yaml` | 新增 `TranslationUnavailable` 与 `_finalize_untranslated`；未合并 `_process_segment*` | 冒烟验证两条路径；补做：`_translate_extra_langs` 的 `None` 解引用走同一异常 |
| B6 | N-P1-3, N-P3-6 | `main.py`, `vad_processor.py` | 密度阈值与分句规则未动 | 6 项新测试：同一短句不重复缓冲、缓冲有上限、密度丢弃计数驱动状态复位 |
| B7 | N-P2-1, N-P3-1 | `main.py` | generation 机制本身未动 | 过期回调改为直接 `return False`；`_on_model_changed` 改走 `_set_translation_workers()` 以复用钳制 |
| B8 | N-P2-17, N-P3-16, N-P3-17 | `subtitle_window.py` | 断行偏好规则与动画未动 | 换行改二分查找（400 字符从约 8000 次测量降到 < 400）；`apply_settings` 复用 `_rebuild_text_widgets` |
| B9 | N-P2-20, N-P3-19 | `subtitle_settings.py`, `subtitle_window.py` | 设置字段格式未动 | `get_settings()` 拆为纯读取 + `emit_settings()`；`lines` 逐元素复制，两侧不再共享可变对象 |
| C1 | P2-2, N-P3-7 | `control_panel.py` | 模型字段格式未动 | 抽出纯函数 `active_index_after_removal()`，7 项测试含穷举属性检查；`_models()` 保证写回持久化对象 |
| C2 | P2-4, N-P2-14, N-P3-13 | `asr_server.py`, `REMOTE_ASR.md` | `/transcribe` 二进制协议与 GPU lock 未动 | 默认绑定改 `127.0.0.1`；新增 `create_app()`/`lifespan`/可选 token/413 上限；两种入口共用 `default_config()`；模块可在无 FastAPI 环境导入，故离线 CI 也能验证其配置面 |
| C3 | P3-1 | `funasr_nano/model.py` | 正常推理流程未动 | 加载失败改为 `raise ... from e` 并带输入引用，不再落到 `UnboundLocalError`；`meta_data` 可变默认参数改 `None` |
| C4 | N-P3-3 | `asr_worker.py` | Pipe 字段与 `recoverable` 语义未动 | 明确转发/忽略并记 debug 日志；签名反射按 engine 实例缓存 |
| C5 | N-P2-4, N-P2-16, N-P3-4, N-P3-8, N-P3-15, N-P3-18 | `main.py`, `mlx_service.py`, `control_panel.py`, `i18n.py`, `i18n/*.yaml`, `subtitle_overlay.py` | `mlx_service` 仍不依赖 i18n：改为可注入的 `translate` 回调 | AST 扫描全仓 UI 调用中的中文字面量，仅剩 `["English", "中文"]`（语言名按惯例用本族名，非可翻译文案）；导出超过 50 条时提示截断并给出 transcript 路径 |
| C6 | N-P2-18 | `control_panel.py` | 缓存条目计算与展示格式未动 | 改为 `_CacheScanThread` + `isRunning()` 去重；`threading` 在该文件只剩 `cancel_event` 一处 |
| C7 | N-P2-22, N-P2-23, N-P3-21, N-P3-22 | `requirements.txt`, `requirements-mac.txt`, `tests/test_requirements.py`, `asr_gigaam.py` | 安装脚本执行顺序未动 | `pyannote-audio`/`torchcodec` 归属已核实（见下）并移除；新增双向对称断言，实测移除 `socksio` 后确实失败 |
| 计划外 | 见「实施中新发现」 | `audio_capture_sck.py`, `.github/workflows/release.yml`, `tests/*` | — | 三处会让真实 CI 变红的缺陷 |

### 实施中新发现（审查未抓到）

| 问题 | 位置 | 处理 |
|---|---|---|
| **第四个 CI 缺陷**：`test_gigaam_cache_is_hf_snapshot_only` 依赖「silero-vad 恰好已安装」 | `tests/test_m2_platform.py` | 已修：monkeypatch `_has_silero_pkg` |
| **第五个 CI 缺陷**：`test_i18n_locales_define_the_same_keys` 等 import `yaml`，但 CI 的依赖清单里没有 PyYAML | `.github/workflows/release.yml` | 已修：CI 依赖补 `PyYAML>=6.0`（i18n 与配置测试确实需要解析 YAML） |
| **第六个 CI 缺陷**：`_SCKStreamDelegate` 的非 macOS 回退分支从未生效——`_SCKStreamDelegate(self)` 对无 `__init__` 的桩类抛 `takes no arguments`，4 项 SCK 测试只在恰好装了 pyobjc 的机器上通过 | `audio_capture_sck.py:464` | 已修：回退路径改用同样的两步 init（`_SCKStreamDelegate().initWithOwner_(self)`） |
| `RemoteASREngine.status` 恒返回 `"ready"`，`unload()` 之后仍如此，`_switch_asr_engine` 会复用已关闭的连接 | `asr_remote.py` | 已修：随 A7 的状态保真一并处理 |
| `_process_interim_final` 的语言回退现有两份 | `main.py` | **刻意保留**：早返分支绕过噪声过滤，否则缓冲的短应答（≤8 字符）会在 ≥2 秒片段里被当噪声丢弃 |

### CI 等价环境验证（D1 最后一项）

沙箱无网络建不了干净虚拟环境，改用**符号链接构造的等价 site-packages**：按
`importlib.metadata` 解析 CI 声明依赖（numpy / pytest / yasbd-lib / PyYAML）的完整传递闭包，
只链接这些分发提供的顶层模块，再以 `python -S` + 重写 `sys.path` 运行 `pytest -q`。
这与元路径拦截器不同——它是**真正的缺包**，`importlib.util.find_spec()` 的返回值语义也正确。

结果：**116 passed, 26 skipped, 0 failed, 0 collection error**。修复前是 6 failed
（上表中的第五、第六个 CI 缺陷各 1 项和 4 项，加 1 项本轮新写的测试缺 `importorskip`）。

复现方式记录在此，便于下次改动后重跑：链接依赖闭包 → `python -S -c "sys.path = [repo, fake_site] + [非 site-packages 路径]; pytest.main(['-q','tests'])"`。

### `pyannote-audio` / `torchcodec` 归属核实（C7 遗留项）

结论：**不是必需传递依赖，已移除**。依据：

1. `pip show` 显示 `pyannote-audio` 的 `Required-by` 为空；`torchcodec` 仅被 `pyannote-audio` 需要。
2. GigaAM-v3 的远程代码（`modeling_gigaam.py:283`）只在 `get_pipeline()` 内 import `pyannote.audio`，
   而 `get_pipeline()` 只被 `transcribe_longform()` 调用。
3. `asr_gigaam.py` 的 docstring 明确不使用 `transcribe_longform`（worker 只喂 VAD 尺寸的短片段），
   且该路径还要求 `HF_TOKEN` 环境变量。

移除理由与恢复条件已写进 `requirements-mac.txt` 与 `asr_gigaam.py` 的注释，并有测试防止无理由地重新加入。

### 未完成 / 环境限制

- **未实机运行**：macOS 设备切换、MLX 服务生命周期、下载对话框取消、远程 ASR 服务、Windows 采集
  读循环（本机为 macOS）均未在真实环境跑过。结论基于源码审查与单元测试。§7 中标记为「实际运行」
  的核对项因此仍未勾选。
- **`ruff` 不可用**：venv 与系统均未安装。回退到 `python -m compileall -q .`（通过）加一个
  基于 AST 的 F401 扫描，其结果只剩既有的 `# noqa` 标注项、`from __future__ import annotations`
  和平台条件再导出，无新增。
- **真实 CI 未跑**：上述等价环境验证覆盖了「能否收集与运行」，但 GitHub Actions 上的
  `bash -n`、macOS runner 版本差异等仍需推分支确认。
- 改动尚未提交。
