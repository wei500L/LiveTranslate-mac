# LiveTranslate 调用链审查报告

审查日期：2026-08-23  
审查基线：当前工作树源码  
审查方式：静态源码审查；未运行测试、编译或外部服务验证；源码零改动。

## 结论摘要

主链路为：

```text
main.py
  -> AudioCapture.start / capture thread
  -> VADProcessor.process_chunk
  -> _enqueue_asr / ASR queue
  -> _asr_loop
  -> ASRClient -> asr_worker.py -> concrete ASR engine
     或 RemoteASREngine -> asr_server.py
  -> _process_segment / _process_segment_text
  -> Translator.translate_iter
  -> SubtitleOverlay / SubtitleWindow / TranscriptWriter
```

发现 13 项确定性问题：

| 优先级 | 数量 | 影响 |
| --- | ---: | --- |
| P1 | 6 | 采集线程或 ASR 线程退出、远程 ASR 故障后永久静默、退出时资源未清理 |
| P2 | 6 | 非流式翻译、模型选择、启动状态、基准测试、ASGI 入口和字幕更新行为不一致 |
| P3 | 1 | FunASR-Nano 异常路径会遮蔽原始错误 |
| P0 | 0 | 本轮未发现有充分源码证据的问题 |

## 调用链问题

### P1-1：ASR 线程没有顶层异常隔离，单个结果契约异常会永久停止 ASR

位置：`main.py:2049-2082`（`LiveTranslateApp._asr_loop`）；结果消费位置 `main.py:1611-1630`、`main.py:1783-1797`。

证据：`_asr_loop` 只捕获 `queue.Empty`，随后直接调用 `_process_segment`、`_process_interim_final` 或 `_do_interim_asr`。这些函数对 ASR 返回值直接做字典下标访问，例如 `result["text"]`、`result["language"]`，没有 schema 校验或外围 `try/except`。`_do_interim_asr` 的异常保护只覆盖 `_run_asr` 调用，覆盖不到后续句子切分和结果访问。

触发条件：后端返回 `None` 之外的非字典结果、缺少 `text`/`language` 字段，或句子切分/输出阶段抛出未预期异常。

影响：ASR 守护线程直接退出；捕获线程仍可能继续向队列写入，但没有消费者，UI 不再产生新的字幕，且不会触发现有 worker 自动重启逻辑。

建议：

1. 在 `_asr_loop` 的每个 item 处理外围增加 `try/except Exception`，记录 `seg_type` 并继续循环；停止流程使用 `finally` 保证状态清理。
2. 在 `_process_segment`、`_process_interim_final` 和 `_do_interim_asr` 入口统一校验 `dict`、`text` 和 `language`，把格式错误转换为可观测的 ASR 错误。
3. 为 worker/remote 两条路径定义同一份结果 schema，缺字段时显式失败，不返回半有效对象。

### P1-2：远程 ASR 将所有请求故障吞成 `None`，绕过自动恢复并造成永久静默

位置：`asr_remote.py:81-113`（`RemoteASREngine.transcribe`）；恢复条件 `main.py:1102-1120`、`main.py:1187-1238`。

证据：远程请求、HTTP 状态、JSON 解析异常在 `asr_remote.py:93-99` 被统一捕获后直接 `return None`。主流程只有在 `ASRWorkerExited`、`ASRWorkerTimeout` 时调用 `_recover_asr_worker`；收到 `None` 时 `main.py:1611-1613` 将其视为“无识别结果”并静默返回。

触发条件：远程服务宕机、连接超时、HTTP 5xx、返回非 JSON 或协议不兼容。

影响：用户只看到 ASR 不再产出字幕，没有错误提示、重连或 worker 状态变化；后续音频会持续被丢弃，直到手动切换模型或重启应用。

建议：

1. 将网络/协议异常转换为专用 `RemoteASRError`，不要与“空语音结果”共用 `None`。
2. 在 `_run_asr` 中对该异常执行与 worker 退出相同的恢复/退避路径，并在 UI 显示远程服务不可用。
3. 仅当响应是合法字典且 `text` 为空时返回 `None`；对 `data` 类型、字段类型和 `language` 做校验。

### P1-3：macOS 设备切换失败后 backend 已停止，但主流程忽略 `set_device()` 返回值

位置：`audio_capture_sck.py:527-537`（`SCKAudioCapture.set_device`）；`audio_capture_sck.py:579-580`（`MacAudioCapture` 转发）；`main.py:411-418`（设置应用）。

证据：`SCKAudioCapture.set_device` 在运行中先 `stop()`，再 `start()`；启动失败时只 `return False`，没有恢复旧设备或旧 stream。`main.py:413` 调用 `self._audio.set_device(...)` 但不检查返回值，只要配置变化就 flush VAD 并继续运行。

触发条件：ScreenCaptureKit 权限暂时不可用、显示器/音频内容重建失败、设备切换时系统返回错误。

影响：底层 capture backend 已停止，`LiveTranslateApp._running` 仍为真；捕获线程会反复拿到空数据，主链路表现为无字幕且无明确错误。

建议：

1. `set_device` 失败时恢复旧设备和旧 stream，恢复失败再把 backend 标记为 stopped。
2. `main.py` 检查返回值；失败时恢复旧设置、提示用户并停止或重启采集，而不是继续保持“运行中”状态。
3. 给 macOS 设备切换增加“停止、重建、失败恢复”的状态转换测试。

### P1-4：Windows 音频队列满时存在 `get_nowait()` 竞态，可能直接杀死采集线程

位置：`audio_capture.py:432-439`（`AudioCapture._read_loop`）。对比实现：`audio_capture_base.py:117-127` 已正确捕获 `queue.Empty` 和第二次 `queue.Full`。

证据：生产者 `put_nowait()` 抛出 `queue.Full` 后直接调用 `self.audio_queue.get_nowait()`，但消费线程可能在这两个调用之间取走了一个元素；此时 `get_nowait()` 抛出 `queue.Empty`，该异常没有被 `_read_loop` 捕获，外层循环随之退出。

触发条件：ASR/捕获处理短时拥塞，队列恰好满；同时 capture consumer 在异常处理窗口内消费一个 block。

影响：音频采集线程退出，后续没有新的音频进入 VAD；主线程和 UI 仍可能显示运行状态，形成静默失效。

建议：直接复用 `AudioCaptureBase._enqueue` 的“丢最旧、捕获 Empty/Full”逻辑，或在此处完整捕获 `queue.Empty`、`queue.Full`，并设置明确的 `last_error`/停止信号。

### P2-1：非流式翻译路径绕过重复输出检测

位置：`translator.py:376-385`（`Translator.translate`）；`translator.py:387-400`（`Translator.translate_iter`）；调用方 `main.py:1391`。

证据：`translate()` 在 `translator.py:382-384` 调用 `_check_repetition` 并抛出 `RepetitionError`。但 `translate_iter()` 在 `streaming=False` 分支直接调用 `_translate_sync`、追加 history 并 `yield result`，没有执行 `_check_repetition`。主流程统一使用 `translate_iter()`，因此关闭 streaming 后不会触发重复检测和对应 UI 错误处理。

触发条件：模型配置 `streaming: false`，且返回重复循环文本。

影响：重复/退化译文被当作成功结果写入字幕和 transcript，质量保护逻辑仅对流式配置生效。

建议：将 repetition check、thinking-budget 诊断和 history 提交收敛到一个共享的“完成结果”函数，并由 `translate()` 与 `translate_iter()` 两条路径共同调用。

### P1-5：删除缓存并退出绕过 LiveTranslateApp.stop()

位置：`control_panel.py:1268-1291`（`ControlPanel._delete_all_and_exit`）；退出清理连接 `main.py:2240`、`main.py:2715-2717`。

证据：缓存删除完成后直接调用 `QApplication.instance().quit()`。主程序只有托盘 `on_quit()` 会先执行 `live_trans.stop()`；`aboutToQuit` 只连接了 `live_trans._mlx_service.stop`，没有连接 ASR worker、音频采集线程、翻译线程池和 transcript writer 的清理。

触发条件：用户在控制面板缓存页确认“删除全部并退出”。

影响：Qt 事件循环退出时，`LiveTranslateApp.stop()` 不会执行；ASR 子进程、音频设备句柄、翻译线程池和转写文件可能未正常关闭，尤其可能留下锁定文件、未刷新的转写内容或后台子进程。

建议：把统一退出函数注册到 `aboutToQuit`，或让缓存删除调用与托盘退出相同的 `live_trans.stop()` 路径；删除缓存前先停止 pipeline，删除后再退出。

### P1-6：停止流程向满载 ASR 队列写哨兵时可能永久阻塞

位置：`main.py:1529-1540`（`LiveTranslateApp.stop`）；队列容量和消费者位置 `main.py:278`、`main.py:2049-2065`。

证据：`stop()` 先把 `_running` 置为 `False`，停止并等待采集线程后，使用无超时的 `_asr_queue.put(None)` 写入退出哨兵。该队列固定容量为 16；如果 ASR 线程已经因未捕获异常退出，或仍未能及时消费队列，队列可能保持满载，`put(None)` 会一直等待。由于该调用通常发生在 Qt/UI 退出路径，主界面会卡在停止阶段。

触发条件：ASR 线程异常退出后队列积压，或停止时 ASR 正在处理长耗时请求且队列已满。

影响：用户点击停止/退出后应用无法完成清理；后续的翻译线程池关闭、转写文件关闭、ASR worker 和 MLX 服务回收都不会执行，放大已有 ASR 线程异常问题的影响。

建议：停止信号使用 `put_nowait(None)` 并在队列满时丢弃/清空一个或多个待处理项；或改用独立 `threading.Event` 作为退出条件，并让消费者用短超时轮询。`stop()` 必须有有界等待和最终清理路径，不能依赖消费者一定存在。

### P2-6：启动延迟回调会覆盖用户在启动窗口内的暂停操作

位置：`main.py:2275-2279`（运行状态初值）；`main.py:2761`（无条件延迟启动）；`main.py:2296-2300`（暂停处理）。

证据：托盘状态 `_is_running` 初始化为 `True`，但真正的 `live_trans.start()` 通过 `QTimer.singleShot(500, on_start)` 延迟执行。用户在这 500ms 内点击暂停时，`on_pause()` 只设置 `_is_running=False` 和界面状态；已经排队的 `on_start` 仍会无条件执行，启动音频/ASR/翻译线程并把状态重新设为运行。

触发条件：应用刚启动、延迟启动回调尚未执行时，用户快速点击托盘或 overlay 的暂停按钮。

影响：用户明确发出的暂停命令被丢失；后端开始采集并处理音频，界面状态与用户最后一次操作不一致。自动启动本地 MLX 服务期间也可能出现相同的时序误解。

建议：用可取消的启动定时器，或在 `on_start()` 入口检查显式暂停/退出标志；初始 `_is_running` 应与真实后端状态一致（通常为 `False`），只有成功启动后再设为 `True`。

### P2-2：删除非活动模型时 active_model 索引未调整

位置：`control_panel.py:1654-1666`（`ControlPanel._remove_model`）。

证据：删除模型后只在 `active >= len(models)` 时把索引压到末尾；当删除行号小于 active 索引时，没有执行 `active -= 1`。例如活动模型为索引 2，删除索引 0 后，原活动模型变为索引 1，但设置仍保存为 2。

触发条件：模型列表存在多个模型，活动模型不是第一个，用户删除其前方任意模型。

影响：应用会把另一个模型误认为活动模型；`_refresh_model_list()`、overlay 菜单和后续翻译器切换均可能指向错误配置。更严重的是，`_remove_model()` 只发出 `models_list_changed`（该信号只更新 overlay 列表），没有在活动项被删除或索引变化后发出 `model_changed`；运行中的 `LiveTranslateApp._translator` 因而继续持有已删除模型的客户端配置，直到用户手动再次切换模型或重启应用。

建议：删除前保存旧 active，删除后按三分支调整：删除活动项则选择相邻项；删除活动项之前的项则 `active -= 1`；删除之后的项保持不变。若活动模型发生变化，必须同时发出 `model_changed`，让运行时 Translator 与持久化 active 索引保持一致；`models_list_changed` 仅用于同步列表显示。

### P2-3：Benchmark 请求没有复用运行时 Translator 的请求参数

位置：`benchmark.py:65-123`（`run_benchmark._test_model`）；运行时组装位置 `translator.py:343-374`、`main.py:619-635`。

证据：基准测试固定发送 `max_tokens=256`、`temperature=0.3`、`stream=True/False`，只处理 `no_system_role`；没有应用模型的 `overrides`、`extra_body`、`thinking_style`、`json_response`、`context_turns`，也没有经过 Translator 的响应解析和重复检测。运行时则通过 `_build_request_kwargs()` 统一注入这些参数。

触发条件：模型依赖 thinking 开关、provider-specific `extra_body`、JSON response 或高级 overrides；用户从 Benchmark 页运行测试。

影响：Benchmark 测出的延迟、成功率和输出行为与实际实时翻译不一致；某些模型在运行时可用但 Benchmark 失败，或 Benchmark 成功但实时调用失败。

建议：Benchmark 直接构造与运行时等价的 Translator/request builder，至少复用 `thinking_disable_body`、overrides、proxy、JSON response 和 streaming fallback。

### P2-4：`asr_server.py` 的 ASGI 导入入口缺少 `app.state.args`

位置：`asr_server.py:42-52`（startup `load_model`）、`asr_server.py:122-124`（`health`）、参数初始化 `asr_server.py:127-137`。

证据：`app.state.args` 只在 `if __name__ == "__main__"` 分支中赋值。若按 FastAPI/Uvicorn 常见方式执行 `uvicorn asr_server:app`，模块会被导入但不会执行参数初始化；startup 的 `load_model()` 和 `/health` 都会访问不存在的 `app.state.args`。

触发条件：使用 ASGI server 方式启动，而不是项目文档中的 `python asr_server.py`。

影响：服务启动阶段直接失败，客户端无法连接；健康检查也无法返回模型信息。

建议：提供 `create_app(args)`/环境变量配置，或在模块级给 `app.state.args` 设置默认 Namespace，并让 `__main__` 仅覆盖命令行参数。

### P2-5：字幕窗口的最短显示时间逻辑会丢弃连续到达的字幕更新

位置：`subtitle_window.py:818-835`（`SubtitleWindow._on_update_text`）；取消逻辑 `subtitle_window.py:852-857`。

证据：每次收到新字幕时，`_on_update_text()` 首先无条件调用 `_cancel_pending_segments()`，然后在上一条字幕尚未达到 `_min_display_ms` 时为当前文本创建延迟 `QTimer`。因此第二条尚未显示的文本会在第三条到达时被停止并删除；代码注释所称“queue rapid updates”并未形成队列，实际策略是只保留最后一条待显示文本。

触发条件：字幕窗口开启，两个或多个最终字幕在 1.5 秒最短显示窗口内连续到达（实时语音或较快翻译时常见）。

影响：中间句子不会进入 `_sentences`，也不会显示在 OBS 字幕窗口；主 overlay/transcript 仍可能保留这些句子，导致不同 UI/导出链路的内容不一致。

建议：把待显示句子放入有序队列，按 `_last_insert_time` 逐条调度；清空队列只应发生在显式 `clear()` 或新会话，而不是每个新字幕到达时。

### P3-1：FunASR-Nano 音频加载失败后继续使用未赋值的 `data_src`

位置：`funasr_nano/model.py:379-394`（`FunASRNano.data_load_speech`）。

证据：`load_audio_text_image_video()` 抛异常时，`except` 分支只记录日志，没有设置 `data_src` 或立即返回；随后无条件把 `data_src` 传给 `extract_fbank()`。该路径必然以 `UnboundLocalError` 结束，原始加载异常被二次错误遮蔽。

触发条件：FunASR-Nano 收到不存在、不可读或格式损坏的音频引用，或底层音频解码器失败。

影响：ASR worker 将该请求报告为泛化的命令错误，丢失真实音频解码原因；连续发生时会触发主流程的 worker 错误计数/不可用逻辑，且用户无法从错误信息判断是输入文件问题。

建议：加载失败时显式抛出带路径和原始异常的领域错误，或返回空结果并让上层按统一 ASR 结果契约处理；不要继续执行依赖 `data_src` 的特征提取。

## 其他模块核对

- `asr_client.py` 与 `asr_worker.py` 的启动、ready、请求、超时、shutdown 协议字段一致；未发现确定性的消息 ID 或生命周期错配。
- `audio_capture_base.py` 的 16 kHz/512-sample 归一化及队列背压实现与主流程契约一致。
- `translator.py` 的 streaming、JSON response、thinking style 三条请求组装路径最终都经过 `_build_request_kwargs()`；未发现确定性的参数漏传。
- `subtitle_overlay.py` 的 streaming 更新采用线程安全缓存 + Qt 定时器刷新，`main.py` 的调用方式匹配；未把它误判为信号未发送问题。
- `asr_server.py` 与 `asr_remote.py` 的二进制协议在语言长度、float32 payload 和响应字段上相互匹配；本报告只把客户端异常吞掉列为缺陷。
- `transcript_writer.py` 的原文/译文写入调用顺序与主流程一致；未发现确定性的文件句柄调用错误。
- `model_manager.py` 的 hub/model id 映射、缓存检测和下载分支已逐项核对；未再发现确定性的缓存路径调用错配，但其结果被 `ControlPanel` 的缓存删除流程消费时暴露出退出清理问题（见 P1-5）。
- `dialogs.py` 的连接测试、模型下载、模型编辑和屏幕适配调用均有对应信号/线程收口；未发现除 ASGI 入口外的确定性异常吞噬问题。
- `control_panel.py` 的 VAD、ASR、翻译、字幕和 MLX 设置控件均已逐函数核对；发现模型索引/运行时切换问题（P2-2）及缓存退出问题（P1-5）。
- `benchmark.py`、`funasr_nano/`、平台辅助模块、安装脚本和测试辅助文件均已扫描；Benchmark 与运行时请求契约偏离列为 P2-3。
- `subtitle_window.py`、`subtitle_overlay.py`、`subtitle_settings.py` 的生命周期、Qt 信号和动画路径已核对；确认字幕延迟更新取消逻辑会丢弃连续句子（P2-5）。Overlay 的消息淘汰后异步翻译更新会被按 `msg_id` 忽略，这是与有界消息窗口一致的预期行为，未列为缺陷。
- `funasr_nano/model.py` 的剩余推理、CTC 和导出函数已核对；除音频加载异常遮蔽（P3-1）外未发现新的确定性调用链错误。

## 文件覆盖清单

本轮按 `git ls-files` 的 79 个受版本控制文件逐项核对；源码、脚本、配置、依赖、文档和测试均纳入范围。以下为完整路径清单：

```text
.github/workflows/release.yml
.gitignore
CLAUDE.md
LICENSE
MAC_PORTING_REPORT.md
MAC_PORTING_TODO.md
README.md
README_zh.md
REMOTE_ASR.md
TRANSLATION_INTEGRATION_AUDIT.md
asr_anime_whisper.py
asr_client.py
asr_engine.py
asr_funasr.py
asr_funasr_nano.py
asr_gigaam.py
asr_remote.py
asr_sensevoice.py
asr_server.py
asr_worker.py
audio_capture.py
audio_capture_base.py
audio_capture_pyaudio.py
audio_capture_sck.py
benchmark.py
build_release.ps1
config.yaml
connection_config.py
control_panel.py
dialogs.py
funasr_nano/__init__.py
funasr_nano/ctc.py
funasr_nano/model.py
funasr_nano/tools/__init__.py
funasr_nano/tools/utils.py
i18n.py
i18n/CHANGELOG_en.md
i18n/CHANGELOG_zh.md
i18n/en.yaml
i18n/zh.yaml
install.bat
install.ps1
install.sh
log_window.py
main.py
mlx_service.py
model_manager.py
platform_app.py
platform_clickthrough.py
platform_config.py
platform_fonts.py
platform_permissions.py
requirements-mac.txt
requirements.txt
screenshot/en.png
screenshot/zh.png
start.bat
start.sh
subtitle_overlay.py
subtitle_settings.py
subtitle_window.py
test_audio.py
tests/test_connection_config.py
tests/test_m0_pipeline.py
tests/test_m0_platform.py
tests/test_m1_sck.py
tests/test_m2_platform.py
tests/test_mlx_service.py
tests/test_requirements.py
tests/test_segmentation.py
tests/test_startup_environment.py
tests/test_translator_thinking.py
torch_backend.py
transcript_writer.py
translator.py
ui_theme.py
update.bat
update.sh
vad_processor.py
```

## 建议修复顺序

1. **P1-1 + P1-2 + P1-5 + P1-6**：先保证 ASR/远程故障可恢复，并建立不会被异常线程和满载队列阻塞的统一退出清理路径。
2. **P1-4**：统一 Windows 队列背压实现，避免采集线程竞态退出。
3. **P1-3**：补齐 macOS 设备切换失败恢复和主流程状态处理。
4. **P2-1 + P2-3 + P2-5 + P2-6**：统一翻译完成逻辑、Benchmark 请求构造、字幕延迟队列和启动状态机。
5. **P2-2 + P2-4**：修正模型索引维护并补齐 ASGI server 配置入口。
6. **P3-1**：修复 FunASR-Nano 音频加载异常的错误保真度。

## 验收标准与测试缺口

后续修复应至少覆盖：

- ASR worker 返回缺少 `text` 或 `language` 时，`_asr_loop` 记录错误但继续处理下一段。
- 远程 ASR 在连接拒绝、超时、HTTP 500、非字典 JSON 下进入可见错误/重试状态；合法空文本仍被视为空结果。
- macOS `set_device()` 失败后恢复旧 stream；恢复失败时主流程停止并提示，而不是保持假运行状态。
- Windows 队列满与并发消费交错时，采集线程不退出，且丢弃计数正确。
- `streaming=False` 的重复译文触发与 streaming 路径相同的 `RepetitionError` 处理。
- 删除缓存并退出、托盘退出、窗口关闭三条路径都执行同一个 pipeline/worker 清理函数；ASR 队列满或 ASR 线程已退出时，`stop()` 仍在有界时间内完成。
- 应用启动延迟期间点击暂停后，不会再被延迟 `on_start` 覆盖；真实后端状态、overlay 和 tray 状态一致。
- 删除活动模型之前的模型后，active index、ControlPanel、overlay、tray menu 和运行中的 Translator 保持一致；删除活动模型时立即切换到新活动模型。
- Benchmark 与实时 Translator 对同一模型生成等价请求。
- `python asr_server.py` 与 `uvicorn asr_server:app` 都能初始化同一套服务配置。
- 字幕最短显示时间内连续提交多条句子时，所有句子按顺序显示，不因后续更新取消中间句。
- FunASR-Nano 音频解码失败时，错误中保留原始路径/解码异常，不出现 `data_src` 未定义。

本轮未执行测试命令，因此上述问题均为源码证据结论，尚未附运行时复现日志。
