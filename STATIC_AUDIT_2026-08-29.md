# LiveTranslate 静态审查报告（第二轮）

审查日期：2026-08-29
审查基线：当前工作树源码（commit `cbd3a53` 之后的未提交状态）
审查方式：纯静态源码审查。未运行应用、未执行测试、未连接外部服务，源码零改动。
工具：`python -m compileall`（通过）+ 自建 AST 检查器（环境无 ruff/pyflakes）。

## 结论摘要

上一轮 `CALL_CHAIN_AUDIT.md` 的 13 项问题**逐项复核后确认全部仍然存在**，`CALL_CHAIN_FIX_TODO.md` 的执行记录表为空白。

本轮新增 59 项发现，分三批：

**第一批（核心管线，18 项）**——`main.py`、`vad_processor.py`、`translator.py`、`asr_client.py`、`transcript_writer.py`、`audio_capture*.py`：

| 优先级 | 数量 | 影响 |
| --- | ---: | --- |
| P1 | 6 | 数据竞争、macOS 音频反复重启、文本重复、退出阻塞、消息永久悬挂、删缓存破坏运行中进程 |
| P2 | 5 | 计数漂移、失败延迟翻倍、采集线程可被杀、i18n 缺口 |
| P3 | 7 | 参数越界、检测能力弱、导出静默截断、清理不可重入等 |

**第二批（扩展模块，25 项）**——`mlx_service.py`、`dialogs.py`、`benchmark.py`、`asr_server.py`、`model_manager.py`、`audio_capture_pyaudio.py`、`i18n.py`、`tests/`、CI：

| 优先级 | 数量 | 影响 |
| --- | ---: | --- |
| P1 | 6 | **当前 main 测试为红**、CI 无法收集测试、测试失败仍发布、MLX 5 秒无退避重试、删运行中模型、stderr 全局泄漏 |
| P2 | 11 | 取消无效、异常覆盖、Benchmark 崩溃路径、远程服务无认证、后端行为不一致 |
| P3 | 8 | 废弃 API、i18n 绕过、默认值不一致等 |

**第三批（UI、后端与依赖，16 项）**——`subtitle_window.py`、`subtitle_settings.py`、`control_panel.py` 余下部分、`audio_capture.py` 读循环、ASR 后端、`requirements*.txt`、安装/构建脚本：

| 优先级 | 数量 | 影响 |
| --- | ---: | --- |
| P1 | 1 | 点选 HY-MT 模型冻结 UI 约 2 秒 |
| P2 | 7 | O(n²) 换行、扫描线程无去重、面板不可关闭、共享可变状态、设备重启静默丢弃、依赖声明缺口 |
| P3 | 8 | 代码重复、空值分支不一致、硬编码文案、未使用重依赖被测试固化等 |

> **最高优先级**：`tests/test_startup_environment.py` 当前失败，根因是最近一次提交 `cbd3a53` 修改了 `start.bat` 的提示文案。详见 N-P1-10。

---

## P1 新发现

### N-P1-1：VAD 缓冲区跨线程数据竞争（Qt 线程绕过 `_vad_lock`）

**位置**：`main.py:384`、`main.py:415-416`、`main.py:811-812`
**对照**：`main.py` 中其余 7 处 VAD 访问（1249、1755、1860、1989、2008、2081）都正确持有 `self._vad_lock`。

**证据**：

```python
# main.py:415-416  _on_settings_changed —— Qt 主线程，无锁
self._vad.flush()
self._vad._reset()

# main.py:811-812  _switch_asr_engine —— Qt 主线程，无锁
self._vad.flush()
self._vad._reset()

# main.py:384 —— Qt 主线程，无锁
self._vad.update_settings(settings)
```

而采集线程在 `main.py:2008-2009` 持锁执行 `self._vad.process_chunk(chunk)`，该函数会 `self._speech_buffer.append(...)`、`self._confidence_history.append(...)`、`self._speech_samples += ...`。

`VADProcessor._reset()`（`vad_processor.py:340-346`）把 `_speech_buffer` 和 `_confidence_history` **分两条语句重新绑定为新列表**：

```python
def _reset(self):
    self._speech_buffer = []
    self._confidence_history = []
    self._speech_samples = 0
```

**触发条件**：用户在说话过程中改动控制面板任意设置，或切换 ASR 引擎。

**影响**：
1. `_reset()` 的两次重新绑定之间，采集线程可能向旧 `_speech_buffer` 追加而向新 `_confidence_history` 追加，造成两个列表**长度失配**。该失配会污染 `_find_best_split_index()`（按 `_confidence_history` 索引切 `_speech_buffer`）和 `_flush_segment()` 的密度判定，直到下一次完整 reset。
2. `flush()` 内的 `np.concatenate(self._speech_buffer)` 可能在列表被并发追加时读到不一致快照。
3. `flush()` 的返回值被丢弃——缓冲区里已积累的语音被静默丢掉，而且 `flush()` 已经会 reset，紧随其后的 `_reset()` 是冗余调用。

**建议**：把这 3 处包进 `with self._vad_lock:`；`_reset()` 内部用单条语句原子替换状态；删除 `flush()` 后多余的 `_reset()`，或明确改用 `_reset()` 单独调用。

---

### N-P1-2：macOS 上每次设置变更都会重启 ScreenCaptureKit 音频流

**位置**：`audio_capture_sck.py:527-537`（`SCKAudioCapture.set_device`）、`control_panel.py:1822-1826`、`control_panel.py:1864`、`main.py:411-413`

**证据**：

`ControlPanel._apply_settings()` **每次都写入 `audio_device` 键**，并在末尾发出**全量**设置字典：

```python
# control_panel.py:1822-1826
self._current_settings["audio_device"] = ...   # 无条件写入
# control_panel.py:1864
self.settings_changed.emit(dict(self._current_settings))
```

`main.py:411-413` 用 `in` 判断而非值比较：

```python
if "audio_device" in settings:          # 恒为 True
    old_device = self._audio._device_name
    self._audio.set_device(settings["audio_device"])
```

Windows 后端有相等性短路（`audio_capture.py:260-261` `if device_name == self._device_name: return`），macOS SCK 后端**没有**：

```python
# audio_capture_sck.py:527-537
def set_device(self, device_name):
    self._device_name = device_name
    if self._running:
        self.stop()          # 无条件停止
        try:
            self.start()
        except Exception:
            return False
    return True
```

`SCKAudioCapture.stop()`（`audio_capture_sck.py:490-517`）包含 `self._async_result(..., timeout=5)` 和 `self._worker_thread.join(timeout=3)`。

**触发条件**：macOS 上拖动任意滑块、勾选任意复选框、改动任意下拉框——控制面板 300ms 防抖自动保存后必然触发。

**影响**：即使音频设备毫无变化，也会执行完整的 SCK 流停止+重建。最坏情况在 **Qt 主线程阻塞约 8 秒**（5s 停流 + 3s join），期间 UI 冻结、音频彻底中断、`flush()` 丢弃缓冲。这条与 N-P1-1 叠加时后果更重：`_on_settings_changed` 先无锁 `_vad.flush()/_reset()`，紧接着重启音频。

**建议**：`SCKAudioCapture.set_device` 补上与 `set_mic_device` 一致的相等性短路（`if self._device_name == device_name: return True`）；`main.py` 改成比较值而非检查键是否存在；把 SCK 重启移出 Qt 线程。

---

### N-P1-3：增量 ASR 的短句缓冲会重复累积同一段文本

**位置**：`main.py:1832-1868`（`_do_interim_asr` 提交循环）

**证据**：

```python
actually_committed = False
for sent in complete:
    text = sent.strip()
    if self._is_short_utterance(text):          # ≤8 个字母数字字符
        self._interim_pending += text           # 累加，不去重
        continue
    ...
    self._process_segment_text(text, ...)
    actually_committed = True

if not actually_committed:
    return False                                # ← 提前返回：不 trim、不更新 committed_tail

if trim_samples > 0:
    self._vad.trim_front(trim_samples)
self._interim_committed_tail = committed_text[-50:] ...
```

当本轮 `complete` 中**全部**是短句时，`actually_committed` 为 False，函数在 trim 与 `_interim_committed_tail` 更新之前返回。因此：

- VAD 缓冲区**未被裁剪**，下一轮 interim 会对包含同一段音频的更长缓冲重新识别；
- `_interim_committed_tail` **未更新**，`_strip_committed_overlap()`（`main.py:1735-1750`）无法去除回声；
- 同一批短句被再次匹配为短句，再次执行 `self._interim_pending += text`。

**触发条件**：句首出现短应答/填充词（日语「はい。」「うん。」、英语 "Yeah." "OK."），随后语音继续。

**影响**：`_interim_pending` 中同一片段被复制 N 次（N = 出现长句前的 interim 轮数）。该字符串随后在 `main.py:1852-1854` 被前置到下一条长句：

```python
if self._interim_pending:
    text = self._interim_pending + text
```

最终字幕与 transcript 出现「はい。はい。はい。这是一个长句。」这类重复。`_interim_pending` 同时**没有任何长度上限**。

**建议**：短句缓冲后也执行 trim 并更新 `_interim_committed_tail`；或对 `_interim_pending` 做追加前去重 + 长度封顶（例如 200 字符）。

---

### N-P1-4：`ASRClient._lock` 抵消了「不阻塞 Qt 线程」的设计，退出/切换最长阻塞 120 秒

**位置**：`asr_client.py:158-181`（`_request`）、`asr_client.py:123-149`（`shutdown`）、`main.py:1096-1101`

**证据**：

`_run_asr` 有明确注释说明它刻意在调用 `transcribe` 前释放 `_asr_lock`：

```python
# main.py:1096-1100
# Snapshot the active client under the lock, then release it: the blocking
# cross-process transcribe must not hold _asr_lock, or a slow/hung worker
# would freeze the Qt thread on every settings change.
with self._asr_lock:
    ...
    client = self._asr
```

但 `ASRClient._request()` 在整个跨进程往返期间持有 `self._lock`（RLock），超时上限是 `request_timeout=120.0`：

```python
def _request(self, request_type, payload, timeout):
    with self._lock:                     # 持锁直到响应返回或超时
        ...
        response = self._recv_response(timeout, expected_id=msg_id)
```

`shutdown()` 同样以 `with self._lock:` 开头。因此 Qt 线程调用 `_shutdown_asr_worker()` → `client.shutdown()` 时，会在 `ASRClient._lock` 上阻塞，直到 ASR 线程的 `transcribe` 结束。

**触发条件**：worker 卡住或处理超长片段时，用户退出应用或切换 ASR 引擎。

**影响**：`_run_asr` 规避 `_asr_lock` 的努力被完全抵消——UI 仍会冻结，最长约 120 秒。`stop()` 中 `self._asr_thread.join(timeout=10)` 超时后继续执行 `_shutdown_asr_worker()`，会在锁上再等最多 110 秒，退出总时长可达约 2 分钟。`wait_ready()` 同样持锁最长 `ready_timeout=180.0` 秒。

**建议**：`shutdown()`/`terminate()` 走独立的取消路径，不与 `_request` 争用同一把锁——例如先设置取消标志并直接 `process.terminate()`，再获取锁做句柄清理；或给 `_lock` 的获取加超时。

---

### N-P1-5：翻译器不可用时消息永久悬挂，且 `TranscriptWriter._pending` 无界泄漏

**位置**：`main.py:1690-1696`、`main.py:1917-1920`、`main.py:685-687`、`transcript_writer.py:70-77`

**证据**：

`_snapshot_translation_request` 在翻译器被禁用时抛出 `RuntimeError`：

```python
# main.py:685-687
base = self._translator
if base is None:
    raise RuntimeError("No translation service is running")
```

`_process_segment` / `_process_segment_text` 把它当作「执行器已关闭」吞掉：

```python
# main.py:1691-1696
try:
    self._submit_translation(msg_id, original_text, source_lang, extra_langs or None)
except RuntimeError:
    log.warning("Translation executor shut down, skipping")
```

但此前已经执行过：

```python
self._overlay.add_message(msg_id, timestamp, original_text, source_lang, asr_ms)
self._transcript.write_original(msg_id, timestamp, original_text)
```

而 `TranscriptWriter.write_original` 会写入 `_pending` 字典（`transcript_writer.py:76`），只有 `write_translation()` 或 `finalize_no_translation()` 才会 `pop`——这两个函数在此路径上都不会被调用。

`_translator` 为 None 的路径真实存在：`_disable_translator()` 在 `main.py:2260`（HY-MT 本地服务未就绪的启动分支）和 `main.py:592` 被调用。

**触发条件**：选中 HY-MT 但 MLX 本地服务未启动，随后有语音输入。

**影响**：
1. Overlay 中每条消息永远停留在「翻译中」状态，从不落定；
2. `all` transcript 文件缺失这些条目（只有 `original` 有）；
3. `TranscriptWriter._pending` 随会话时长**无界增长**；
4. 日志写的是 "Translation executor shut down"，与真实原因（无翻译服务）不符，误导排查。

**建议**：区分这两种 `RuntimeError`（用专门的异常类型），翻译器不可用时调用 `self._transcript.finalize_no_translation(msg_id)` 并在 overlay 上显示明确错误。

---

### N-P1-6：删除缓存时 ASR worker 仍在运行，模型目录被 `rmtree` 破坏

**位置**：`control_panel.py:1268-1291`（`_delete_all_and_exit`）

**证据**：

```python
for name, path, _ in self._cache_entries:
    try:
        shutil.rmtree(path)
        log.info(f"Deleted: {path}")
    except Exception as e:
        log.error(f"Failed to delete {path}: {e}")
QApplication.instance().quit()
```

删除发生时 ASR worker 子进程仍存活，并且持有这些模型目录下的文件句柄/mmap（faster-whisper 的 ctranslate2、FunASR 的 torch 权重都会保持映射）。`LiveTranslateApp.stop()` 从未被调用——`app.aboutToQuit` 只连了 `live_trans._mlx_service.stop`（`main.py:2240`）。

**触发条件**：用户在「缓存」页点击「删除全部并退出」。

**影响**：
1. Windows 上文件被占用，`rmtree` 抛异常并被吞进日志，用户以为删干净了，实际残留数 GB；
2. macOS/Linux 上文件被删除但 inode 仍被 worker 持有，磁盘空间直到进程退出才释放，且 worker 可能在读到半删状态时崩溃；
3. `shutil.rmtree` 在 **Qt 主线程同步执行**，删除数 GB 目录期间 UI 完全无响应且无进度提示；
4. 上一轮 P1-5 已指出退出未走 `stop()`，本条补充了「删除动作本身会破坏运行中进程」这一更直接的后果。

**建议**：删除前先执行完整的 `live_trans.stop()`（至少 `_shutdown_asr_worker()`）确认 worker 已退出；把 `rmtree` 放到工作线程并显示进度；删除失败必须在 UI 上报错而非仅记日志。

---

## P2 新发现

### N-P2-1：过期 generation 的翻译回调会递减新一代的 `_translation_pending`

**位置**：`main.py:704-712`

```python
if generation != self._translator_generation:
    self._translation_results.pop(msg_id, None)
    try:
        self._translation_order.remove(msg_id)
    except ValueError:
        pass
    self._translation_pending = max(0, self._translation_pending - 1)   # ← 递减的是新一代计数器
    return False
```

`_disable_translator()` 与 `_on_model_changed()` 都会把 `_translation_pending` 归零并递增 generation。此后旧 generation 的在途请求陆续完成，每个都会把**新一代**的计数器减 1。

**影响**：`_record_latency` 输出的 `translation_pending=%d`（`main.py:1371`）长期偏低甚至恒为 0，性能日志失真。当前该字段只用于日志，未参与控制流，故列为 P2。

**建议**：generation 不匹配时直接 `return False`，不触碰任何新一代的共享计数。

---

### N-P2-2：流式请求的 `stream_options` 回退会捕获所有异常，使失败延迟翻倍

**位置**：`translator.py:407-414`、`translator.py:487-494`（两处完全重复的代码）

```python
try:
    stream = self._client.chat.completions.create(
        **base_kwargs, stream_options={"include_usage": True},
    )
except Exception:
    stream = self._client.chat.completions.create(**base_kwargs)
```

**影响**：该 `except Exception` 本意是兼容不支持 `stream_options` 的服务端（应为 `BadRequestError`/`TypeError`），实际会吞下 `APIConnectionError`、`APITimeoutError`、`AuthenticationError` 并**原样重试一次**。服务不可达时，用户感知的失败延迟从 10 秒变成 20 秒；最终抛出的是第二次尝试的异常，第一次的真实错误信息丢失。这在实时字幕场景下直接表现为长时间无输出。

**建议**：只捕获参数不兼容类异常；把两处重复的流式循环收敛为一个私有生成器，由 `_translate_streaming` 与 `translate_iter` 共用。

---

### N-P2-3：`_enqueue_asr` 可能取出停止哨兵 `None` 并以 TypeError 杀死采集线程

**位置**：`main.py:2035-2047`、`main.py:1536`

```python
def _enqueue_asr(self, seg_type, segment):
    try:
        self._asr_queue.put_nowait((seg_type, segment))
    except queue.Full:
        try:
            dropped = self._asr_queue.get_nowait()
            log.warning(f"ASR queue full, dropped {dropped[0]} segment")   # ← dropped 可能是 None
        except queue.Empty:
            pass
```

`stop()` 在 `main.py:1536` 通过 `self._asr_queue.put(None)` 写入停止哨兵。若采集线程未在 `join(timeout=3)` 内退出并仍在入队，`get_nowait()` 可能取回该 `None`，`dropped[0]` 抛出 `TypeError: 'NoneType' object is not subscriptable`。该异常在 `_capture_loop` 中无任何捕获。

**影响**：采集线程带异常退出；同时停止哨兵被吞掉，`_asr_loop` 只能靠 `while self._running` 在最多 1 秒后退出（该分支尚可自愈，但采集线程的退出路径不受控）。与上一轮 P1-6 属于同一处停止协议缺陷的两个侧面。

**建议**：用 `threading.Event` 而非队列内哨兵表达停止；或在取出后判空 `if dropped is not None`。

---

### N-P2-4：HY-MT 服务未启动的提示硬编码中文，未走 i18n

**位置**：`main.py:589-593`

```python
message = (
    "HY-MT 本地服务尚未启动。\n"
    "请在'翻译'设置中的模型配置区域手动启动本地服务后再选择此模型。"
)
```

同一函数内其余用户可见文案均使用 `t()`（如紧随其后的 `t("error_title")`）。项目已提供 `i18n/zh.yaml` 与 `i18n/en.yaml`。

**影响**：英文界面用户会看到中文错误弹窗。

**建议**：新增 `i18n` 键（如 `error_mlx_not_running`），改为 `t(...)`。

---

### N-P2-5：`ASRClient.terminate()` 后 `status` 可能仍报告 "ready"

**位置**：`asr_client.py:150-157`、`asr_client.py:53-58`

```python
def terminate(self):
    with self._lock:
        if self._process is not None and self._process.is_alive():
            self._status = "failed"          # ← 仅在进程仍存活时才更新状态
            ...
        self._close_handles()                # 把 _process 置为 None
```

若进程已自行退出，`is_alive()` 为 False，`_status` 保持原值（可能是 `"ready"`）。`_close_handles()` 随后把 `_process = None`，于是 `status` 属性的退出检测分支（`if self._process is not None and self._process.exitcode is not None`）永远不成立，属性持续返回 `"ready"`。

**影响**：`_switch_asr_engine` 的复用判断依赖 `current_asr.status == "ready"`（`main.py:794`），可能误判一个已死的 client 为可用，跳过重建。后续 `_request` 会抛 `ASRClientError("ASR worker has not been started")`——该异常类型不在 `_run_asr` 的恢复分支中，不会触发 `_recover_asr_worker`。

**建议**：`terminate()` 无条件设置 `self._status = "failed"`。

---

## P3 新发现

| 编号 | 位置 | 问题 |
| --- | --- | --- |
| N-P3-1 | `main.py:650-655` | `_on_model_changed` 直接给 `self._translation_workers` 赋值，绕过 `_set_translation_workers` 的 `max(4, min(16, ...))` 钳制。`user_settings.json` 中写入 `translation_workers: 100` 会让 `start()` 创建 100 线程池。 |
| N-P3-2 | `translator.py:460-467` | `_check_repetition` 只判断 `text[plen:plen*2] == text[:plen]`，即**仅检测从位置 0 开始**的重复。真实 LLM 复读通常在正常输出若干字后才开始，这类输出检测不到。 |
| N-P3-3 | `asr_worker.py:124-134` | `_transcribe` 只转发 `word_timestamps`，`ASRClient.transcribe(**kwargs)` 传入的其他参数被静默丢弃；且每次调用都执行 `inspect.signature(engine.transcribe)`，实时路径上的无谓开销。 |
| N-P3-4 | `subtitle_overlay.py:914`、`1237-1262` | `_max_messages = 50`，`export_messages` 直接遍历 `self._messages`，因此「导出全部」实际只导出最近 50 条，UI 无任何提示。完整内容仅存在于 transcript 文件中。 |
| N-P3-5 | `main.py:2762` | `signal.signal(signal.SIGINT, lambda *_: on_quit())` 在信号处理器内执行 `live_trans.stop()`，含 3s + 10s 阻塞 join 与子进程回收。`stop()` 无重入保护，第二次 Ctrl-C 会重复执行 `_transcript.close()`、再次 `put(None)`。 |
| N-P3-6 | `vad_processor.py:318-338` | `_flush_segment` 因语音密度 < 25% 丢弃片段时返回 `None`，`_capture_loop` 因而不会入队 `vad_flush`。于是 `_asr_loop` 中的 `_interim_active = False` / `_interim_pending = ""` 清理不会执行，残留状态会泄漏到下一段无关语音。 |
| N-P3-7 | `control_panel.py:1643-1646`、`1656-1658` | `_dup_model` / `_remove_model` 使用 `self._current_settings.get("models", [])`。当键缺失时默认值是一个**临时空列表**，对其 `append`/`pop` 不会写回 `_current_settings`，`_save_settings` 保存的内容与 UI 不一致。 |

---

## 第二批：扩展模块新发现（25 项）

覆盖第一批未深入的模块：`mlx_service.py`、`dialogs.py`、`benchmark.py`、`asr_server.py`、`model_manager.py`、`audio_capture_pyaudio.py`、`i18n.py`、`torch_backend.py`、`tests/`、`.github/workflows/`。

### P1

#### N-P1-10：`start.bat` 文案变更打破测试，当前 main 上测试为红

**位置**：`start.bat:12` · `tests/test_startup_environment.py:11-15`

**证据**：本地实测 `python -m pytest -q` 结果为 `1 failed, 80 passed`：

```text
FAILED tests/test_startup_environment.py::test_source_launcher_rejects_an_incomplete_environment
E   assert 'setup is incomplete' in '...echo environment is incomplete; running the installer first...'
```

`git show cbd3a53 -- start.bat` 显示这是最近一次提交引入的：

```diff
-    echo [ERROR] Virtual environment setup is incomplete.
+    echo Environment is incomplete; running the installer first...
```

测试断言的是 `start.sh` 采用的措辞（`start.sh:8` 为 `"Setup is incomplete; ..."`），`start.bat` 改后两个平台的启动器措辞不再一致。

**影响**：仓库当前状态下测试套件不通过。结合 N-P1-11 与 N-P1-12，这一失败既没有在 CI 中被看到，也不会阻止发布。

**建议**：统一两个启动器的措辞（改 `start.bat` 或同时放宽测试断言为两种措辞之一），并在提交前把 `pytest` 纳入本地检查。

#### N-P1-11：CI 测试依赖不完整，两个测试文件无法被收集

**位置**：`.github/workflows/release.yml:26`（`test-macos-arm64` 作业）

**证据**：

```yaml
- name: Install test dependencies
  run: python -m pip install --disable-pip-version-check "numpy>=1.24,<2.3" pytest
- name: Run offline platform tests
  run: python -m pytest -q
```

但 `translator.py:1-6` 有模块级第三方导入：

```python
import httpx
from openai import OpenAI
```

`tests/test_translator_thinking.py:1` 和 `tests/test_mlx_service.py:9` 都在模块级 `import translator`。仓库中不存在 `conftest.py`、`pytest.ini`、`pyproject.toml` 或 `setup.cfg`，因此没有任何 skip/ignore 配置。

**影响**：这两个文件在 CI 中以 `ModuleNotFoundError: No module named 'openai'` 收集失败，`pytest` 退出码非零。也就是说 CI 的 macOS 测试作业处于失败状态，且其覆盖范围远小于表面上的 81 个用例。

**建议**：CI 依赖补 `httpx`、`openai`；或把 `translator.py` 的第三方导入改为惰性导入，使纯逻辑测试（thinking style 解析、prompt 组装）在无网络依赖下可跑。

#### N-P1-12：`build` 作业未依赖测试作业，测试失败仍会发布

**位置**：`.github/workflows/release.yml`

**证据**：三个作业中只有一个声明了依赖。

```yaml
test-macos-arm64:        # 第 18 行
build:                   # 第 35 行 —— 没有 needs:
  runs-on: windows-latest
  ...
  - name: Attach to GitHub Release
    if: github.ref_type == 'tag'
    uses: softprops/action-gh-release@v2
package-macos-arm64:     # 第 60 行
  needs: test-macos-arm64
```

**影响**：`build` 与测试作业并行执行。打 tag 时，即使测试作业失败，Windows 便携包仍会被构建并附加到 GitHub Release。macOS 包受 `needs` 保护，Windows 包不受保护——两个平台的发布门禁不一致。结合 N-P1-10 和 N-P1-11，当前已经处于「测试红、发布照常」的状态。

**建议**：给 `build` 加 `needs: test-macos-arm64`；若希望 Windows 侧也有自己的测试门禁，另加一个 windows 测试作业。

#### N-P1-8：MLX 服务不可用时每 5 秒无退避重试，并反复重置翻译状态

**位置**：`main.py:358-368`（`_on_mlx_probe_result`）· `main.py:344-346`（5 秒定时器）· `control_panel.py:1534-1548`

**证据**：

```python
def _on_mlx_probe_result(self, running: bool):
    ...
    if running:
        ...
        return
    self._disable_translator()                      # ← 每 5 秒执行一次
    if not self._mlx_restart_pending:
        self._mlx_restart_pending = self._panel.auto_start_mlx_service()
```

`auto_start_mlx_service()` 在环境未就绪时返回 `False`，于是 `_mlx_restart_pending` 保持 `False`，下一次 5 秒探测再次进入同一分支。该函数内部会调用 `is_environment_ready()` → `_versions_are_compatible()`：

```python
# mlx_service.py:177-195
check = subprocess.run([str(self.env_dir / "bin" / "python"), "-c", ...])
```

**影响**：选中 HY-MT 但本地环境未准备好时：

1. 每 5 秒 `_disable_translator()` 一次，`_translator_generation` 持续自增、`_translation_history` 被清空；与 N-P2-1 叠加会让 `_translation_pending` 计数彻底失真，且任何在途翻译都被丢弃。
2. 每 5 秒派生一个 Python 解释器子进程做版本检查，外加一次 `ps` 和一次 1.5 秒 `urlopen` 探测。笔记本上这是持续的 CPU 与电量开销。
3. 与 ASR 侧形成对比：ASR 有 `_asr_restart_count` 上限，MLX 侧既无退避也无次数上限。

**建议**：为 MLX 重试引入与 ASR 一致的退避与次数上限；`_disable_translator()` 只在状态由「可用」变为「不可用」的边沿执行一次，而非每次探测都执行。

#### N-P1-7：`prepare_model` 在服务运行时删除并替换模型目录

**位置**：`mlx_service.py:343-345`

**证据**：

```python
shutil.rmtree(self.model_dir, ignore_errors=True)
os.replace(temp_model_dir, self.model_dir)
```

函数入口只检查 `is_supported_platform()` 与 `is_model_ready()`，**没有检查 `is_running()`**。而 `ensure_running()` 启动的 `mlx_lm.server` 正是以 `--model str(self.model_dir)` 运行并持有该目录下的权重映射。

**影响**：用户在服务运行期间点击「准备本地模型」重新转换时，正在服务的模型目录被删除。属于与 N-P1-6 相同的类别——对运行中进程持有的目录做 `rmtree`。`ignore_errors=True` 还会把失败完全吞掉。

**建议**：`prepare_model` 入口先检查 `is_running()`，运行中则先 `stop()` 或拒绝执行并提示用户；`os.replace` 前再确认一次服务未运行。

#### N-P1-9：下载期间退出会永久留下被替换的 `sys.stderr` 与 root logger handler

**位置**：`dialogs.py:326-328`、`dialogs.py:439-441`（安装）· `dialogs.py:352-354`、`dialogs.py:487-489`（恢复）

**证据**：

```python
logging.getLogger().addHandler(self._log_handler)
self._orig_stderr = sys.stderr
sys.stderr = _StderrCapture(self._log_signal.emit, self._orig_stderr)
```

恢复只发生在 `_check_done()` 中，而 `_check_done` 只在下载线程结束后才执行：

```python
def _check_done(self):
    if self._download_thread.is_alive():
        return
    self._poll_timer.stop()
    sys.stderr = self._orig_stderr
    logging.getLogger().removeHandler(self._log_handler)
```

下载线程是 `daemon=True` 且从不 join。两个对话框都没有 `closeEvent`（`dialogs.py:796` 的 `closeEvent` 属于 `ModelEditDialog`）。

**影响**：下载进行中退出应用（托盘退出、Ctrl-C、强制退出）时，`sys.stderr` 保持为 `_StderrCapture`，其 `write()` 会调用 `self._cb` —— 即已销毁 QDialog 的绑定信号。此后任何 stderr 写入（包括 Python 自身的关闭期报错和 traceback）都会触发 `RuntimeError: wrapped C/C++ object of type ... has been deleted`，真实的错误信息随之丢失。root logger 也会一直持有该 handler。

**建议**：把 stderr/handler 的安装与恢复放进 `try/finally` 或上下文管理器；为对话框加 `closeEvent`/`reject` 时的恢复路径；改用 `QThread` + `finished` 信号（见 N-P3-10）。

### P2

| 编号 | 位置 | 问题 |
| --- | --- | --- |
| N-P2-6 | `mlx_service.py:477-478` | `ensure_running` 的等待循环里 `_check_cancel(cancel_event)` 直接抛异常，**没有调用 `stop()`**。取消一次启动会留下仍在运行的 `mlx_lm.server` 进程和已写入的 pid 文件；只有超时分支（第 496 行）才会 `self.stop()`。 |
| N-P2-7 | `mlx_service.py:230-236` | `_check_cancel` 只在 `for line in process.stdout` 循环体内调用。ModelScope 下载 7B 权重期间若长时间无行输出，取消按钮完全无效——用户只能等下载结束。 |
| N-P2-8 | `mlx_service.py:238-243` | 异常路径 `process.wait(timeout=10)` 自身会抛 `TimeoutExpired`，覆盖掉原始异常（含用户取消）。应包 try/except 后 `kill()`。 |
| N-P2-9 | `mlx_service.py:499-528` | `stop()` 使用 `os.killpg` 与 `signal.SIGKILL`，二者在 Windows 上都不存在，会抛 `AttributeError`——而 `except` 只捕获 `ProcessLookupError`/`OSError`。该方法挂在 `app.aboutToQuit`（`main.py:2240`）上且不分平台。正常情况下 Windows 无 pid 文件会提前返回，但存在残留 pid 文件时退出流程会异常。 |
| N-P2-10 | `dialogs.py:388-497` | `ModelDownloadDialog` 与 `SetupWizardDialog` 的下载**无法取消**：无 `cancel_event`、无取消按钮，窗口也没有关闭按钮（`CustomizeWindowHint` 未带 `WindowCloseButtonHint`）。对比同文件 `_MLXTaskThread` 已具备 `cancel_event`。多 GB 下载期间用户只能强杀进程。 |
| N-P2-11 | `benchmark.py:100-103` | `delta = chunk.choices[0].delta` **未判空**。`translator.py:428` 对同一循环有 `if chunk.choices:` 保护。当 provider 发送 usage-only（空 choices）分片时 Benchmark 抛 `IndexError`，回退到非流式路径，测得的延迟数据失真。 |
| N-P2-12 | `benchmark.py:118` | `resp.choices[0].message.content.strip()` **缺少 `or ""`**。`translator.py:478` 写的是 `(resp.choices[0].message.content or "").strip()`。thinking 模型把预算烧光时返回 `content=None`，Benchmark 抛 `AttributeError`——正是 issue #38 的场景。 |
| N-P2-13 | `benchmark.py:105-106` | 流式失败回退使用宽泛 `except Exception:`，与 N-P2-2 同一反模式。服务不可达时每个句子付出 2× timeout，`rounds` 轮下来等待时间成倍放大。 |
| N-P2-14 | `asr_server.py:129`、`asr_server.py:88-91` | 默认 `--host 0.0.0.0`，`/transcribe` 无认证、无请求体大小限制，`await request.body()` 一次性读入内存后直接触发 GPU 推理。任何同网段主机都可消耗 GPU 或以超大 body 触发内存耗尽。 |
| N-P2-15 | `audio_capture_pyaudio.py:208-212` | `get_audio` 用 `except Exception: return None` 吞掉一切，使 `CaptureRuntimeError` 无法传播。`SCKAudioCapture.get_audio`（`audio_capture_sck.py:519-525`）则会主动抛出该异常，`_capture_loop`（`main.py:1976-1984`）也据此停止管线。结果是同一个终止条件在 SCK 后端会停机、在 PyAudio 后端永远静默——两个后端行为不一致。 |
| N-P2-16 | `i18n.py:21-27`、`i18n.py:38` | `set_lang()` 内 `yaml.safe_load(f.read_text())` 无任何异常保护，且在**模块导入期**执行（`set_lang(_detect_system_lang())`）。i18n YAML 损坏或编码异常会在任何 UI 出现之前直接终止进程，用户看不到任何提示。 |

### P3

| 编号 | 位置 | 问题 |
| --- | --- | --- |
| N-P3-8 | `mlx_service.py` 全模块 | 用户可见文案硬编码且中英混排，完全绕过 i18n：`ensure_running` 的两处错误信息（`"...请在翻译设置中点击'准备本地模型'。"`）、等待提示 `"正在等待 HY-MT 服务加载模型..."`、`prepare_model` 的全部进度文案（`"检查 MLX 运行环境..."`、`"转换为 MLX 4-bit 模型..."` 等）。与 N-P2-4 同类，但数量更多。 |
| N-P3-9 | `mlx_service.py:296-302` | `prepare_model` 在运行期用 `sys.executable -m pip install modelscope>=1.20.0` 把依赖装进**应用自身正在运行的 venv**（而非 `.mlx-venv`）。运行时修改自己的解释器环境，且失败模式与 `requirements.txt` 声明不一致。 |
| N-P3-10 | `dialogs.py:330-338`、`dialogs.py:445-452` | 下载使用 `threading.Thread` + 200ms `QTimer` 轮询 `is_alive()`，而同文件的 `_ConnectionTestThread`（124 行）与 `control_panel.py` 的 `_MLXTaskThread`/`_MLXHealthThread` 都用 `QThread` + 信号。轮询模式正是 N-P1-9 与 N-P2-10 的结构性根因。 |
| N-P3-11 | `dialogs.py:370-383` vs `config.yaml:32` | 首启向导写入 `"vad_threshold": 0.3`，`config.yaml` 的文档化默认值是 `0.5`。向导直接 `_save_settings(hardcoded_dict)`，绕过 `_apply_settings()`，因此持久化内容与面板正常产出的内容不同源。 |
| N-P3-12 | `dialogs.py:98-121` | `_StderrCapture` 只实现 `write`/`flush`/`isatty`，缺少 `fileno()`、`encoding`、`errors`。任何探测这些属性的库（含以 `stderr=sys.stderr` 派生子进程的代码）会抛 `AttributeError`。下载路径恰好会调用会派生子进程的第三方代码。 |
| N-P3-13 | `asr_server.py:42` | `@app.on_event("startup")` 自 FastAPI 0.93 起废弃，官方推荐 `lifespan`。升级 FastAPI 时会失效。 |
| N-P3-14 | `audio_capture_pyaudio.py:210-214` | `__del__` 调用 `self.stop()`，其中包含 `self._thread.join(timeout=3)` 与 PyAudio `terminate()`。解释器关闭期 `__del__` 的执行环境不可靠，用它做需要阻塞等待的资源回收是脆弱模式。 |
| N-P3-15 | `i18n.py:13`、`i18n.py:34` | `locale.getdefaultlocale()` 自 Python 3.11 废弃、计划于 3.15 移除（项目已使用 3.10+ 的 `X | None` 语法）。另外 `t()` 在 key 缺失时静默返回 key 本身，拼写错误会以原始标识符形式出现在 UI 上且无任何日志。 |

## 第二批已核对但未发现确定性问题的模块

以下模块本轮已逐函数核对，未发现有充分源码证据的缺陷，记录以界定覆盖范围：

- `torch_backend.py`：`normalize_device` 的 `cuda:N` 路径经 `asr_worker._parse_device`（`asr_worker.py:46-55`）在调用前已拆分出索引，CTranslate2 不会收到 `"cuda:0"`；`for_ct2` 语义正确。
- `platform_permissions.py`、`platform_app.py`、`platform_clickthrough.py`、`platform_config.py`、`platform_fonts.py`：异常分类与降级路径一致，未发现调用链错配。
- `connection_config.py`：URL 归一化与错误分类的分支已逐条核对。
- `audio_capture_base.py`：`_enqueue` 的丢最旧策略正确捕获 `Empty`/`Full`（正是 P1-4 应当复用的实现）。
- `ui_theme.py`、`log_window.py`、`asr_funasr.py`：规模小、无跨线程状态，未发现问题。
- `model_manager.py`：缓存路径探测、hub 映射与下载分支已核对；`_hf_repo_complete` 的断链检测逻辑正确。唯一观察是 `_ms_model_path`（`model_manager.py:369-372`）在存在多个 snapshot 时按字典序取 `snaps[-1]`，而 snapshot 目录名是 commit hash，字典序与时间序无关——但这只影响多快照场景下选中哪一个，不构成确定性缺陷，未列入编号。

## 第三批：UI、后端与依赖（16 项）

覆盖前两批未深入的部分：`subtitle_window.py`、`subtitle_settings.py`、`control_panel.py` 余下部分、`audio_capture.py` 读循环、ASR 后端实现、`requirements*.txt`、安装/构建脚本。

### P1

#### N-P1-13：选中 HY-MT 模型时，模型列表点击会同步阻塞 UI 约 2 秒

**位置**：`control_panel.py:695`（信号连接）· `control_panel.py:1455-1466`（`_update_mlx_controls`）

**证据**：

```python
# control_panel.py:695
self._model_list.currentRowChanged.connect(self._update_mlx_controls)

# control_panel.py:1455-1466 —— 全部在 Qt 主线程同步执行
def _update_mlx_controls(self, *_):
    model = self._selected_mlx_model()
    if model is None:
        return
    ready = self._mlx_manager.is_model_ready() and self._mlx_manager.is_environment_ready()
    running = self._mlx_manager.is_running() if ready else False
```

这两个调用的实际代价：

- `is_environment_ready()` → `_versions_are_compatible()`（`mlx_service.py:180`）**派生一个 Python 解释器子进程**做版本断言。
- `is_running()` → `_read_pid()` + `_pid_is_owned()`（**`ps` 子进程**）+ `_probe()`（**`urlopen(timeout=1.5)`**）。

**触发条件**：在模型列表中点选 HY-MT 模型（`_selected_mlx_model()` 返回非 None 时才走到这里）。

**影响**：每次点击冻结 UI 约 2 秒（1.5s HTTP 探测 + 两次进程派生）。项目在别处已经为同样的操作做了线程化处理——`_MLXHealthThread`（`control_panel.py:119`）存在的目的正是「不阻塞 Qt 事件循环」，但这条同步路径绕过了它。`_update_mlx_controls` 另有 4 个调用点（719、1446、1560、1579），代价相同。

**建议**：把 `is_environment_ready()`/`is_running()` 的结果缓存在 `_MLXHealthThread` 的回调里，`_update_mlx_controls` 只读缓存状态；或复用 `request_mlx_health_check()` 的既有线程化入口。

### P2

| 编号 | 位置 | 问题 |
| --- | --- | --- |
| N-P2-17 | `subtitle_window.py:359-368` | `split_text` 对每个字符位置测量**整个前缀**：`for i in range(1, len(text)+1): fm.horizontalAdvance(text[:i])`。`horizontalAdvance` 本身与前缀长度成正比，故单段换行是 O(n²) 字形测量，多段则更高。该函数经 `_rewrap()` 在每条字幕更新和每次 `resizeEvent` 时于 Qt 主线程执行。应改为二分查找或改用 `QTextLayout`。 |
| N-P2-18 | `control_panel.py:1241-1253`、`control_panel.py:1237-1239` | `_on_tab_changed` 每次切到缓存页都调 `_refresh_cache()`，后者无条件 `threading.Thread(target=_scan, daemon=True).start()`。无 `isRunning()` 去重（对比同文件 `request_mlx_health_check` 的 1583 行有该保护）。反复切换标签页会并发派生多个遍历多 GB 模型目录的扫描线程，全部向 `_cache_result` 发信号。 |
| N-P2-19 | `control_panel.py:1562-1572` | `closeEvent` 在 MLX 任务运行时执行 `task.cancel_event.set()` + `setEnabled(False)` + `event.ignore()`，等待 `_maybe_close_after_mlx_tasks` 放行。但取消标志只在子进程有 stdout 输出时才被检查（N-P2-7）。多 GB 模型下载的静默期内，**设置面板既无法关闭也无法操作**，用户只能强杀进程。这是 N-P2-7 的直接后果放大。 |
| N-P2-20 | `subtitle_settings.py:628-629` | `_emit_settings` 发射的是内部字典本体：`self.settings_changed.emit(self._settings)`，且 `s["lines"]` 与 `self._settings["lines"]` 是同一个 list 对象。接收方拿到的是与控件共享的可变状态，后续 `self._settings.update(s)` 会直接改变接收方持有的对象。对比 `ControlPanel._apply_settings`（`control_panel.py:1864`）发的是 `dict(self._current_settings)` 副本。 |
| N-P2-21 | `audio_capture.py:308-321` | `_read_loop` 的重启分支先 `self._restart_event.clear()` 再尝试 `_restart_stream()`；失败时只 `log.error` + `sleep(0.5)` + `continue`。事件已清除，因此**不会重试**，也不设置 `_metrics.last_error`、不停止采集。设备切换请求被静默丢弃，采集线程带着失败的流状态继续运行。与 P1-3（macOS 侧）同类，但发生在 Windows 路径上。 |
| N-P2-22 | `requirements.txt`、`requirements-mac.txt` vs `main.py:1701` | `main.py` 的 `_get_segmenter` 执行 `from yasbd import get_supported_langs, pysbd_adapter`，但**两个 requirements 文件都没有声明 `yasbd-lib`**，连注释都没有（对比 torch 在 `requirements.txt` 末尾有说明性注释）。5 个入口脚本（`install.ps1:283`、`install.sh:26`、`update.bat:73`、`update.sh:14`、`build_release.ps1:168`）都把它作为独立步骤安装，`tests/test_requirements.py` 也逐个断言了这一点——所以设计意图明确，但 requirements 不自足这件事没有任何记录。后果不只是「装不全」：`_split_sentences` 的这行 import 位于 `_do_interim_asr` 中**没有 try/except 覆盖**的代码段，ImportError 会直接杀死 ASR 线程。这是 P1-1 的一个具体触发器。 |
| N-P2-23 | `requirements-mac.txt:6` vs `requirements.txt` | `socksio>=1.0.0` 只在 macOS 依赖中声明。模型配置支持自定义代理 URL（`proxy` 字段可填任意 URL，见 `translator.make_openai_client`），Windows 用户填写 `socks5://...` 时 httpx 抛 `ImportError: Using SOCKS proxy, but the 'socksio' package is not installed`。`tests/test_requirements.py::test_mac_requirements_contain_every_cross_platform_dependency` 只校验 windows ⊆ mac，反方向不校验，因此这类不对称不会被测试发现。 |

### P3

| 编号 | 位置 | 问题 |
| --- | --- | --- |
| N-P3-16 | `subtitle_window.py:585-598` vs `subtitle_window.py:670-683` | `apply_settings` **内联复制**了 `_rebuild_text_widgets` 的 12 行（移除旧 widget、deleteLater、按 `lines` 重建、连 `height_changed`、加入布局），两个副本逐行相同。`_rebuild_text_widgets` 在 577 行确有调用，因此这不是死代码而是并存的两份实现——改一处必然发散。`apply_settings` 应直接调用 `_rebuild_text_widgets()`。 |
| N-P3-17 | `subtitle_window.py:879-891` | `_refresh_display` 取译文的四个分支里，只有 `elif lang and lang in tl_dict: texts.append(tl_dict[lang])` **不检查值是否为空**；另外三个分支都有 `if v`/`and tl_dict[""]` 判断。译文为空串时会进入 `" | ".join(texts)`，产生 `" | 正文"` 这样的前导分隔符。 |
| N-P3-18 | `control_panel.py:1742-1757` | `_on_ui_lang_changed` 的提示框硬编码中英双语字面量（`"Language changed. Please restart...\n语言已更改，请重启应用程序。"`），而该函数刚刚调用过 `set_lang(lang)`——`t()` 本可正常工作。同时它直接 `_save_settings(self._current_settings)`，绕过 `_apply_settings()`，与其余设置项统一走 `_auto_save()` 的做法不一致。与 N-P2-4、N-P3-8 同类。 |
| N-P3-19 | `subtitle_settings.py:631-633` | `get_settings()` 内部调用 `_emit_settings()`，因此这个 getter **会发射 `settings_changed` 信号**。读取操作触发一次完整的设置传播（在 `main.py` 中会一路走到 `subwin.apply_settings()` 的 widget 重建）。 |
| N-P3-20 | `audio_capture.py:368-378` | `self._stream.read(native_chunk, exception_on_overflow=False)` 在 `with self._lock:` 内阻塞执行，而 `_restart_stream()` 需要同一把锁。与 N-P1-4（`ASRClient` 单锁）同类：阻塞 IO 与生命周期操作共用一把锁。此处阻塞时长受 chunk 大小约束（32ms 量级），故影响远小于 ASR 侧。 |
| N-P3-21 | `requirements.txt:28` vs `requirements-mac.txt:38` | `transformers>=4.40.0`（Windows，**无上界**）vs `transformers==4.57.1`（macOS，钉死）。而 `mlx_service._versions_are_compatible`（`mlx_service.py:186`）断言 `int(m.version('transformers').split('.')[0]) < 5`。transformers 5.x 发布后，Windows 全新安装会解析到不受支持的大版本。 |
| N-P3-22 | `requirements-mac.txt:39-40` · `tests/test_requirements.py:81-82` | `pyannote-audio>=4.0,<5` 与 `torchcodec>=0.7` 在 macOS 依赖中声明，但**全代码库无任何 import**（`grep -rn "pyannote\|torchcodec" *.py` 为空）。`pyannote-audio` 会拉入 lightning 等重依赖。更麻烦的是 `test_requirements.py` 断言了它们必须存在，等于把未使用的重依赖固化进契约。若确为 GigaAM 的传递依赖，应写明理由；否则应同时移除依赖与断言。 |
| N-P3-23 | `tests/test_requirements.py:124-129` | `test_release_workflow_has_arm64_test_and_distinct_macos_artifact` 校验了 CI 中存在测试作业、runner 版本和 macOS 产物名，但**没有断言任何作业的 `needs:`**。N-P1-12（Windows `build` 作业缺少 `needs: test-macos-arm64`，测试失败仍发布）正是因此未被测试捕获。 |

### 对既有 P1-4 的补充证据

`audio_capture.py:434-438` 的实际代码比第一轮记录的更严重——**两个** `put_nowait` 都无保护：

```python
except queue.Full:
    self._metrics.dropped_blocks += 1
    self.audio_queue.get_nowait()                    # ← queue.Empty 未捕获
    self.audio_queue.put_nowait((audio, mic_rms))    # ← queue.Full 未捕获
```

对照 `audio_capture_base.py:119-127` 的正确实现，它把两者一并包在 `except (queue.Empty, queue.Full): return` 中。因此消费者在这个窗口内取走一项、或生产者在 get 与 put 之间被抢占导致队列重新填满，都会让未捕获异常逃逸到无顶层保护的 `_read_loop`，采集线程随之退出。

## 第三批已核对但未发现确定性问题的模块

- **ASR 后端结果契约一致**：`asr_engine.py:149`、`asr_sensevoice.py:214`、`asr_funasr_nano.py:120`、`asr_anime_whisper.py:99`、`asr_gigaam.py:290/304` 全部返回 `{"text", "language", "language_name"}` 或 `None`，Whisper 在 `word_timestamps` 时附加 `words`。A1 的统一结果契约可直接实现，无需先统一后端。
- `asr_gigaam.py:265-304`：内存输入失败后确实会落到 WAV 回退路径（`else:` 分支只在成功时 return）；`_write_wav`（248-262）正确 `os.close(fd)` 并在写失败时清理临时文件。
- `audio_capture.py:412-415`：无数据时有 `time.sleep(0.005)`，不存在空转。
- `benchmark.py:180`：`run_benchmark` 结尾确实 `threading.Thread(...).start()`，不阻塞 Qt 线程（`ThreadPoolExecutor(max_workers=len(models))` 在 `models` 为空时会抛 ValueError，但唯一调用方 `control_panel.py:1677` 已有 `if not models: return` 保护）。
- `subtitle_window.py:824-834`：`_pending_segment_timers` 不会无界增长——`_cancel_pending_segments()` 是 `_on_update_text` 的第一步，列表最多持有一项。
- `torch_backend.py` + `asr_worker.py:46-55`：`cuda:N` 在进入 `normalize_device` 之前已由 `_parse_device` 拆出索引，CTranslate2 不会收到 `"cuda:0"`。
- `install.ps1`、`install.sh`、`update.bat`、`update.sh`、`build_release.ps1`：ready-marker 的写入顺序均在 `pip check` 之后，与 `test_requirements.py` 的断言一致；未发现顺序错配。

## 上一轮问题复核结果

逐项核对 `CALL_CHAIN_AUDIT.md`，**13 项全部仍然存在**：

| 编号 | 复核位置 | 状态 |
| --- | --- | --- |
| P1-1 ASR 线程无顶层异常隔离 | `main.py:2049-2082` | 仍存在。`_asr_loop` 仅在 `_maybe_recycle_asr_worker` 外加了 try/except；`_process_segment` / `_process_interim_final` / `_do_interim_asr` 的调用无保护，`result["text"]`、`result["language"]` 仍为直接下标访问。 |
| P1-2 远程 ASR 吞异常成 `None` | `asr_remote.py:93-99` | 仍存在，代码逐字未变。 |
| P1-3 SCK `set_device` 失败不恢复 | `audio_capture_sck.py:527-537`、`main.py:413` | 仍存在。旁证：同一函数内 `set_mic_device` 的返回值在 `main.py:419-427` 被检查，`set_device` 未被检查，处理不一致。 |
| P1-4 Windows 队列 `get_nowait()` 竞态 | `audio_capture.py:434-438` | 仍存在。`audio_capture_base.py:117-127` 已有正确实现，未被复用。 |
| P1-5 删缓存退出绕过 `stop()` | `control_panel.py:1291`、`main.py:2240` | 仍存在。见本轮 N-P1-6（后果更严重）。 |
| P1-6 停止时 `put(None)` 可能永久阻塞 | `main.py:1536` | 仍存在。见本轮 N-P2-3（同一处的另一侧面）。 |
| P2-1 非流式路径绕过重复检测 | `translator.py:394-398` | 仍存在。补充：该路径同时跳过了 `_warn_if_thinking_burned()`，因此非流式模型的「thinking 烧光预算」诊断也完全失效。 |
| P2-2 删除非活动模型索引未调整 | `control_panel.py:1654-1666` | 仍存在，仅有 `if active >= len(models)` 一个分支。 |
| P2-3 Benchmark 未复用运行时请求参数 | `benchmark.py:65-123` | 仍存在。 |
| P2-4 `asr_server.py` ASGI 入口缺 `app.state.args` | `asr_server.py:127-137` | 仍存在。 |
| P2-5 字幕窗口最短显示时间丢句 | `subtitle_window.py:818-857` | 仍存在。 |
| P2-6 启动延迟回调覆盖用户暂停 | `main.py:2275`、`main.py:2761` | 仍存在。`_is_running = [True]` 初值与 `QTimer.singleShot(500, on_start)` 的组合未变。 |
| P3-1 FunASR-Nano 遮蔽原始错误 | `funasr_nano/model.py:379-394` | 仍存在。 |

---

## 建议修复顺序

1. **N-P1-1（VAD 竞争）+ N-P1-2（macOS 重启）** —— 两者都由 `_on_settings_changed` 触发，是日常使用中最高频的故障路径，且改动局部、风险低。
2. **N-P1-5 + P1-1 + P1-2** —— 统一 ASR/翻译的结果契约与失败可见性，消除「静默无输出」这一最难排查的故障类别。
3. **N-P1-4 + P1-6 + N-P2-3** —— 一并重做停止/切换协议：锁分离 + `threading.Event` 停止信号 + 有界等待。
4. **N-P1-6 + P1-5** —— 缓存删除必须先停 worker，并移出 Qt 线程。
5. **N-P1-3** —— 增量 ASR 短句去重，直接影响字幕正确性。
6. **N-P2-1 / N-P2-2 / N-P2-4 / N-P2-5 + P2-x** —— 计数、重试语义、i18n 与状态一致性。
7. **P3 与 N-P3-x** —— 质量与可维护性收尾。

## 审查范围与限制

- 覆盖：仓库内全部 Python 源码（`*.py`、`funasr_nano/`、`tests/`）、`config.yaml`、`user_settings.json`、安装与启动脚本。
- **未执行**：应用运行、单元测试、真实音频设备、真实模型加载、外部 API 调用。所有结论均基于源码证据，**未附运行时复现日志**。
- 环境限制：该 macOS 工作树中未安装 `ruff` / `pyflakes`，静态检查以 `compileall`（通过，无语法错误）加自建 AST 扫描器替代，机械性缺陷的覆盖度低于完整 linter。
- 本轮不重复列出 `CALL_CHAIN_AUDIT.md` 已记录的问题细节，仅给出复核结论与新增证据。
