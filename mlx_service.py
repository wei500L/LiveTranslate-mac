"""Local MLX-LM service management for Apple Silicon translation models."""

from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from connection_config import normalize_api_base


log = logging.getLogger("LiveTranslate.MLX")

APP_DIR = Path(__file__).resolve().parent
MLX_ENV_DIR = APP_DIR / ".mlx-venv"
MLX_MODEL_DIR = APP_DIR / "models" / "hy-mt1.5-7b-mlx-4bit"
MLX_LOG_DIR = APP_DIR / "logs"
MLX_PID_FILE = MLX_LOG_DIR / "hy-mt1.5-7b-mlx.pid"
MLX_HOST = "127.0.0.1"
try:
    MLX_PORT = int(os.getenv("LIVETRANSLATE_MLX_PORT", "8080"))
except ValueError:
    MLX_PORT = 8080
if not 1 <= MLX_PORT <= 65535:
    MLX_PORT = 8080
MLX_BASE_URL = normalize_api_base(f"http://{MLX_HOST}:{MLX_PORT}")

HY_MT_MODEL_ID = "default_model"
HY_MT_MODEL_NAME = "HY-MT1.5-7B (MLX 4-bit)"
HY_MT_REPO = "Tencent-Hunyuan/HY-MT1.5-7B"
MLX_VERSION = "0.29.4"
MLX_LM_VERSION = "0.29.1"


class MLXServiceError(RuntimeError):
    """Raised when the managed local MLX service cannot be used."""


def hy_mt_model_config() -> dict[str, Any]:
    """Return the persisted LiveTranslate model entry for HY-MT."""
    return {
        "name": HY_MT_MODEL_NAME,
        "api_base": MLX_BASE_URL,
        "api_key": "local",
        "model": HY_MT_MODEL_ID,
        "proxy": "none",
        "streaming": True,
        "no_system_role": True,
        "thinking_style": "off",
        "context_turns": 2,
        "system_prompt": (
            "你是俄语课堂的实时翻译助手。请将课堂中的{source_lang}内容翻译成{target_lang}。\n"
            "场景：学校或大学课堂，内容可能包括教师讲解、学生提问、课堂讨论、例句、术语、板书和作业要求。\n"
            "规则：\n"
            "- 只输出一条准确、自然的{target_lang}译文，不要解释、分析、前缀、引号或多个候选。\n"
            "- 保持教师讲解或学生发言的逻辑、语气、否定、条件、因果、时间和指代关系，不擅自补充未说内容。\n"
            "- 课程术语、人名、地名、书名、课程名、缩写、数字、公式和符号使用目标语言通行表达；没有把握时保留原文，不要臆造。\n"
            "- 结合课堂语境和近期上下文纠正俄语 ASR 的错词、同音词和断句；无法确定时忠实翻译，不要编造。\n"
            "- 可适度压缩口语重复和填充词，但不要省略定义、例子、数字、公式、作业要求或关键限定。\n"
            "- 保持适合实时字幕的简洁长度；原句未完时翻译当前可确定的内容，不添加说明。\n"
            "近期课堂上下文：\n"
            "{context}"
        ),
        "overrides": {
            "temperature": 0.7,
            "top_p": 0.6,
            "max_tokens": 128,
        },
        "extra_body": {
            "top_k": 20,
            "repetition_penalty": 1.05,
        },
        "managed_service": {
            "type": "mlx_lm",
            "model_path": str(MLX_MODEL_DIR),
            "host": MLX_HOST,
            "port": MLX_PORT,
        },
    }


def is_hy_mt_model(model: dict[str, Any] | None) -> bool:
    service = (model or {}).get("managed_service") or {}
    return service.get("type") == "mlx_lm"


def ensure_hy_mt_model(settings: dict[str, Any] | None, activate_if_ready: bool = False) -> bool:
    """Add the HY-MT entry without deleting user models.

    Existing settings are intentionally preserved. If the model is already
    deployed, ``activate_if_ready`` can select it as the active model.
    """
    if not isinstance(settings, dict):
        return False
    models = settings.setdefault("models", [])
    target = next((m for m in models if is_hy_mt_model(m)), None)
    changed = False
    if target is None:
        models.append(hy_mt_model_config())
        target = models[-1]
        changed = True
    else:
        # Keep user edits, but backfill fields introduced by the managed preset.
        preset = hy_mt_model_config()
        for key, value in preset.items():
            if key not in target:
                target[key] = value
                changed = True
        # The local model is deliberately kept on a low-latency profile. These
        # controls are operational safeguards rather than user prompt content.
        for key in ("streaming", "no_system_role", "thinking_style", "context_turns"):
            if target.get(key) != preset[key]:
                target[key] = preset[key]
                changed = True
        overrides = target.setdefault("overrides", {})
        for key in ("temperature", "top_p", "max_tokens"):
            if overrides.get(key) != preset["overrides"][key]:
                overrides[key] = preset["overrides"][key]
                changed = True
        # The managed endpoint follows the active local service port. This
        # also migrates older settings when LIVETRANSLATE_MLX_PORT changes.
        if target.get("api_base") != preset["api_base"]:
            target["api_base"] = preset["api_base"]
            changed = True
        service = target.setdefault("managed_service", {})
        for key in ("host", "port"):
            if service.get(key) != preset["managed_service"][key]:
                service[key] = preset["managed_service"][key]
                changed = True
    if activate_if_ready and MLXServiceManager().is_model_ready():
        index = models.index(target)
        if settings.get("active_model") != index:
            settings["active_model"] = index
            changed = True
    return changed


class MLXServiceManager:
    """Start, probe, and stop the app-owned MLX-LM HTTP server."""

    def __init__(self, root: Path | None = None):
        self.root = Path(root or APP_DIR).resolve()
        self.env_dir = self.root / ".mlx-venv"
        self.model_dir = self.root / "models" / MLX_MODEL_DIR.name
        self.hf_cache_dir = self.root / ".hf-cache" / "hub"
        self.log_dir = self.root / "logs"
        self.pid_file = self.log_dir / MLX_PID_FILE.name
        self.process: subprocess.Popen | None = None
        # Set by the UI layer so user-visible strings get localized without this
        # module importing i18n. Keys pass through untranslated by default.
        self.translate = None
        self._version_cache: tuple[float, bool] | None = None

    def _text(self, key: str, **params) -> str:
        """Localize a message key, falling back to the key itself."""
        text = key
        if self.translate is not None:
            try:
                text = self.translate(key)
            except Exception:
                text = key
        if params:
            try:
                return text.format(**params)
            except (KeyError, IndexError, ValueError):
                return text
        return text

    @property
    def server_executable(self) -> Path:
        return self.env_dir / "bin" / "mlx_lm.server"

    @property
    def base_url(self) -> str:
        return MLX_BASE_URL

    def is_model_ready(self) -> bool:
        required = ("config.json", "tokenizer.json", "chat_template.jinja")
        if not self.model_dir.is_dir() or not all((self.model_dir / f).is_file() for f in required):
            return False
        return any(self.model_dir.glob("*.safetensors"))

    def is_supported_platform(self) -> bool:
        return sys.platform == "darwin" and os.uname().machine == "arm64"

    def _versions_are_compatible(self) -> bool:
        """Whether .mlx-venv holds the pinned package versions.

        Cached against the venv's mtime: this spawns a Python interpreter, and
        it used to run on the Qt thread on every model-list selection change.
        """
        python = self.env_dir / "bin" / "python"
        if not python.is_file():
            self._version_cache = None
            return False
        try:
            stamp = self.env_dir.stat().st_mtime
        except OSError:
            stamp = 0.0
        if self._version_cache is not None and self._version_cache[0] == stamp:
            return self._version_cache[1]
        result = self._probe_versions()
        self._version_cache = (stamp, result)
        return result

    def _probe_versions(self) -> bool:
        check = subprocess.run(
            [
                str(self.env_dir / "bin" / "python"),
                "-c",
                (
                    "import importlib.metadata as m; "
                    f"assert m.version('mlx-lm') == '{MLX_LM_VERSION}'; "
                    f"assert m.version('mlx') == '{MLX_VERSION}'; "
                    "assert int(m.version('transformers').split('.')[0]) < 5; "
                    "assert int(m.version('huggingface-hub').split('.')[0]) < 1"
                ),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return check.returncode == 0

    def is_environment_ready(self) -> bool:
        return (
            self.server_executable.is_file()
            and os.access(self.server_executable, os.X_OK)
            and self._versions_are_compatible()
        )

    @staticmethod
    def _notify(progress_callback, text: str) -> None:
        if progress_callback is not None:
            progress_callback(text)

    @staticmethod
    def _check_cancel(cancel_event: threading.Event | None) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise MLXServiceError("HY-MT model preparation was cancelled")

    def _run_logged(
        self,
        command: list[str],
        progress_callback=None,
        cancel_event: threading.Event | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        """Run a child process, streaming its output, cancellable at any time.

        Cancellation used to be checked only inside `for line in stdout`, so a
        multi-GB download that produced no output for minutes ignored the
        cancel button entirely. The reader now runs on its own thread while this
        one polls the event on a fixed period.
        """
        self._notify(progress_callback, "$ " + " ".join(command))
        process = subprocess.Popen(
            command,
            cwd=str(self.root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            # Own process group, so cancelling kills the whole tree (pip spawns
            # its own children) rather than just the parent.
            start_new_session=True,
        )

        def _pump():
            try:
                assert process.stdout is not None
                for line in process.stdout:
                    line = line.strip()
                    if line:
                        self._notify(progress_callback, line)
            except Exception:
                log.debug("Output pump ended early", exc_info=True)

        reader = threading.Thread(target=_pump, name="mlx-output", daemon=True)
        reader.start()
        try:
            while True:
                try:
                    code = process.wait(timeout=0.25)
                    break
                except subprocess.TimeoutExpired:
                    self._check_cancel(cancel_event)
        except BaseException:
            self._kill_process_tree(process)
            raise
        finally:
            reader.join(timeout=2)
        if code != 0:
            raise MLXServiceError(
                self._text(
                    "mlx_command_failed", code=code, command=" ".join(command)
                )
            )

    @staticmethod
    def _kill_process_tree(process: subprocess.Popen) -> None:
        """Terminate a child and its group; never let cleanup mask the cause."""
        if process.poll() is not None:
            return
        try:
            if hasattr(os, "killpg"):
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            else:
                process.terminate()
        except (OSError, AttributeError, ValueError):
            try:
                process.terminate()
            except OSError:
                return
        try:
            process.wait(timeout=10)
            return
        except subprocess.TimeoutExpired:
            log.warning("Child %s ignored SIGTERM; killing", process.pid)
        except OSError:
            return
        try:
            process.kill()
            process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            log.error("Child %s could not be killed", process.pid)

    def prepare_model(
        self,
        progress_callback=None,
        cancel_event: threading.Event | None = None,
    ) -> None:
        """Install MLX dependencies and prepare the 4-bit model in-app.

        BF16 weights are downloaded only into a temporary directory and are
        removed in ``finally`` after conversion succeeds or fails.
        """
        if not self.is_supported_platform():
            raise MLXServiceError(self._text("mlx_requires_apple_silicon"))
        # The server holds the model directory open; replacing it underneath a
        # running mlx_lm.server would delete files it is still reading.
        if self.is_running():
            self._notify(progress_callback, self._text("mlx_stopping_for_prepare"))
            self.stop()
            if self.is_running():
                raise MLXServiceError(self._text("mlx_stop_before_prepare"))
        models_dir = self.root / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        source_dir = models_dir / ".hy-mt1.5-7b-bf16.tmp"
        temp_model_dir = models_dir / f".{self.model_dir.name}.tmp"
        try:
            self._check_cancel(cancel_event)
            self._notify(progress_callback, self._text("mlx_checking_env"))
            if not self.is_environment_ready():
                if not self.env_dir.joinpath("bin", "python").is_file():
                    self._notify(progress_callback, self._text("mlx_creating_venv"))
                    self._run_logged(
                        [sys.executable, "-m", "venv", str(self.env_dir)],
                        progress_callback,
                        cancel_event,
                    )
                self._run_logged(
                    [
                        str(self.env_dir / "bin" / "python"),
                        "-m",
                        "pip",
                        "install",
                        f"mlx-lm=={MLX_LM_VERSION}",
                        f"mlx=={MLX_VERSION}",
                        "transformers<5",
                        "huggingface_hub<1",
                    ],
                    progress_callback,
                    cancel_event,
                )
                self._version_cache = None

            self._check_cancel(cancel_event)
            if self.is_model_ready():
                self._notify(progress_callback, self._text("mlx_model_already_ready"))
                return

            # modelscope goes into .mlx-venv, never into the venv the app itself
            # is running from: preparing a translation model must not mutate the
            # interpreter that is currently executing the GUI.
            mlx_python = str(self.env_dir / "bin" / "python")
            try:
                subprocess.run(
                    [mlx_python, "-c", "import modelscope"],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except (OSError, subprocess.CalledProcessError):
                self._run_logged(
                    [mlx_python, "-m", "pip", "install", "modelscope>=1.20.0"],
                    progress_callback,
                    cancel_event,
                )

            shutil.rmtree(source_dir, ignore_errors=True)
            shutil.rmtree(temp_model_dir, ignore_errors=True)
            source_dir.mkdir(parents=True, exist_ok=True)
            self._notify(
                progress_callback, self._text("mlx_downloading", repo=HY_MT_REPO)
            )
            download_script = (
                "import sys; from modelscope import snapshot_download; "
                "snapshot_download(model_id=sys.argv[2], local_dir=sys.argv[1])"
            )
            self._run_logged(
                [mlx_python, "-c", download_script, str(source_dir), HY_MT_REPO],
                progress_callback,
                cancel_event,
            )
            self._check_cancel(cancel_event)
            self._notify(progress_callback, self._text("mlx_converting"))
            self._run_logged(
                [
                    str(self.env_dir / "bin" / "mlx_lm.convert"),
                    "--hf-path",
                    str(source_dir),
                    "--mlx-path",
                    str(temp_model_dir),
                    "--quantize",
                    "--q-bits",
                    "4",
                    "--q-group-size",
                    "64",
                    "--q-mode",
                    "affine",
                    "--trust-remote-code",
                ],
                progress_callback,
                cancel_event,
            )
            tokenizer_config = source_dir / "tokenizer_config.json"
            if tokenizer_config.is_file():
                shutil.copy2(tokenizer_config, temp_model_dir / "tokenizer_config.json")
            # No ignore_errors here: a half-removed model directory that then
            # gets os.replace()d over is exactly how a corrupt install happens.
            if self.model_dir.exists():
                try:
                    shutil.rmtree(self.model_dir)
                except OSError as exc:
                    raise MLXServiceError(
                        self._text(
                            "mlx_replace_failed",
                            path=str(self.model_dir),
                            error=exc,
                        )
                    ) from exc
            os.replace(temp_model_dir, self.model_dir)
            self._notify(progress_callback, self._text("mlx_model_ready"))
        finally:
            shutil.rmtree(source_dir, ignore_errors=True)
            shutil.rmtree(temp_model_dir, ignore_errors=True)

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _probe(self) -> dict[str, Any] | None:
        request = Request(self._url("/models"), headers={"Accept": "application/json"})
        try:
            with urlopen(request, timeout=1.5) as response:
                if response.status != 200:
                    return None
                payload = json.loads(response.read().decode("utf-8"))
                if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
                    return None
                return payload
        except (OSError, ValueError, HTTPError, URLError):
            return None

    def is_running(self) -> bool:
        pid = self._read_pid()
        return bool(pid and self._pid_is_owned(pid) and self._probe() is not None)

    def _port_is_open(self) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            return sock.connect_ex((MLX_HOST, MLX_PORT)) == 0

    def _read_pid(self) -> int | None:
        try:
            pid = int(self.pid_file.read_text(encoding="ascii").strip())
            return pid if pid > 0 else None
        except (OSError, ValueError):
            return None

    def _pid_is_alive(self, pid: int | None) -> bool:
        if not pid:
            return False
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def _pid_is_owned(self, pid: int | None) -> bool:
        """Verify a persisted PID still belongs to our MLX command."""
        if not self._pid_is_alive(pid):
            return False
        if self.process is not None and self.process.pid == pid:
            return True
        try:
            output = subprocess.check_output(
                ["ps", "-p", str(pid), "-o", "command="],
                text=True,
                stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return "mlx_lm.server" in output and str(self.model_dir) in output

    def _cleanup_stale_pid(self) -> None:
        pid = self._read_pid()
        if pid and not self._pid_is_alive(pid):
            self.pid_file.unlink(missing_ok=True)

    def ensure_running(
        self, timeout: float = 120.0, progress_callback=None,
        cancel_event: threading.Event | None = None,
    ) -> None:
        if self.is_running():
            return
        self._cleanup_stale_pid()
        if not self.is_model_ready():
            raise MLXServiceError(
                self._text("mlx_model_not_deployed", path=str(self.model_dir))
            )
        if not self.is_environment_ready():
            raise MLXServiceError(
                self._text("mlx_env_not_installed", path=str(self.env_dir))
            )
        if self._port_is_open():
            raise MLXServiceError(self._text("mlx_port_occupied", port=MLX_PORT))

        self.log_dir.mkdir(parents=True, exist_ok=True)
        # mlx-lm's /v1/models handler scans the Hugging Face cache even for a
        # local model. Give it a deterministic, project-local cache directory
        # so a fresh install does not fail with CacheNotFound.
        self.hf_cache_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / "hy-mt1.5-7b-mlx.log"
        command = [
            str(self.server_executable),
            "--model",
            str(self.model_dir),
            "--host",
            MLX_HOST,
            "--port",
            str(MLX_PORT),
            "--log-level",
            "INFO",
            "--temp",
            "0.7",
            "--top-p",
            "0.6",
            "--top-k",
            "20",
        ]
        log.info("Starting managed MLX service: %s", " ".join(command))
        output = log_path.open("ab")
        environment = os.environ.copy()
        environment["HF_HUB_CACHE"] = str(self.hf_cache_dir)
        try:
            self.process = subprocess.Popen(
                command,
                cwd=str(self.root),
                env=environment,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            output.close()
        self.pid_file.write_text(str(self.process.pid), encoding="ascii")

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                self._check_cancel(cancel_event)
            except MLXServiceError:
                # Cancelling the wait must also reap the server we just spawned,
                # exactly as the timeout branch below does — otherwise the
                # process and its pid file outlive the cancelled task.
                self.stop()
                raise
            if progress_callback is not None:
                progress_callback(self._text("mlx_waiting_for_model"))
            if self._probe() is not None:
                log.info("Managed MLX service is ready on %s", self.base_url)
                return
            if self.process.poll() is not None:
                tail = ""
                try:
                    tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
                except OSError:
                    pass
                self.pid_file.unlink(missing_ok=True)
                raise MLXServiceError(
                    self._text(
                        "mlx_service_exited", code=self.process.returncode
                    )
                    + f"\n{tail}"
                )
            time.sleep(0.5)
        self.stop()
        raise MLXServiceError(self._text("mlx_start_timeout", url=self.base_url))

    def stop(self) -> None:
        pid = self.process.pid if self.process and self.process.poll() is None else self._read_pid()
        if not pid:
            self.pid_file.unlink(missing_ok=True)
            return
        if not self._pid_is_owned(pid):
            log.warning("Refusing to stop non-owned process recorded in %s", self.pid_file)
            self.pid_file.unlink(missing_ok=True)
            self.process = None
            return
        log.info("Stopping managed MLX service (pid=%s)", pid)
        # os.killpg and SIGKILL do not exist on Windows, and this runs from
        # aboutToQuit on every platform: an AttributeError here would escape
        # into Qt's shutdown rather than falling back to terminate().
        self._signal_pid(pid, "SIGTERM")
        deadline = time.monotonic() + 5
        while self._pid_is_alive(pid) and time.monotonic() < deadline:
            time.sleep(0.1)
        if self._pid_is_alive(pid):
            self._signal_pid(pid, "SIGKILL")
        self.pid_file.unlink(missing_ok=True)
        self.process = None

    def _signal_pid(self, pid: int, name: str) -> None:
        """Signal a process group, falling back to the process, then to Popen."""
        sig = getattr(signal, name, None) or getattr(signal, "SIGTERM")
        for send in (
            lambda: os.killpg(pid, sig),
            lambda: os.kill(pid, sig),
        ):
            try:
                send()
                return
            except ProcessLookupError:
                return
            except (OSError, AttributeError):
                continue
        if self.process is not None and self.process.pid == pid:
            try:
                if name == "SIGKILL":
                    self.process.kill()
                else:
                    self.process.terminate()
            except OSError:
                pass
