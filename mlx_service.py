"""Local MLX-LM service management for Apple Silicon translation models."""

from __future__ import annotations

import json
import logging
import os
import re
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
_MLX_PORT_ENV = os.getenv("LIVETRANSLATE_MLX_PORT")
try:
    MLX_PORT = int(_MLX_PORT_ENV) if _MLX_PORT_ENV else 8080
except ValueError:
    MLX_PORT = 8080
if not 1 <= MLX_PORT <= 65535:
    MLX_PORT = 8080
MLX_BASE_URL = normalize_api_base(f"http://{MLX_HOST}:{MLX_PORT}")
# An explicitly exported LIVETRANSLATE_MLX_PORT pins the port: settings are
# force-migrated to it. Without it the persisted managed_service.port is
# authoritative, because the manager can move the port at runtime (see
# MLX_PORT_SCAN_ATTEMPTS) and that choice must survive restarts.
MLX_PORT_FROM_ENV = bool(_MLX_PORT_ENV)
# How many ports to try upward when the configured one is occupied by a
# service this app does not own. mlx_lm.server serves whatever model it
# loaded and ignores the request's model field, so binding onto -- or
# connecting to -- someone else's endpoint would silently run the wrong
# model. Moving to a free port lets both services coexist.
MLX_PORT_SCAN_ATTEMPTS = 20

HY_MT_MODEL_ID = "default_model"
HY_MT_MODEL_NAME = "HY-MT1.5-7B (MLX 4-bit)"
HY_MT_REPO = "Tencent-Hunyuan/HY-MT1.5-7B"
MLX_VERSION = "0.29.4"
MLX_LM_VERSION = "0.29.1"
# mlx_lm.server does not handle SIGTERM: a direct SIGTERM leaves it running past
# 20s (measured). Every quit therefore burned the whole grace period before
# escalating. Keep a short one in case a future build starts handling it, then
# escalate — the server holds no state worth a graceful shutdown, just a
# read-only model in memory.
MLX_STOP_GRACE_SECONDS = 1.5


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
        # HY-MT is a translation-specialized model, not an instruction-following
        # chat model: given a multi-line instruction block it continues the block
        # instead of translating it, burning the whole max_tokens budget. Measured
        # on this exact 4-bit build: the long classroom prompt failed 3/3 (~2.5s,
        # 128 tokens of regurgitated prompt), a one-line directive succeeded 4/4
        # (~0.3s, ~9 tokens).
        #
        # context_turns must stay 0 for the same reason. With no_system_role the
        # only way context reaches this model is the {context} placeholder, and a
        # context block re-triggers the runaway (0/3 with it, 3/3 without).
        # Two turns of real conversation context. Measured on this build: it
        # resolves pronouns the isolated sentence cannot ("find it" -> "find the
        # derivative of the function"), stays 8/8 stable across a lecture, and
        # costs ~60 extra prompt tokens. This works only as *turns* — the same
        # history pasted into the prompt as a text block makes the model
        # continue the block instead of translating (0/3).
        "context_turns": 2,
        # English, matching both HY-MT's documented prompt format and the
        # English language names LANGUAGE_DISPLAY substitutes in.
        "system_prompt": (
            "Translate the following university classroom {source_lang} into "
            "{target_lang}. Keep terminology, numbers and formulas accurate. "
            "Output only the translation.\n\n"
        ),
        # Greedy decoding. Translation wants the single best rendering, not a
        # sample: measured 4/4 identical outputs for a repeated sentence versus
        # 2/4 at temperature 0.7, with equal quality and lower latency (0.39s vs
        # 0.49s average). A subtitle for the same sentence should not change
        # between one utterance and the next.
        "overrides": {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 128,
        },
        # Deliberately empty. The runaway guard this field used to carry
        # (repetition_penalty 1.05) disqualifies every request from
        # mlx_lm.server's batch path — _is_batchable() rejects
        # repetition_penalty != 0 — so concurrent translations serialized on
        # the server's single generation thread and the app's translation
        # worker pool queued behind each other. Measured on this build (4
        # concurrent requests): last completion 2.2s serialized versus 1.6s
        # batched, all four streaming from ~0.8s. The loop guard now lives
        # app-side: Translator._check_repetition raises RepetitionError (a
        # visible warning per sentence) and max_tokens=128 bounds the worst
        # case.
        "extra_body": {},
        "managed_service": {
            "type": "mlx_lm",
            "model_path": str(MLX_MODEL_DIR),
            "host": MLX_HOST,
            "port": MLX_PORT,
        },
    }


# Prompts this project has shipped for the managed model and has since replaced.
# A config still carrying one verbatim is migrated to the current preset; an
# edited prompt belongs to the user and is left alone. Each entry is the set of
# markers identifying one generation of the prompt.
_SUPERSEDED_HY_MT_PROMPTS = (
    # The original long classroom prompt. This model continues a multi-line
    # instruction block instead of following it (3/3 failures when measured).
    ("你是俄语课堂的实时翻译助手", "近期课堂上下文"),
    # A one-line Chinese directive; correct, but it substituted the English
    # language names LANGUAGE_DISPLAY provides into a Chinese sentence.
    ("把大学课堂上的{source_lang}内容翻译成{target_lang}",),
)


def _is_superseded_hy_mt_prompt(prompt: Any) -> bool:
    if not isinstance(prompt, str):
        return False
    return any(
        all(marker in prompt for marker in markers)
        for markers in _SUPERSEDED_HY_MT_PROMPTS
    )


def is_hy_mt_model(model: dict[str, Any] | None) -> bool:
    service = (model or {}).get("managed_service") or {}
    return service.get("type") == "mlx_lm"


def _valid_port(value: Any) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 1 <= value <= 65535
    )


def mlx_base_url_for(port: int) -> str:
    return normalize_api_base(f"http://{MLX_HOST}:{port}")


def managed_port_for(settings: dict[str, Any] | None) -> int:
    """The port the HY-MT entry expects its local service on.

    Falls back to the module default when there is no entry or no valid
    persisted port, so callers always get a port they can hand to the
    manager.
    """
    if isinstance(settings, dict):
        for model in settings.get("models") or []:
            if is_hy_mt_model(model):
                port = (model.get("managed_service") or {}).get("port")
                if _valid_port(port):
                    return port
                break
    return MLX_PORT


def sync_managed_endpoint(entry: dict[str, Any], port: int) -> bool:
    """Point a HY-MT model entry at the port its local service runs on.

    Pure dict surgery, so the panel can call it from whichever hook learned
    the real port (task success, health probe) and handle saving and signal
    emission itself. Returns whether anything changed.
    """
    if not _valid_port(port) or not is_hy_mt_model(entry):
        return False
    service = entry.setdefault("managed_service", {})
    if (
        service.get("port") == port
        and entry.get("api_base") == mlx_base_url_for(port)
    ):
        return False
    service["host"] = MLX_HOST
    service["port"] = port
    entry["api_base"] = mlx_base_url_for(port)
    return True


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
        # Replace the previous preset prompt, which this model cannot follow.
        # Only when it is still verbatim the one we shipped — a prompt the user
        # has edited is theirs to keep.
        if _is_superseded_hy_mt_prompt(target.get("system_prompt")):
            target["system_prompt"] = preset["system_prompt"]
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
        # extra_body is operational too, and replaced wholesale so a key the
        # preset has dropped goes away (top_k was meaningless once decoding
        # became greedy) rather than lingering forever.
        if target.get("extra_body") != preset["extra_body"]:
            target["extra_body"] = dict(preset["extra_body"])
            changed = True
        # The managed endpoint follows the local service port. With an
        # explicit LIVETRANSLATE_MLX_PORT the preset wins (that is how
        # changing the env var migrates older settings); otherwise the
        # entry's own persisted port wins, because the manager may have
        # moved it at runtime to dodge an occupied port and that choice
        # must survive restarts.
        service = target.setdefault("managed_service", {})
        if MLX_PORT_FROM_ENV:
            port = preset["managed_service"]["port"]
        else:
            port = service.get("port")
            if not _valid_port(port):
                port = preset["managed_service"]["port"]
        base = mlx_base_url_for(port)
        if target.get("api_base") != base:
            target["api_base"] = base
            changed = True
        if service.get("port") != port:
            service["port"] = port
            changed = True
        if service.get("host") != MLX_HOST:
            service["host"] = MLX_HOST
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
        # The port the service is expected on / was last started with. The
        # panel re-points it at the persisted managed_service.port, and
        # ensure_running may move it to dodge an occupied port (the panel
        # then persists the new value, so the next start goes straight to
        # it).
        self.port = MLX_PORT
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
        return mlx_base_url_for(self.port)

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

    def _probe(self, port: int | None = None) -> dict[str, Any] | None:
        base = mlx_base_url_for(self.port if port is None else port)
        request = Request(f"{base}/models", headers={"Accept": "application/json"})
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
        if not (pid and self._pid_is_owned(pid)):
            return False
        # The owned process's command line is the only reliable statement of
        # where our server actually listens: /v1/models cannot identify the
        # model (mlx_lm scans the HF cache instead of reporting the loaded
        # model), and the persisted port can drift -- re-selecting the preset
        # in the model dialog resets the entry while the service keeps
        # running on a port chosen earlier to dodge a conflict. Adopting the
        # real port keeps the monitor, the settings and the translator
        # pointing at the server that exists.
        port = self._owned_process_port(pid) or self.port
        if self._probe(port) is None:
            return False
        self.port = port
        return True

    def _owned_process_port(self, pid: int) -> int | None:
        """The --port argument of an owned server process, if readable."""
        try:
            output = subprocess.check_output(
                ["ps", "-p", str(pid), "-o", "command="],
                text=True,
                stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        match = re.search(r"--port[= ](\d{1,5})", output)
        if match and _valid_port(int(match.group(1))):
            return int(match.group(1))
        return None

    def _port_is_open(self, port: int | None = None) -> bool:
        probe_port = self.port if port is None else port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            return sock.connect_ex((MLX_HOST, probe_port)) == 0

    def _next_free_port(self) -> int | None:
        """The first free port above the configured one, or None."""
        for candidate in range(
            self.port + 1, self.port + 1 + MLX_PORT_SCAN_ATTEMPTS
        ):
            if not self._port_is_open(candidate):
                return candidate
        return None

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
            # Occupied by a service we do not own: killing it was never an
            # option, and connecting to it would silently run the wrong
            # model (mlx_lm serves whatever it loaded). Move to the next
            # free port; the caller persists the choice so the next start
            # does not have to move again.
            free = self._next_free_port()
            if free is None:
                raise MLXServiceError(
                    self._text(
                        "mlx_port_occupied",
                        port=self.port,
                        attempts=MLX_PORT_SCAN_ATTEMPTS,
                    )
                )
            log.info(
                "Port %s is occupied by a service this app does not own; "
                "starting the managed service on port %s instead",
                self.port,
                free,
            )
            self.port = free

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
            str(self.port),
            "--log-level",
            "INFO",
            # Fallbacks for a request that omits them; the app always sends its
            # own. Kept aligned with the preset so both paths behave the same.
            "--temp",
            "0.0",
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
        if not self._await_exit(pid, MLX_STOP_GRACE_SECONDS):
            log.info("MLX service ignored SIGTERM; killing pid=%s", pid)
            self._signal_pid(pid, "SIGKILL")
            if not self._await_exit(pid, 2.0):
                log.error("MLX service pid=%s could not be killed", pid)
        self.pid_file.unlink(missing_ok=True)
        self.process = None

    def _await_exit(self, pid: int, timeout: float) -> bool:
        """Wait for the service to die, reaping it when it is our own child.

        os.kill(pid, 0) succeeds for a zombie, so polling alone reported a
        killed-but-unreaped child as still alive and logged a false failure.
        """
        if self.process is not None and self.process.pid == pid:
            try:
                self.process.wait(timeout=timeout)
                return True
            except subprocess.TimeoutExpired:
                return False
            except OSError:
                return True
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self._pid_is_alive(pid):
                return True
            # A killed child stays a zombie until someone waits on it, and
            # os.kill(pid, 0) keeps succeeding until then. If it is a child of
            # this process, reaping it settles the question.
            try:
                reaped, _ = os.waitpid(pid, os.WNOHANG)
                if reaped == pid:
                    return True
            except (ChildProcessError, OSError):
                pass
            time.sleep(0.05)
        return not self._pid_is_alive(pid)

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
