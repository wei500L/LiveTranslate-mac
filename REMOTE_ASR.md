# Remote Whisper ASR

Run speech recognition on a **separate GPU machine** and have LiveTranslate talk
to it over HTTP. Useful when the PC running LiveTranslate has no NVIDIA GPU
(CPU-only faster-whisper is too slow for real-time) but another machine on the
LAN does.

```
LiveTranslate (this PC) ──HTTP──> asr_server.py (GPU machine) ──> faster-whisper / CUDA
      RemoteASREngine              /transcribe, /health
```

## 1. On the GPU machine — run the server

Requires Python 3.10+ and an NVIDIA GPU.

```bash
pip install faster-whisper fastapi uvicorn numpy

python asr_server.py --host 0.0.0.0 --port 8765 \
    --model large-v3 --device cuda --compute-type float16
```

`--host` **defaults to `127.0.0.1`**, so the server is unreachable from other
machines until you pass `--host 0.0.0.0` as above. That is deliberate: the
endpoint runs GPU inference with no authentication, and a default of `0.0.0.0`
exposed it to the whole network the moment anyone ran it.

`--model` accepts any faster-whisper size: `tiny`, `base`, `small`, `medium`,
`large-v3`. On an 8 GB card, `large-v3` (float16) uses ~4 GB and is the most
accurate; `medium` is a lighter option. The model downloads from Hugging Face on
first run.

When the log shows `Uvicorn running on http://0.0.0.0:8765`, it's ready:

```bash
curl http://localhost:8765/health      # {"status":"ok","model":"large-v3"}
```

### Two ways to start it

| Command | Where configuration comes from |
|---|---|
| `python asr_server.py --model large-v3 ...` | Command-line flags, falling back to the `LIVETRANSLATE_ASR_*` environment variables, then to the built-in defaults. |
| `uvicorn asr_server:app` | No flags are parsed: the `LIVETRANSLATE_ASR_*` environment variables, then the built-in defaults. |

Both entry points read the same `default_config()`, and `/health` reports the
configuration that is actually in effect. (`uvicorn asr_server:app` previously
crashed on startup because `app.state.args` only existed under `__main__`.)

Environment variables: `LIVETRANSLATE_ASR_HOST`, `_PORT`, `_MODEL`, `_DEVICE`,
`_COMPUTE_TYPE`, `_TOKEN`.

### Security

The server has **no built-in authentication** and should only be deployed on a
trusted network. Two guards are available:

- **Bind address** — stays on `127.0.0.1` unless you opt into a wider one.
- **Shared secret** — set `LIVETRANSLATE_ASR_TOKEN` (or `--token`) and every
  `/transcribe` request must carry a matching `X-ASR-Token` header, else 401.

`/transcribe` also rejects bodies larger than about 5 minutes of 16 kHz float32
audio (~19 MB) with 413, instead of reading an arbitrary upload into memory.

### CUDA libraries

CTranslate2 (faster-whisper's backend) needs the CUDA 12 cuBLAS + cuDNN 9
libraries on the library path. If you hit `Library libcublas.so.12 is not found`:

```bash
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
export LD_LIBRARY_PATH=`python3 -c 'import os, nvidia.cublas.lib, nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))'`
```

(or use a system CUDA toolkit install).

### Slow model download?

Set a Hugging Face mirror before launching:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### Run as a service (optional, Linux/systemd)

So the server starts on boot and restarts on failure:

```ini
# /etc/systemd/system/asr.service
[Unit]
Description=LiveTranslate Remote ASR Server
After=network-online.target

[Service]
User=youruser
WorkingDirectory=/home/youruser
Environment=HF_ENDPOINT=https://hf-mirror.com
ExecStart=/usr/bin/python3 -u /home/youruser/asr_server.py --host 0.0.0.0 --port 8765 --model large-v3 --device cuda --compute-type float16
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now asr.service
journalctl -u asr.service -f          # follow logs
```

## 2. In LiveTranslate — point at the server

1. Open **Settings → VAD / ASR**.
2. Set **ASR engine** to **Remote Whisper (remote GPU server)**.
3. Enter the address in **Remote ASR Server URL**, e.g. `http://192.168.1.10:8765`.

Recognition now runs on the GPU machine; no local ASR model download is needed.
Translation still uses whatever model is configured in the Translation tab.

## HTTP API

| Method | Path | Body | Response |
|--------|------|------|----------|
| `POST` | `/transcribe` | `[uint32 lang_len][lang bytes][float32 PCM 16 kHz mono]` | `{"text", "language", "elapsed"}` |
| `GET`  | `/health` | — | `{"status": "ok", "model": "..."}` |

`lang_len` + `lang` is an optional language hint (e.g. `de`); send length `0` (or
`auto`) to let Whisper auto-detect.
