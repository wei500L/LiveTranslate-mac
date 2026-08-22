"""Offline audio smoke test with an optional Windows loopback diagnostic.

The default command never opens a real device and is suitable for CI. Use
``python test_audio.py --live-windows`` only on Windows when diagnosing a
WASAPI loopback installation.
"""

import argparse


def offline_smoke() -> None:
    import numpy as np
    from audio_capture_base import AudioCaptureBase

    capture = AudioCaptureBase(queue_size=2)
    # 1536 native samples at 48 kHz become exactly 512 samples at 16 kHz.
    capture.push_audio(np.ones((1536, 2), dtype=np.float32), native_channels=2, native_rate=48000)
    item = capture.get_audio(timeout=0)
    if item is None or item[0].shape != (512,) or item[0].dtype != np.float32:
        raise RuntimeError("offline audio normalization did not produce a 512-sample float32 block")
    capture.stop()
    print("offline audio smoke: OK (16 kHz mono, 512 samples)")


def windows_loopback_diagnostic():
    import numpy as np
    import pyaudiowpatch as pyaudio

    pa = pyaudio.PyAudio()
    try:
        wasapi = None
        for i in range(pa.get_host_api_count()):
            info = pa.get_host_api_info_by_index(i)
            if "WASAPI" in info["name"]:
                wasapi = info
                break
        if wasapi is None:
            raise RuntimeError("WASAPI host API not found")
        print(f"WASAPI: {wasapi['name']}")

        loopback = None
        for i in range(pa.get_device_count()):
            dev = pa.get_device_info_by_index(i)
            if dev.get("isLoopbackDevice", False):
                loopback = dev
                print(
                    f"  Loopback: [{i}] {dev['name']} "
                    f"ch={dev['maxInputChannels']} rate={dev['defaultSampleRate']}"
                )
        if loopback is None:
            raise RuntimeError("No loopback device found")

        channels = loopback["maxInputChannels"]
        rate = int(loopback["defaultSampleRate"])
        chunk = int(rate * 0.5)
        print(f"\nCapturing from: {loopback['name']}")
        print(f"Config: {rate}Hz, {channels}ch, chunk={chunk}")
        stream = pa.open(
            format=pyaudio.paFloat32,
            channels=channels,
            rate=rate,
            input=True,
            input_device_index=loopback["index"],
            frames_per_buffer=chunk,
        )
        try:
            print("\nReading 6 chunks (3 seconds)...")
            for i in range(6):
                data = stream.read(chunk, exception_on_overflow=False)
                audio = np.frombuffer(data, dtype=np.float32)
                mono = audio.reshape(-1, channels).mean(axis=1) if channels > 1 else audio
                rms = np.sqrt(np.mean(mono**2))
                print(
                    f"  Chunk {i}: samples={len(mono)}, rms={rms:.6f}, "
                    f"max={np.abs(mono).max():.6f}"
                )
        finally:
            stream.stop_stream()
            stream.close()
    finally:
        pa.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-windows", action="store_true", help="probe a real Windows WASAPI loopback device")
    args = parser.parse_args()
    if args.live_windows:
        windows_loopback_diagnostic()
    else:
        offline_smoke()
