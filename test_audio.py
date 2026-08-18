"""Manual Windows WASAPI loopback diagnostic.

This is intentionally not executed during pytest collection. Use
``python test_audio.py`` on a Windows machine with a loopback device.
"""


def main():
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
    main()
