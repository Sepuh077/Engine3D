"""Generate simple test .wav files for the audio example."""
import struct
import wave
import math
import os

SAMPLE_RATE = 44100
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sounds")


def _write_wav(path: str, samples: list[float], sample_rate: int = SAMPLE_RATE):
    """Write mono 16-bit PCM WAV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        for s in samples:
            clamped = max(-1.0, min(1.0, s))
            wf.writeframes(struct.pack("<h", int(clamped * 32767)))


def sine_tone(freq: float, duration: float, volume: float = 0.5) -> list[float]:
    n = int(SAMPLE_RATE * duration)
    return [volume * math.sin(2 * math.pi * freq * i / SAMPLE_RATE) for i in range(n)]


def square_tone(freq: float, duration: float, volume: float = 0.4) -> list[float]:
    n = int(SAMPLE_RATE * duration)
    period = SAMPLE_RATE / freq
    return [volume * (1.0 if (i % period) < period / 2 else -1.0) for i in range(n)]


def noise_burst(duration: float, volume: float = 0.3) -> list[float]:
    import random
    random.seed(42)
    n = int(SAMPLE_RATE * duration)
    samples = []
    for i in range(n):
        env = 1.0 - i / n  # fade out
        samples.append(volume * env * (random.random() * 2.0 - 1.0))
    return samples


def chirp(f_start: float, f_end: float, duration: float, volume: float = 0.5) -> list[float]:
    n = int(SAMPLE_RATE * duration)
    samples = []
    for i in range(n):
        t = i / SAMPLE_RATE
        freq = f_start + (f_end - f_start) * (t / duration)
        samples.append(volume * math.sin(2 * math.pi * freq * t))
    return samples


if __name__ == "__main__":
    files = {
        "tone_440hz.wav": sine_tone(440, 1.5),
        "tone_660hz.wav": sine_tone(660, 1.0, 0.4),
        "square_220hz.wav": square_tone(220, 1.0),
        "noise_burst.wav": noise_burst(0.8),
        "chirp_up.wav": chirp(200, 1200, 1.2),
    }
    for name, samples in files.items():
        path = os.path.join(OUTPUT_DIR, name)
        _write_wav(path, samples)
        print(f"  wrote {path} ({len(samples)} samples, {len(samples)/SAMPLE_RATE:.2f}s)")
    print("Done – all test sounds generated.")
