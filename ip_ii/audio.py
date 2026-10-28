"""Load and save 16-bit PCM WAV recordings.

The rest of the project works with float64 samples scaled to [-1, 1).
No filtering, click detection or other signal modification occurs here.
"""

from pathlib import Path

import numpy as np
from scipy.io import wavfile


PCM16_SCALE = 2**15  # 32768: range of int16 is -32768 ... +32767.
PCM16_MAX_NORMALIZED = (PCM16_SCALE - 1) / PCM16_SCALE


def pcm16_to_float(pcm: np.ndarray) -> np.ndarray:
    """Convert signed 16-bit PCM samples to float64 in [-1, 1).

    Mono audio has shape (frames,); multichannel audio (frames, channels).
    The returned values are exact multiples of 1 / 32768.
    """
    if pcm.dtype != np.dtype("int16"):
        raise TypeError(f"Expected int16 PCM samples, got {pcm.dtype}.")
    if pcm.ndim not in (1, 2) or 0 in pcm.shape:
        raise ValueError("Audio must contain frames, with shape (N,) or (N, channels).")
    return pcm.astype(np.float64) / PCM16_SCALE


def float_to_pcm16(samples: np.ndarray) -> np.ndarray:
    """Convert normalized floating-point audio to signed 16-bit PCM.

    Values beyond the representable PCM16 range are clipped to prevent
    integer overflow; values between PCM16 levels are rounded to the
    nearest integer. An unmodified pcm16_to_float() output round-trips
    without losing any samples.
    """
    values = np.asarray(samples)
    if not np.issubdtype(values.dtype, np.floating):
        raise TypeError("Normalized audio must have a floating-point dtype.")
    if values.ndim not in (1, 2) or 0 in values.shape:
        raise ValueError("Audio must contain frames, with shape (N,) or (N, channels).")
    if not np.all(np.isfinite(values)):
        raise ValueError("Audio samples must not contain NaN or infinity.")

    limited = np.clip(values, -1.0, PCM16_MAX_NORMALIZED)
    return np.rint(limited * PCM16_SCALE).astype(np.int16)


def read_wav(
    path: str | Path,
    *,
    expected_sample_rate: int = 22_050,
) -> tuple[int, np.ndarray]:
    """Read a 16-bit PCM WAV and return (sample_rate, float64 samples).

    The project assignment specifies 22,050 samples/second. A different
    sample rate is rejected, not silently resampled.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Input WAV does not exist: {source}")

    sample_rate, pcm = wavfile.read(source)
    if pcm.dtype != np.dtype("int16"):
        raise ValueError(
            f"Unsupported WAV sample format ({pcm.dtype}) in {source}. "
            "This project expects signed 16-bit PCM WAV."
        )
    if sample_rate != expected_sample_rate:
        raise ValueError(
            f"Unsupported sampling rate ({sample_rate} Hz) in {source}. "
            f"Expected {expected_sample_rate} Hz."
        )

    return int(sample_rate), pcm16_to_float(pcm)


def write_wav(path: str | Path, sample_rate: int, samples: np.ndarray) -> None:
    """Write normalized floating-point audio as a signed 16-bit PCM WAV.

    A mono signal stays mono; for a 2D signal, channels are preserved.
    """
    if sample_rate <= 0:
        raise ValueError("Sampling rate must be positive.")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(destination, sample_rate, float_to_pcm16(samples))
