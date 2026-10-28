"""Linear reconstruction of short runs of suspected damaged audio samples.

Project 1 (proj_1_pl.pdf, point c) requests interpolation from *valid*
neighboring samples and assumes no more than four consecutively corrupted
samples. For a run [start, end] of length m, the linear extension is

    restored[start - 1 + j] = left + j / (m + 1) * (right - left),
    j = 1, ..., m,

where left = signal[start - 1] and right = signal[end + 1].

This module does NOT classify samples or change EW-LS state. It operates on
one already-decoded, one-dimensional audio channel (floating-point samples).
The input array is never modified. Integration with the detector is step 10.
"""

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ip_ii.bursts import ImpulseBurst


@dataclass(frozen=True)
class SkippedBurst:
    """A candidate run deliberately left unchanged, with an explicit reason."""

    start_index: int
    end_index: int
    reason: str


@dataclass(frozen=True)
class InterpolationResult:
    """Restored signal and an auditable record of interpolation decisions."""

    samples: np.ndarray
    reconstructed_bursts: tuple[ImpulseBurst, ...]
    skipped_bursts: tuple[SkippedBurst, ...]

    @property
    def reconstructed_sample_count(self) -> int:
        """Number of individual samples actually replaced by interpolation."""
        return sum(burst.length for burst in self.reconstructed_bursts)


def _copy_validated_samples(samples: np.ndarray) -> np.ndarray:
    """Validate a single floating-point channel and copy to float64."""
    values = np.asarray(samples)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Signal must be a nonempty one-dimensional audio channel.")
    if not np.issubdtype(values.dtype, np.floating):
        raise TypeError("Signal must contain floating-point audio samples (as from read_wav).")
    if not np.all(np.isfinite(values)):
        raise ValueError("Signal samples must all be finite.")
    return np.array(values, dtype=np.float64, copy=True)


def _validated_valid_mask(valid_mask: np.ndarray | None, length: int) -> np.ndarray:
    """Return a mask of eligible anchors (True means independently trusted)."""
    if valid_mask is None:
        return np.ones(length, dtype=np.bool_)
    mask = np.asarray(valid_mask)
    if mask.ndim != 1 or mask.shape != (length,) or mask.dtype != np.dtype("bool"):
        raise ValueError("valid_mask must be a one-dimensional bool array matching samples.")
    return mask.copy()


def _validate_burst_inputs(
    bursts: Iterable[ImpulseBurst], length: int, max_length: int
) -> list[ImpulseBurst]:
    """Accept nonoverlapping maximal candidate runs, ordered or unordered."""
    if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 1:
        raise ValueError("max_length must be a positive integer.")

    ordered = list(bursts)
    if any(not isinstance(burst, ImpulseBurst) for burst in ordered):
        raise TypeError("bursts must contain ImpulseBurst instances from ip_ii.bursts.")
    for burst in ordered:
        if (isinstance(burst.start_index, bool)
                or isinstance(burst.end_index, bool)
                or not isinstance(burst.start_index, int)
                or not isinstance(burst.end_index, int)
                or burst.start_index < 0
                or burst.end_index < burst.start_index
                or burst.end_index >= length):
            raise ValueError("Burst indices must describe a valid inclusive sample interval.")
        if burst.max_length != max_length:
            raise ValueError("Burst max_length differs from the interpolation max_length.")

    ordered.sort(key=lambda burst: burst.start_index)
    for previous, current in zip(ordered, ordered[1:]):
        if current.start_index <= previous.end_index + 1:
            raise ValueError("Bursts must be separate maximal runs (not overlapping/adjacent).")
    return ordered


def _interpolate_slice(samples: np.ndarray, start: int, end: int) -> None:
    """Fill a known-valid interior interval in a private working copy."""
    count = end - start + 1
    left = float(samples[start - 1])
    right = float(samples[end + 1])
    alpha = np.arange(1, count + 1, dtype=np.float64) / (count + 1)
    samples[start : end + 1] = left + (right - left) * alpha


def interpolate_bursts(
    samples: np.ndarray,
    bursts: Iterable[ImpulseBurst],
    *,
    max_length: int = 4,
    valid_mask: np.ndarray | None = None,
) -> InterpolationResult:
    """Interpolate separate candidate runs in one mono channel.

    A candidate run is reconstructed only when:
    - its length is at most ``max_length``;
    - it has both a preceding and a succeeding sample;
    - BOTH neighbors are trusted by ``valid_mask`` (if provided);
    - neither neighboring sample belongs to ANY flagged run, even a skipped one.

    Overlong runs and runs without two trusted anchors remain UNCHANGED,
    and their reasons are returned in ``skipped_bursts``. No extrapolation
    or silent splitting of overlong runs is performed. The original input
    remains unchanged even when reconstruction is successful.

    ``valid_mask`` applies to the ORIGINAL signal: True marks a trusted
    sample, False a sample that must not be used as an interpolation anchor.
    Flagged bursts are automatically excluded regardless of this mask.
    Stereo WAV should be processed once per channel, not as a 2D array.
    """
    restored = _copy_validated_samples(samples)
    ordered = _validate_burst_inputs(bursts, restored.size, max_length)
    anchors_valid = _validated_valid_mask(valid_mask, restored.size)

    # Mask all candidate runs first: even a skipped/overlong run is not
    # a safe source of endpoints for another interpolation.
    for burst in ordered:
        anchors_valid[burst.start_index : burst.end_index + 1] = False

    reconstructed: list[ImpulseBurst] = []
    skipped: list[SkippedBurst] = []
    for burst in ordered:
        start, end = burst.start_index, burst.end_index
        if burst.length > max_length:
            skipped.append(SkippedBurst(start, end, "exceeds_max_length"))
        elif (start == 0 or end == restored.size - 1
              or not anchors_valid[start - 1]
              or not anchors_valid[end + 1]):
            skipped.append(SkippedBurst(start, end, "missing_valid_neighbors"))
        else:
            _interpolate_slice(restored, start, end)
            reconstructed.append(burst)

    return InterpolationResult(
        samples=restored,
        reconstructed_bursts=tuple(reconstructed),
        skipped_bursts=tuple(skipped),
    )


def interpolate_run(
    samples: np.ndarray,
    start_index: int,
    end_index: int,
    *,
    max_length: int = 4,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Convenience function for ONE run; raise if it cannot be restored.

    For batches with mixed valid/invalid runs, use ``interpolate_bursts``
    and inspect ``skipped_bursts`` instead of receiving an exception.
    """
    result = interpolate_bursts(
        samples,
        [ImpulseBurst(start_index, end_index, max_length)],
        max_length=max_length,
        valid_mask=valid_mask,
    )
    if result.skipped_bursts:
        reason = result.skipped_bursts[0].reason
        raise ValueError(f"Cannot interpolate burst [{start_index}, {end_index}]: {reason}.")
    return result.samples


class LinearInterpolator:
    """Linear reconstruction policy with the original four-sample limit.

    Uses the unchanged pure functions to preserve anchor and skip logic.
    """

    def __init__(self, max_length: int = 4) -> None:
        self.max_length = max_length

    def interpolate_run(
        self, samples: np.ndarray, start_index: int, end_index: int,
        *, valid_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        return interpolate_run(samples, start_index, end_index,
                               max_length=self.max_length, valid_mask=valid_mask)

    def interpolate_bursts(
        self, samples: np.ndarray, bursts: Iterable[ImpulseBurst],
        *, valid_mask: np.ndarray | None = None,
    ) -> InterpolationResult:
        return interpolate_bursts(samples, bursts,
                                  max_length=self.max_length, valid_mask=valid_mask)
