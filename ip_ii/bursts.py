"""Group consecutive, *detected* impulse candidates into contiguous runs.

Project 1 allows true corruption bursts of up to four samples. This
module groups consecutive True decisions from the step-7 detector; it
DOES NOT infer that every grouped candidate is truly corrupted.

An overlong detected run is returned as ONE run and explicitly marked
as exceeding the assignment's maximum. It must not be silently split
into apparently acceptable 4-sample runs. Reconstruction will be added
in steps 9-10, not here.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ImpulseBurst:
    """Closed interval of consecutive flagged sample indices, inclusive."""

    start_index: int
    end_index: int
    max_length: int

    @property
    def length(self) -> int:
        """Number of samples in the closed interval [start_index, end_index]."""
        return self.end_index - self.start_index + 1

    @property
    def within_limit(self) -> bool:
        """Whether the candidate burst is compatible with the 4-sample assumption."""
        return self.length <= self.max_length


class ImpulseBurstGrouper:
    """Online grouping of successive detector decisions for ONE channel.

    Feed one decision per *consecutive* sample index, in chronological
    order. True extends a run; False closes it; None (warm-up / no
    decision) also breaks the run without declaring a clean sample.

    ``add`` returns a finished ImpulseBurst once a run ends, otherwise
    None. ``finish`` flushes a run reaching the last processed sample.
    Memory complexity is O(1) regardless of recording duration.
    """

    def __init__(self, max_length: int = 4) -> None:
        if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 1:
            raise ValueError("Maximum burst length must be a positive integer.")
        self.max_length = max_length
        self._start_index: int | None = None
        self._end_index: int | None = None
        self._last_processed_index: int | None = None

    def add(self, index: int, detected: bool | None) -> ImpulseBurst | None:
        """Consume a detector decision at a zero-based sample index."""
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("Sample index must be a nonnegative integer.")
        if self._last_processed_index is not None and index != self._last_processed_index + 1:
            raise ValueError("Samples must be supplied in consecutive increasing order.")
        if detected is not None and not isinstance(detected, bool):
            raise TypeError("Detection must be True, False, or None (unclassified).")

        self._last_processed_index = index
        if detected:
            if self._start_index is None:
                self._start_index = index
            self._end_index = index
            return None
        return self.finish()

    def finish(self) -> ImpulseBurst | None:
        """Close and return any active burst (e.g. at end of recording)."""
        if self._start_index is None:
            return None
        assert self._end_index is not None
        burst = ImpulseBurst(
            start_index=self._start_index,
            end_index=self._end_index,
            max_length=self.max_length,
        )
        self._start_index = None
        self._end_index = None
        return burst
