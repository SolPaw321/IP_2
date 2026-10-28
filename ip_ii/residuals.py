"""Local statistics of AR one-step prediction errors (project steps 6-7).

The lecture (wyklad_1.pdf, p. 49) estimates residual variance using the
PREVIOUS M prediction errors (not the current error):

    variance(k) = (1/M) * sum(error(k-i)**2 for i in 1..M)
    sigma(k)    = sqrt(variance(k))

This is a mean SQUARE, without subtraction of a sample mean. It assumes
prediction errors should be approximately zero-mean. This class computes
statistics only; the 3-sigma decision rule is in detector.py (step 7).
"""

from collections import deque
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ResidualStatistics:
    """Variance and root-mean-square level of a FULL residual window."""

    variance: float
    standard_deviation: float
    sample_count: int


class ResidualWindow:
    """Maintain a local window of the last M accepted prediction errors.

    Each sample should be processed in this order:

        statistics = window.snapshot()  # previous errors ONLY
        # predict and (later) classify current error using statistics
        window.add(error)              # after using statistics

    ``snapshot()`` returns None until M previous errors exist. In steps 6-7, every computed error is added (diagnostic-only detection).
    Step 10 will integrate exclusion/reconstruction of corrupt samples.

    Bookkeeping of the squared-error sum is O(1) per sample. A periodic
    re-sum with math.fsum limits numerical drift in long audio recordings.
    """

    def __init__(self, window_size: int) -> None:
        if isinstance(window_size, bool) or not isinstance(window_size, int) or window_size < 1:
            raise ValueError("Residual window size M must be a positive integer.")
        self.window_size = window_size
        self._squared_errors: deque[float] = deque()
        self._sum_squared_errors = 0.0
        self._updates = 0

    @property
    def history_count(self) -> int:
        """How many previous residuals are available (at most M)."""
        return len(self._squared_errors)

    def snapshot(self) -> ResidualStatistics | None:
        """Return local statistics of PREVIOUS errors; None during warm-up.

        No use is made of the current sample or of future samples.
        """
        if self.history_count < self.window_size:
            return None
        variance = max(0.0, self._sum_squared_errors / self.window_size)
        return ResidualStatistics(
            variance=variance,
            standard_deviation=math.sqrt(variance),
            sample_count=self.window_size,
        )

    def add(self, error: float) -> None:
        """Append a finite error and discard the oldest one if necessary."""
        residual = float(error)
        if not math.isfinite(residual):
            raise ValueError("Prediction error must be finite.")
        squared = residual * residual
        if not math.isfinite(squared):
            raise ValueError("Squared prediction error must be finite.")

        if self.history_count == self.window_size:
            self._sum_squared_errors -= self._squared_errors.popleft()
        self._squared_errors.append(squared)
        self._sum_squared_errors += squared
        self._updates += 1

        # Guard against floating-point cancellation in long recordings.
        if self._updates % max(128, 8 * self.window_size) == 0:
            self._sum_squared_errors = math.fsum(self._squared_errors)
        if self._sum_squared_errors < 0:
            self._sum_squared_errors = math.fsum(self._squared_errors)
