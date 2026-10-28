"""One-step impulse detector for the AR(4) prediction residual (step 7).

Assignment: mark y(k) as suspicious when

    abs(epsilon(k)) > 3 * sigma_epsilon(k)

where sigma_epsilon(k) comes from PREVIOUS M prediction errors, as in
wyklad_1.pdf, p. 49.  This module makes a decision only: it does not
change the audio, discard EW-LS observations or interpolate bursts.
Those operations are addressed in steps 8-10.
"""

from dataclasses import dataclass
import math

from ip_ii.residuals import ResidualStatistics


@dataclass(frozen=True)
class ImpulseDecision:
    """The decision for one sample.

    detected is None until the residual window has M earlier errors.
    Otherwise True means a strict threshold exceedance, False means
    no exceedance.  In particular, equality to the threshold is False.

    normalized_error = |epsilon| / sigma (when sigma > 0), and is +inf
    for nonzero epsilon and sigma=0.  It is zero for epsilon=sigma=0.
    """

    detected: bool | None
    threshold: float | None
    normalized_error: float | None


class ImpulseDetector:
    """Apply the lecture's local-threshold rule, without side effects."""

    def __init__(self, *, threshold_multiplier: float = 3.0) -> None:
        factor = float(threshold_multiplier)
        if not math.isfinite(factor) or factor <= 0:
            raise ValueError("Detection threshold multiplier must be positive and finite.")
        self.threshold_multiplier = factor

    def evaluate(
        self,
        error: float,
        statistics: ResidualStatistics | None,
    ) -> ImpulseDecision:
        """Classify the current residual using prior-window statistics.

        No decision is possible during the first M prediction errors.
        A sigma of zero is handled by the *exact* 3-sigma rule rather
        than by inventing an undocumented minimum threshold.
        """
        residual = float(error)
        if not math.isfinite(residual):
            raise ValueError("Prediction error must be finite.")
        if statistics is None:
            return ImpulseDecision(None, None, None)

        sigma = float(statistics.standard_deviation)
        if not math.isfinite(sigma) or sigma < 0:
            raise ValueError("Local residual standard deviation must be finite and nonnegative.")

        threshold = self.threshold_multiplier * sigma
        if not math.isfinite(threshold):
            raise ValueError("Computed impulse threshold must be finite.")
        magnitude = abs(residual)
        normalized = (magnitude / sigma) if sigma > 0 else (
            math.inf if magnitude > 0 else 0.0
        )
        return ImpulseDecision(
            detected=magnitude > threshold,
            threshold=threshold,
            normalized_error=normalized,
        )
