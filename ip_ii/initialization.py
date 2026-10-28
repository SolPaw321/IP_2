"""Offline EW-LS pilot initialization (original formula and windows unchanged)."""

from __future__ import annotations

import numpy as np

from ip_ii.ewls import EWLS
from ip_ii.ar import ARModel


class EWLSInitializer:
    """Select a predictable early segment to seed theta(0), not P(0)."""

    def __init__(self, model: ARModel, residual_window_size: int) -> None:
        self.model = model
        self.residual_window_size = residual_window_size

    def calibrate(
        self, raw: np.ndarray, estimator: EWLS, *, sample_rate: int,
    ) -> tuple[int, int] | None:
        """Choose a first-half-second pilot based on held-out prediction RMS.

        EW-LS initial P, lambda and the main residual warmup remain unchanged.
        """
        order = estimator.order
        prefix_end = min(raw.size, max(order + 1, round(sample_rate * 0.5)))
        window = max(64, 4 * self.residual_window_size, round(sample_rate * 0.025))
        if prefix_end < window or window < 2 * (order + 4):
            return None

        stride = max(1, window // 2)
        starts = list(range(0, prefix_end - window + 1, stride))
        if starts[-1] != prefix_end - window:
            starts.append(prefix_end - window)

        best_score = float("inf")
        best_coefficients: np.ndarray | None = None
        best_segment: tuple[int, int] | None = None

        for start in starts:
            end = start + window
            middle = start + window // 2
            target = raw[start + order : end]
            if np.sqrt(np.mean(target * target)) < 16.0 / 32768.0:
                continue

            pilot = EWLS(
                order=order,
                forgetting_factor=estimator.forgetting_factor,
                initial_inverse_correlation=float(estimator.inverse_correlation[0, 0]),
            )
            try:
                for k in range(start + order, middle):
                    phi = self.model.regressor_trusted(raw, k)
                    pilot._update_trusted(float(raw[k]), phi)
            except (FloatingPointError, ValueError):
                continue

            coefficients = pilot.coefficients
            if not np.all(np.isfinite(coefficients)) or np.linalg.norm(coefficients, ord=1) > 8.0:
                continue

            # Evaluate prediction quality on samples not used to fit the pilot.
            errors = []
            for k in range(middle, end):
                phi = self.model.regressor_trusted(raw, k)
                errors.append(float(raw[k]) - pilot._predict_trusted(phi))
            errors_array = np.asarray(errors, dtype=np.float64)
            observed = raw[middle:end]
            energy = float(np.sqrt(np.mean(observed * observed)))
            if energy < 16.0 / 32768.0:
                continue
            score = float(np.sqrt(np.mean(errors_array**2)) / energy)
            if np.isfinite(score) and score < best_score:
                best_score = score
                best_coefficients = coefficients.copy()
                best_segment = (start, end - 1)

        if best_coefficients is None:
            return None
        estimator.coefficients[:] = best_coefficients
        return best_segment

