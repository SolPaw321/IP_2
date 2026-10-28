"""Impulse removal by AR(4), gated EW-LS, 3-sigma detection, interpolation.

Two optional improvements (both enabled by default):

1. After a flagged burst of at most four samples, treat the next raw
   observation as a *possible* right anchor. Hypothetically interpolate the
   burst, then evaluate that right anchor against an AR regressor built from
   the interpolated values. Commit interpolation and update EW-LS/residual
   history ONLY if the anchor passes the unchanged 3-sigma test; otherwise
   retain the original provisional context and extend the burst. This avoids
   a stale, pre-interpolation regressor in the accepted EW-LS update.

2. Offline EW-LS pre-calibration: find a sufficiently audible/predictable short
   window in the first half-second, train a *separate* EW-LS on that window,
   and copy ONLY its coefficients into the actual estimator as theta(0).
   The main run starts from k=order with its original P(0), lambda, residual
   history and 3-sigma rule. If calibration is impossible, theta(0)=0 as before.
   This initialization uses future observations; hence the complete algorithm
   is offline and is NOT strictly causal in its initialization stage.

Detection flags remain fallible. Bursts longer than the instructed limit are
not silently split or repaired. The overlong-burst history-reset heuristic
from step 10 is intentionally retained unchanged.
"""

from dataclasses import dataclass
from typing import Callable
import math

import numpy as np

from ip_ii.ar import ARModel
from ip_ii.audio import PCM16_MAX_NORMALIZED
from ip_ii.bursts import ImpulseBurst, ImpulseBurstGrouper
from ip_ii.detector import ImpulseDecision, ImpulseDetector
from ip_ii.ewls import EWLS
from ip_ii.interpolation import InterpolationResult, LinearInterpolator
from ip_ii.residuals import ResidualStatistics, ResidualWindow
from ip_ii.initialization import EWLSInitializer


ALGORITHM_REVISION = "ewls_pilot_and_anchor_refit_v1"


@dataclass(frozen=True)
class DeclickObservation:
    """Selected sample diagnostics for CSV writing (one audio channel)."""

    index: int
    observed: float
    prediction: float
    error: float
    statistics: ResidualStatistics | None
    residual_history_count: int
    decision: ImpulseDecision
    updated: bool
    coefficients: np.ndarray
    save_parameters: bool


@dataclass(frozen=True)
class DeclickResult:
    """Cleaned channel and diagnostics; the original array is never modified."""

    samples: np.ndarray
    coefficients: np.ndarray
    prediction_rmse: float
    ewls_updates: int
    residual_restarts: int
    classified: int
    detected: int
    unclassified: int
    bursts: tuple[ImpulseBurst, ...]
    interpolation: InterpolationResult
    final_residual_statistics: ResidualStatistics | None
    initialization_segment: tuple[int, int] | None
    rechecked_anchor_count: int


ObservationCallback = Callable[[DeclickObservation], None]


def _calibrate_initial_coefficients(
    raw: np.ndarray,
    estimator: EWLS,
    *,
    sample_rate: int,
    residual_window_size: int,
) -> tuple[int, int] | None:
    """Use a separate EW-LS pilot to warm-start theta without changing P(0).

    Candidate windows are drawn from the first 0.5 s. Each pilot learns on
    the first half of a window; the second half supplies an out-of-sample
    normalized RMS prediction error for selection. Very quiet segments and
    pathological coefficient estimates are rejected. All pilots use the
    SAME order, lambda and P(0) as the main estimator. No extra denoising,
    interpolation, classification or alternate AR model is introduced.

    This is an *offline* calibration and may select a future audio segment.
    It is not guaranteed to be click-free; a poor/no candidate falls back to
    the previous all-zero coefficient initialization.
    """
    order = estimator.order
    prefix_end = min(raw.size, max(order + 1, round(sample_rate * 0.5)))
    window = max(64, 4 * residual_window_size, round(sample_rate * 0.025))
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
                phi = _regression_vector_trusted(raw, k, order)
                pilot._update_trusted(float(raw[k]), phi)
        except (FloatingPointError, ValueError):
            continue

        coefficients = pilot.coefficients
        if not np.all(np.isfinite(coefficients)) or np.linalg.norm(coefficients, ord=1) > 8.0:
            continue

        # Evaluate prediction quality on samples not used to fit the pilot.
        errors = []
        for k in range(middle, end):
            phi = _regression_vector_trusted(raw, k, order)
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


class DeclickProcessor:
    """Orchestrate one-channel denoising with independent state for each call.

    The numerical loop is moved unchanged from process_channel. The processor
    composes ARModel, EWLSInitializer, LinearInterpolator and the existing
    EWLS/ResidualWindow/ImpulseDetector/ImpulseBurstGrouper classes.
    """

    def __init__(
        self, *, order: int = 4, forgetting_factor: float = 0.999,
        initial_inverse_correlation: float = 100.0,
        residual_window_size: int = 20, detection_threshold: float = 3.0,
        max_burst_length: int = 4, log_stride: int = 220,
        bootstrap_initialization: bool = True,
        recompute_after_interpolation: bool = True,
    ) -> None:
        self.model = ARModel(order)
        self.initializer = EWLSInitializer(self.model, residual_window_size)
        self.interpolator = LinearInterpolator(max_burst_length)
        self.forgetting_factor = forgetting_factor
        self.initial_inverse_correlation = initial_inverse_correlation
        self.residual_window_size = residual_window_size
        self.detection_threshold = detection_threshold
        self.max_burst_length = max_burst_length
        self.log_stride = log_stride
        self.bootstrap_initialization = bootstrap_initialization
        self.recompute_after_interpolation = recompute_after_interpolation

    def process(
        self, samples: np.ndarray, *, sample_rate: int,
        on_observation: ObservationCallback | None = None,
    ) -> DeclickResult:
        """Run the original AR(4)/EW-LS/3-sigma/interpolation loop."""
        order = self.model.order
        forgetting_factor = self.forgetting_factor
        initial_inverse_correlation = self.initial_inverse_correlation
        residual_window_size = self.residual_window_size
        detection_threshold = self.detection_threshold
        max_burst_length = self.max_burst_length
        log_stride = self.log_stride
        bootstrap_initialization = self.bootstrap_initialization
        recompute_after_interpolation = self.recompute_after_interpolation
        x = np.asarray(samples)
        if x.ndim != 1 or not np.issubdtype(x.dtype, np.floating) or x.size == 0:
            raise ValueError("Expected a nonempty one-dimensional float audio channel.")
        if not np.all(np.isfinite(x)):
            raise ValueError("Audio contains NaN or infinity.")
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
            raise ValueError("sample_rate must be a positive integer.")
        if isinstance(log_stride, bool) or not isinstance(log_stride, int) or log_stride < 1:
            raise ValueError("log_stride must be a positive integer.")
        if x.size <= order:
            raise ValueError("Recording must contain more samples than the AR order.")

        raw = np.array(x, dtype=np.float64, copy=True)
        context = raw.copy()
        estimator = EWLS(
            order=order,
            forgetting_factor=forgetting_factor,
            initial_inverse_correlation=initial_inverse_correlation,
        )
        initialization_segment = (
            self.initializer.calibrate(raw, estimator, sample_rate=sample_rate)
            if bootstrap_initialization else None
        )
        history = ResidualWindow(window_size=residual_window_size)
        detector = ImpulseDetector(threshold_multiplier=detection_threshold)
        grouper = ImpulseBurstGrouper(max_length=max_burst_length)

        bursts: list[ImpulseBurst] = []
        classified = detected_count = unclassified = 0
        warmup_end = max(order, round(sample_rate * 0.1))
        sum_squared_error = 0.0
        mse_count = 0
        consecutive_flags = 0
        residual_restarts = 0
        rechecked_anchor_count = 0
        pending_start: int | None = None

        def close_burst(burst: ImpulseBurst | None, right_index: int | None) -> None:
            if burst is None:
                return
            bursts.append(burst)
            if (burst.within_limit and right_index is not None
                    and burst.start_index > 0 and right_index == burst.end_index + 1):
                # Short local slice means the step-9 interpolator copies <= 6
                # values instead of an entire 20-second recording every time.
                section = context[burst.start_index - 1 : right_index + 1]
                replaced = self.interpolator.interpolate_run(
                    section, 1, burst.length,
                )
                context[burst.start_index : right_index] = replaced[1:-1]
            # For a long/unclosed run, keep bounded forecasts in *context* to
            # avoid injecting known suspect raw values into the next AR lags.
            # Final output, however, retains raw samples when reconstruction
            # is not possible, as the assignment does not allow extrapolation.

        for k in range(order, raw.size):
            observed = float(raw[k])
            stats = history.snapshot()  # previous ACCEPTED residuals only
            history_count = history.history_count

            # Recompute the right-anchor regressor *after* hypothetical
            # interpolation, not from the stale bounded forecasts.
            pending_length = (k - pending_start) if pending_start is not None else 0
            try_interpolation = (
                recompute_after_interpolation
                and pending_start is not None
                and pending_start > 0
                and 1 <= pending_length <= max_burst_length
            )
            former_context: np.ndarray | None = None
            if try_interpolation:
                assert pending_start is not None
                former_context = context[pending_start:k].copy()
                # A candidate right endpoint must NOT be copied into the AR
                # regressor as the current observed sample. It is used only
                # to interpolate already flagged samples before k.
                section = context[pending_start - 1 : k + 1].copy()
                section[-1] = observed
                hypothetical = self.interpolator.interpolate_run(
                    section, 1, pending_length,
                )
                context[pending_start:k] = hypothetical[1:-1]

            phi = self.model.regressor_trusted(context, k)
            predicted = estimator._predict_trusted(phi)
            error = self.model.error(observed, predicted)
            decision = detector.evaluate(error, stats)

            if decision.detected is True:
                detected_count += 1
                consecutive_flags += 1
                if former_context is not None:
                    # This was NOT a valid right anchor: rollback the
                    # provisional interpolation before extending the run.
                    assert pending_start is not None
                    context[pending_start:k] = former_context
                    forecast_phi = self.model.regressor_trusted(context, k)
                    working_prediction = estimator._predict_trusted(forecast_phi)
                else:
                    working_prediction = predicted
                context[k] = float(np.clip(working_prediction, -1.0, PCM16_MAX_NORMALIZED))
                if pending_start is None:
                    pending_start = k
                close_burst(grouper.add(k, True), None)
                updated = False
                if consecutive_flags > max_burst_length:
                    # Retain the existing overlong-run recovery heuristic.
                    history = ResidualWindow(window_size=residual_window_size)
                    residual_restarts += 1
                    consecutive_flags = 0
            else:
                if try_interpolation:
                    rechecked_anchor_count += 1
                consecutive_flags = 0
                context[k] = observed
                close_burst(grouper.add(k, decision.detected), k)
                pending_start = None
                if decision.detected is None:
                    unclassified += 1
                else:
                    classified += 1
                # phi and error were obtained using the repaired context.
                estimator._update_trusted(observed, phi, prior_prediction=predicted)
                history.add(error)
                updated = True
                if k >= warmup_end:
                    sum_squared_error += error * error
                    mse_count += 1

            if decision.detected is True:
                classified += 1

            save_params = (k in (order, order + residual_window_size, raw.size - 1)
                           or k % log_stride == 0)
            if on_observation is not None and (save_params or decision.detected is True):
                on_observation(DeclickObservation(
                    index=k,
                    observed=observed,
                    prediction=predicted,
                    error=error,
                    statistics=stats,
                    residual_history_count=history_count,
                    decision=decision,
                    updated=updated,
                    coefficients=estimator.coefficients.copy(),
                    save_parameters=save_params,
                ))

        # No trusted right neighbor for any run that reaches the file ending.
        close_burst(grouper.finish(), None)

        interpolation = self.interpolator.interpolate_bursts(raw, bursts)
        rmse = math.sqrt(sum_squared_error / mse_count) if mse_count else float("nan")
        if estimator.steps != raw.size - order - detected_count:
            raise RuntimeError("EW-LS update bookkeeping is inconsistent.")
        return DeclickResult(
            samples=interpolation.samples,
            coefficients=estimator.coefficients.copy(),
            prediction_rmse=rmse,
            ewls_updates=estimator.steps,
            residual_restarts=residual_restarts,
            classified=classified,
            detected=detected_count,
            unclassified=unclassified,
            bursts=tuple(bursts),
            interpolation=interpolation,
            final_residual_statistics=history.snapshot(),
            initialization_segment=initialization_segment,
            rechecked_anchor_count=rechecked_anchor_count,
        )


def process_channel(
    samples: np.ndarray,
    *, sample_rate: int,
    order: int = 4,
    forgetting_factor: float = 0.999,
    initial_inverse_correlation: float = 100.0,
    residual_window_size: int = 20,
    detection_threshold: float = 3.0,
    max_burst_length: int = 4,
    log_stride: int = 220,
    on_observation: ObservationCallback | None = None,
    bootstrap_initialization: bool = True,
    recompute_after_interpolation: bool = True,
) -> DeclickResult:
    """Backward-compatible functional entry point; delegates to DeclickProcessor."""
    return DeclickProcessor(
        order=order, forgetting_factor=forgetting_factor,
        initial_inverse_correlation=initial_inverse_correlation,
        residual_window_size=residual_window_size,
        detection_threshold=detection_threshold, max_burst_length=max_burst_length,
        log_stride=log_stride, bootstrap_initialization=bootstrap_initialization,
        recompute_after_interpolation=recompute_after_interpolation,
    ).process(samples, sample_rate=sample_rate, on_observation=on_observation)
