"""Recursive exponentially weighted least squares (EW-LS).

Reference: wyklad_1.pdf, pp. 126-130.  For the regression equation
    y(k) = phi(k).T @ theta(k) + n(k),
EW-LS sequentially minimizes the exponentially weighted squared errors.

This module estimates model parameters only.  Impulse rejection, residual
statistics and sample reconstruction belong to later project steps.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EWLSUpdate:
    """Values from one EW-LS iteration, computed *before* and *after* update.

    prediction and error use theta(k-1) (the actual one-step forecast).
    coefficients are theta(k), after the current observation is incorporated.
    """

    prediction: float
    error: float
    coefficients: np.ndarray


class EWLS:
    """Recursive exponentially weighted least-squares parameter estimator.

    Parameters
    ----------
    order:
        Number of unknown coefficients, equal to AR order for this project.
    forgetting_factor:
        lambda in (0, 1]; smaller lambda follows changes more quickly but
        yields noisier estimates.  lambda=1 reduces to ordinary recursive LS.
    initial_inverse_correlation:
        Positive scalar delta for P(0) = delta * I.  It represents an
        initialization choice, NOT a value specified by the assignment.

    Notes
    -----
    The initial coefficients are zero. The recursion, for phi(k) and y(k), is:

        eps  = y - phi.T @ theta
        v    = P @ phi
        gain = v / (lambda + phi.T @ v)
        theta <- theta + gain * eps
        P     <- (P - outer(v, v) / (lambda + phi.T @ v)) / lambda

    Both numerator and denominator are computed with P from the previous step.
    P is symmetrized to remove floating-point asymmetry (not to regularize).

    During future click detection, the caller should use predict() to inspect
    the error BEFORE deciding whether to call update() for this observation.
    """

    def __init__(
        self,
        order: int,
        *,
        forgetting_factor: float,
        initial_inverse_correlation: float = 100.0,
    ) -> None:
        if isinstance(order, (bool, np.bool_)) or not isinstance(order, (int, np.integer)) or order < 1:
            raise ValueError("EW-LS order must be a positive integer.")
        lam = float(forgetting_factor)
        delta = float(initial_inverse_correlation)
        if not np.isfinite(lam) or not 0 < lam <= 1:
            raise ValueError("Forgetting factor lambda must be in (0, 1].")
        if not np.isfinite(delta) or delta <= 0:
            raise ValueError("Initial inverse-correlation scale must be positive and finite.")

        self.order = int(order)
        self.forgetting_factor = lam
        self.coefficients = np.zeros(self.order, dtype=np.float64)
        self.inverse_correlation = delta * np.eye(self.order, dtype=np.float64)
        self.steps = 0

    def _regressor(self, regressor: np.ndarray) -> np.ndarray:
        phi = np.asarray(regressor, dtype=np.float64)
        if phi.shape != (self.order,) or not np.all(np.isfinite(phi)):
            raise ValueError(f"EW-LS regressor must be a finite vector of shape ({self.order},).")
        return phi

    def predict(self, regressor: np.ndarray) -> float:
        """Predict y(k) using current theta, without changing estimator state."""
        phi = self._regressor(regressor)
        return self._predict_trusted(phi)

    def _predict_trusted(self, phi: np.ndarray) -> float:
        """Internal AR forecast; caller guarantees finite shape-(order,) float64 phi.

        No validation is repeated inside the per-sample declick loop.
        Mathematically identical to the checked public ``predict`` method.
        """
        return float(phi @ self.coefficients)

    def update(self, observed: float, regressor: np.ndarray) -> EWLSUpdate:
        """Incorporate one observation and return its prior forecast and residual.

        No rejection of anomalous observations is performed at this stage.
        """
        phi = self._regressor(regressor)
        y = float(observed)
        if not np.isfinite(y):
            raise ValueError("Observed sample must be finite.")

        return self._update_trusted(y, phi)

    def _update_trusted(
        self, y: float, phi: np.ndarray, *, prior_prediction: float | None = None,
    ) -> EWLSUpdate:
        """Internal update for validated inputs, with optional cached prediction.

        The same floating-point expressions and their order are used as in
        ``update``. ``prior_prediction`` must be the current-state forecast
        computed immediately before the update (no intervening EW-LS update).
        """
        prediction = self._predict_trusted(phi) if prior_prediction is None else prior_prediction
        error = y - prediction
        previous_p_phi = self.inverse_correlation @ phi
        denominator = self.forgetting_factor + float(phi @ previous_p_phi)
        if not np.isfinite(denominator) or denominator <= 0:
            raise FloatingPointError("EW-LS gain denominator is invalid.")

        gain = previous_p_phi / denominator
        new_coefficients = self.coefficients + gain * error
        new_inverse_correlation = (
            self.inverse_correlation
            - np.outer(previous_p_phi, previous_p_phi) / denominator
        ) / self.forgetting_factor
        # In exact arithmetic P is symmetric. Floating-point rounding may
        # introduce very small asymmetries; removing them limits drift.
        new_inverse_correlation = 0.5 * (new_inverse_correlation + new_inverse_correlation.T)
        if not np.all(np.isfinite(new_coefficients)) or not np.all(np.isfinite(new_inverse_correlation)):
            raise FloatingPointError("EW-LS produced a non-finite parameter or matrix. Try different initialization/forgetting.")

        self.coefficients = new_coefficients
        self.inverse_correlation = new_inverse_correlation
        self.steps += 1
        # A snapshot is copied so it will not change at subsequent iterations.
        return EWLSUpdate(prediction, error, new_coefficients.copy())
