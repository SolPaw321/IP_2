"""Autoregressive AR(r) model: lagged observations and one-step prediction.

Lecture 1 uses the convention

    y(k) = a_1*y(k-1) + ... + a_r*y(k-r) + n(k)
         = phi(k).T @ theta + n(k)

where phi(k) = [y(k-1), ..., y(k-r)] and theta = [a_1, ..., a_r].

The project uses r=4 (set in config.py). No parameters are estimated
here: step 5 will supply theta through exponentially weighted LS.

All indices are ZERO-BASED Python indices. For example, when order=4,
index=4 means predicting samples[4] from samples[3:0:-1] and samples[0].
"""

import numpy as np


def regression_vector(samples: np.ndarray, index: int, *, order: int = 4) -> np.ndarray:
    """Return phi(index) = [y(index-1), ..., y(index-order)].

    ``samples`` must be a one-dimensional, real-valued signal (ONE audio
    channel). ``index`` is the sample being predicted, not the first lag.

    The admissible range is ``order <= index <= len(samples)``. The last
    value represents forecasting the *next*, not yet recorded, sample.
    An independent array is returned, in float64, newest sample first.

    Only the selected ``order`` samples are examined for finite values.
    This makes the function inexpensive when EW-LS later calls it for
    every time step of a long recording.
    """
    values = np.asarray(samples)
    if values.ndim != 1:
        raise ValueError("AR regression expects one channel: a 1D signal.")
    if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
        raise TypeError("AR regression expects real-valued numeric samples.")
    if isinstance(order, (bool, np.bool_)) or not isinstance(order, (int, np.integer)) or order < 1:
        raise ValueError("AR order must be a positive integer.")
    if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)):
        raise TypeError("Sample index must be an integer (zero-based).")
    if not order <= index <= values.size:
        raise ValueError(
            f"Sample index must be within [{order}, {values.size}] "
            f"for AR({order})."
        )

    # Newest first: y(k-1), y(k-2), ..., y(k-r).
    lags = np.array(values[index - order : index][::-1], dtype=np.float64)
    if not np.all(np.isfinite(lags)):
        raise ValueError("AR regression cannot use NaN or infinite lagged samples.")
    return lags


def _regression_vector_trusted(samples: np.ndarray, index: int, order: int) -> np.ndarray:
    """Fast internal regressor for a validated finite 1D float64 channel.

    PRECONDITIONS: ``0 < order <= index <= len(samples)`` and all samples
    are finite. Kept separate from the fully validating public function.
    Identical slice, reversal, copy and float64 dtype as ``regression_vector``.
    """
    return np.array(samples[index - order : index][::-1], dtype=np.float64)


def predict_one_step(regressor: np.ndarray, coefficients: np.ndarray) -> float:
    """Predict one sample as phi(k).T @ theta, with no noise term.

    The coefficient order is [a_1, ..., a_r] and must match the
    regressor [y(k-1), ..., y(k-r)]. This function does NOT estimate
    the coefficients; they must come from a separate identification step.
    """
    phi = np.asarray(regressor, dtype=np.float64)
    theta = np.asarray(coefficients, dtype=np.float64)
    if phi.ndim != 1 or theta.ndim != 1 or phi.size == 0 or phi.shape != theta.shape:
        raise ValueError("Regressor and coefficients must be nonempty 1D arrays of equal length.")
    if not np.all(np.isfinite(phi)) or not np.all(np.isfinite(theta)):
        raise ValueError("Regressor and coefficients must contain only finite numbers.")
    return float(np.dot(phi, theta))


def prediction_error(observed: float, predicted: float) -> float:
    """Compute epsilon(k) = y(k) - y_hat(k|k-1).

    This is a *one-step-ahead* prediction error, not an interpolated
    sample and not yet a decision that an impulse disturbance occurred.
    """
    error = float(observed) - float(predicted)
    if not np.isfinite(error):
        raise ValueError("Observed and predicted values must be finite.")
    return error


class ARModel:
    """AR(r) regression/forecast operations; estimation remains in EWLS.

    The trusted regressor retains the same float64 slice and copy, allowing
    the numerical loop to avoid repeated public input validation.
    """

    def __init__(self, order: int = 4) -> None:
        if isinstance(order, bool) or not isinstance(order, int) or order < 1:
            raise ValueError("AR order must be a positive integer.")
        self.order = order

    def regressor(self, samples: np.ndarray, index: int) -> np.ndarray:
        return regression_vector(samples, index, order=self.order)

    def regressor_trusted(self, samples: np.ndarray, index: int) -> np.ndarray:
        return _regression_vector_trusted(samples, index, self.order)

    @staticmethod
    def predict(phi: np.ndarray, coefficients: np.ndarray) -> float:
        return predict_one_step(phi, coefficients)

    @staticmethod
    def error(observed: float, predicted: float) -> float:
        return prediction_error(observed, predicted)
