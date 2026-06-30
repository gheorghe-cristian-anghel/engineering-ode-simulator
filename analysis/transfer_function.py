"""Reusable continuous-time transfer function utilities."""

from dataclasses import dataclass

import numpy as np
from scipy import signal


def _as_coefficient_array(coefficients, name):
    """Return transfer-function coefficients as a validated 1D float array."""
    coefficient_array = np.asarray(coefficients, dtype=float)

    if coefficient_array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional sequence")

    if len(coefficient_array) == 0:
        raise ValueError(f"{name} must not be empty")

    if not np.all(np.isfinite(coefficient_array)):
        raise ValueError(f"{name} must contain only finite values")

    return coefficient_array


def _validate_denominator(denominator):
    """Validate denominator-specific transfer-function requirements."""
    if np.isclose(denominator[0], 0.0):
        raise ValueError("denominator leading coefficient must not be zero")


def _validate_time_span(t_span):
    """Validate a simulation time span."""
    if len(t_span) != 2:
        raise ValueError("t_span must contain start and end times")

    start_time = float(t_span[0])
    end_time = float(t_span[1])

    if end_time <= start_time:
        raise ValueError("t_span final time must be greater than initial time")

    return start_time, end_time


def _validate_time_array(t):
    """Return an explicitly supplied time array after validation."""
    time = np.asarray(t, dtype=float)

    if time.ndim != 1:
        raise ValueError("t must be a one-dimensional array")

    if len(time) == 0:
        raise ValueError("t must not be empty")

    if not np.all(np.isfinite(time)):
        raise ValueError("t must contain only finite values")

    if len(time) > 1 and not np.all(np.diff(time) > 0):
        raise ValueError("t must be strictly increasing")

    return time


def _time_samples(t, t_span, num_points):
    """Return validated time samples for a response simulation."""
    if t is not None:
        return _validate_time_array(t)

    if num_points <= 0:
        raise ValueError("num_points must be positive")

    start_time, end_time = _validate_time_span(t_span)
    return np.linspace(start_time, end_time, num_points)


@dataclass
class TransferFunctionModel:
    """Continuous-time transfer function model.

    Coefficients are stored in descending powers of ``s``.
    """

    numerator: object
    denominator: object
    name: str = "Transfer Function"

    def __post_init__(self):
        """Validate and store numerator and denominator as float arrays."""
        numerator = _as_coefficient_array(self.numerator, "numerator")
        denominator = _as_coefficient_array(self.denominator, "denominator")
        _validate_denominator(denominator)

        object.__setattr__(self, "numerator", numerator)
        object.__setattr__(self, "denominator", denominator)

    def to_scipy(self):
        """Return a ``scipy.signal.TransferFunction`` representation."""
        return signal.TransferFunction(self.numerator, self.denominator)


def create_transfer_function(num, den, name="Transfer Function"):
    """Create a validated continuous-time transfer function model."""
    return TransferFunctionModel(num, den, name=name)


def simulate_step_response(
    tf_model,
    t=None,
    t_span=(0.0, 10.0),
    num_points=1000,
):
    """Simulate the continuous-time unit step response of a transfer function."""
    time = _time_samples(t, t_span, num_points)
    system = tf_model.to_scipy()
    response_time, response = signal.step(system, T=time)

    return response_time, response


def simulate_impulse_response(
    tf_model,
    t=None,
    t_span=(0.0, 10.0),
    num_points=1000,
):
    """Simulate the continuous-time impulse response of a transfer function."""
    time = _time_samples(t, t_span, num_points)
    system = tf_model.to_scipy()
    response_time, response = signal.impulse(system, T=time)

    return response_time, response


def first_order_lowpass_tf(K=1.0, tau=1.0):
    """Return ``G(s) = K / (tau*s + 1)`` as a transfer function model."""
    if tau <= 0:
        raise ValueError("tau must be positive")

    return TransferFunctionModel(
        [K],
        [tau, 1.0],
        name="First-Order Low-Pass",
    )


def second_order_lowpass_tf(K=1.0, omega_n=1.0, zeta=0.5):
    """Return a standard second-order low-pass transfer function model."""
    if omega_n <= 0:
        raise ValueError("omega_n must be positive")

    if zeta < 0:
        raise ValueError("zeta must be nonnegative")

    return TransferFunctionModel(
        [K * omega_n**2],
        [1.0, 2 * zeta * omega_n, omega_n**2],
        name="Second-Order Low-Pass",
    )


def rlc_capacitor_voltage_tf(R=2.0, L=1.0, C_value=0.25):
    """Return the series RLC capacitor-voltage low-pass transfer function."""
    if R < 0:
        raise ValueError("R must be nonnegative")

    if L <= 0:
        raise ValueError("L must be positive")

    if C_value <= 0:
        raise ValueError("C_value must be positive")

    return TransferFunctionModel(
        [1.0],
        [L * C_value, R * C_value, 1.0],
        name="Series RLC Capacitor Voltage",
    )


def plot_time_response(
    t,
    y,
    title="Transfer Function Response",
    ylabel="Output",
):
    """Return a Matplotlib figure for a transfer-function time response."""
    import matplotlib.pyplot as plt

    time = np.asarray(t, dtype=float)
    response = np.asarray(y, dtype=float)

    if time.ndim != 1 or response.ndim != 1:
        raise ValueError("t and y must be one-dimensional arrays")

    if len(time) != len(response):
        raise ValueError("t and y must have the same length")

    figure, axis = plt.subplots()
    axis.plot(time, response, label=ylabel)
    axis.set_title(title)
    axis.set_xlabel("Time (s)")
    axis.set_ylabel(ylabel)
    axis.grid(True)
    axis.legend()
    figure.tight_layout()

    return figure
