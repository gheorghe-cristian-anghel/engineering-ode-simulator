"""Reusable continuous-time frequency response and Bode plot helpers."""

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

    if np.allclose(coefficient_array, 0):
        raise ValueError(f"{name} must not be all zeros")

    return coefficient_array


def _validate_frequency_range(w_min, w_max, num_points):
    """Validate automatic logarithmic frequency range inputs."""
    if w_min <= 0:
        raise ValueError("w_min must be positive")

    if w_max <= w_min:
        raise ValueError("w_max must be greater than w_min")

    if num_points <= 0:
        raise ValueError("num_points must be positive")


def _validate_frequency_array(w):
    """Return an explicitly supplied frequency array after validation."""
    frequency = np.asarray(w, dtype=float)

    if frequency.ndim != 1:
        raise ValueError("w must be a one-dimensional array")

    if len(frequency) == 0:
        raise ValueError("w must not be empty")

    if not np.all(np.isfinite(frequency)):
        raise ValueError("w must contain only finite values")

    if np.any(frequency <= 0):
        raise ValueError("w values must be positive")

    return frequency


def compute_frequency_response(
    num,
    den,
    w=None,
    w_min=1e-2,
    w_max=1e2,
    num_points=500,
):
    """Compute continuous-time transfer-function frequency response.

    Parameters
    ----------
    num : sequence
        Numerator coefficients in descending powers of ``s``.
    den : sequence
        Denominator coefficients in descending powers of ``s``.
    w : array-like, optional
        Angular frequency samples in radians per second. If omitted,
        logarithmically spaced samples are generated from ``w_min`` to
        ``w_max``.
    w_min : float, optional
        Minimum angular frequency in radians per second.
    w_max : float, optional
        Maximum angular frequency in radians per second.
    num_points : int, optional
        Number of logarithmically spaced frequency samples.

    Returns
    -------
    tuple
        ``(w, magnitude_db, phase_deg)``.
    """
    numerator = _as_coefficient_array(num, "num")
    denominator = _as_coefficient_array(den, "den")

    if w is None:
        _validate_frequency_range(w_min, w_max, num_points)
        frequency = np.logspace(np.log10(w_min), np.log10(w_max), num_points)
    else:
        frequency = _validate_frequency_array(w)

    frequency, response = signal.freqs(numerator, denominator, worN=frequency)
    magnitude = np.abs(response)

    with np.errstate(divide="ignore"):
        magnitude_db = 20 * np.log10(magnitude)

    phase_deg = np.rad2deg(np.unwrap(np.angle(response)))

    return frequency, magnitude_db, phase_deg


def first_order_transfer_function(K=1.0, tau=1.0):
    """Return coefficients for ``G(s) = K / (tau*s + 1)``."""
    if tau <= 0:
        raise ValueError("tau must be positive")

    return [K], [tau, 1.0]


def second_order_transfer_function(omega_n=1.0, zeta=0.5, K=1.0):
    """Return coefficients for a standard second-order low-pass system."""
    if omega_n <= 0:
        raise ValueError("omega_n must be positive")

    if zeta < 0:
        raise ValueError("zeta must be nonnegative")

    return [K * omega_n**2], [1.0, 2 * zeta * omega_n, omega_n**2]


def rlc_lowpass_transfer_function(R=10.0, L=1.0, C=0.01):
    """Return coefficients for series RLC capacitor voltage ``Vc(s)/Vin(s)``."""
    if R < 0:
        raise ValueError("R must be nonnegative")

    if L <= 0:
        raise ValueError("L must be positive")

    if C <= 0:
        raise ValueError("C must be positive")

    return [1.0], [L * C, R * C, 1.0]


def plot_bode_response(w, magnitude_db, phase_deg, title="Bode Plot"):
    """Return a Matplotlib figure containing magnitude and phase Bode plots."""
    import matplotlib.pyplot as plt

    frequency = np.asarray(w, dtype=float)
    magnitude = np.asarray(magnitude_db, dtype=float)
    phase = np.asarray(phase_deg, dtype=float)

    if not (len(frequency) == len(magnitude) == len(phase)):
        raise ValueError("w, magnitude_db, and phase_deg must have the same length")

    figure, axes = plt.subplots(2, 1, sharex=True)

    axes[0].semilogx(frequency, magnitude)
    axes[0].set_title(title)
    axes[0].set_ylabel("Magnitude (dB)")
    axes[0].grid(True, which="both")

    axes[1].semilogx(frequency, phase)
    axes[1].set_xlabel("Angular frequency (rad/s)")
    axes[1].set_ylabel("Phase (deg)")
    axes[1].grid(True, which="both")

    figure.tight_layout()
    return figure
