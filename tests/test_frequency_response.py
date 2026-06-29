import numpy as np
import pytest

from analysis.frequency_response import (
    compute_frequency_response,
    first_order_transfer_function,
    rlc_lowpass_transfer_function,
    second_order_transfer_function,
)


def test_frequency_array_is_increasing_and_positive():
    """Generated frequency samples should be positive and increasing."""
    num, den = first_order_transfer_function()

    w, _, _ = compute_frequency_response(num, den)

    assert np.all(w > 0)
    assert np.all(np.diff(w) > 0)


def test_first_order_low_frequency_magnitude_is_near_zero_db():
    """A unity-gain first-order low-pass should be near 0 dB at low frequency."""
    num, den = first_order_transfer_function(K=1.0, tau=1.0)

    _, magnitude_db, _ = compute_frequency_response(num, den, w=[1e-3])

    assert magnitude_db[0] == pytest.approx(0.0, abs=0.01)


def test_first_order_cutoff_is_near_minus_three_db():
    """At w = 1/tau, a first-order low-pass is approximately -3.01 dB."""
    num, den = first_order_transfer_function(K=1.0, tau=1.0)

    _, magnitude_db, _ = compute_frequency_response(num, den, w=[1.0])

    assert magnitude_db[0] == pytest.approx(-3.0103, abs=0.01)


def test_first_order_high_frequency_phase_approaches_minus_ninety_degrees():
    """A first-order low-pass phase should approach -90 degrees."""
    num, den = first_order_transfer_function(K=1.0, tau=1.0)

    _, _, phase_deg = compute_frequency_response(num, den, w=[1e3])

    assert phase_deg[0] == pytest.approx(-90.0, abs=0.1)


def test_second_order_helper_returns_expected_coefficients():
    """Second-order helper should return standard low-pass coefficients."""
    num, den = second_order_transfer_function(omega_n=5.0, zeta=0.3, K=2.0)

    assert num == pytest.approx([50.0])
    assert den == pytest.approx([1.0, 3.0, 25.0])


def test_rlc_helper_returns_expected_coefficients():
    """RLC helper should return capacitor-voltage transfer coefficients."""
    num, den = rlc_lowpass_transfer_function(R=2.0, L=1.0, C=0.25)

    assert num == pytest.approx([1.0])
    assert den == pytest.approx([0.25, 0.5, 1.0])
