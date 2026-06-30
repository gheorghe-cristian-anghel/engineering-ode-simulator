import numpy as np
import pytest
from numpy.testing import assert_allclose

from analysis.transfer_function import (
    TransferFunctionModel,
    first_order_lowpass_tf,
    rlc_capacitor_voltage_tf,
    second_order_lowpass_tf,
    simulate_step_response,
)


def test_transfer_function_model_stores_float_arrays():
    """Numerator and denominator should be stored as float arrays."""
    model = TransferFunctionModel([1], [2, 1])

    assert model.numerator.dtype == float
    assert model.denominator.dtype == float
    assert_allclose(model.numerator, [1.0])
    assert_allclose(model.denominator, [2.0, 1.0])


def test_empty_numerator_raises_value_error():
    """A transfer function must have numerator coefficients."""
    with pytest.raises(ValueError):
        TransferFunctionModel([], [1.0, 1.0])


def test_empty_denominator_raises_value_error():
    """A transfer function must have denominator coefficients."""
    with pytest.raises(ValueError):
        TransferFunctionModel([1.0], [])


def test_denominator_with_zero_leading_coefficient_raises_value_error():
    """The highest-order denominator coefficient must not be zero."""
    with pytest.raises(ValueError):
        TransferFunctionModel([1.0], [0.0, 1.0])


def test_first_order_lowpass_tf_returns_expected_coefficients():
    """First-order helper should return K / (tau*s + 1)."""
    model = first_order_lowpass_tf(K=2.0, tau=1.5)

    assert_allclose(model.numerator, [2.0])
    assert_allclose(model.denominator, [1.5, 1.0])
    assert model.name == "First-Order Low-Pass"


def test_second_order_lowpass_tf_returns_expected_coefficients():
    """Second-order helper should return standard low-pass coefficients."""
    model = second_order_lowpass_tf(K=2.0, omega_n=5.0, zeta=0.3)

    assert_allclose(model.numerator, [50.0])
    assert_allclose(model.denominator, [1.0, 3.0, 25.0])
    assert model.name == "Second-Order Low-Pass"


def test_rlc_capacitor_voltage_tf_returns_expected_coefficients():
    """RLC helper should return capacitor-voltage transfer coefficients."""
    model = rlc_capacitor_voltage_tf(R=2.0, L=1.0, C_value=0.25)

    assert_allclose(model.numerator, [1.0])
    assert_allclose(model.denominator, [0.25, 0.5, 1.0])
    assert model.name == "Series RLC Capacitor Voltage"


def test_simulate_step_response_returns_arrays_of_equal_length():
    """Step response simulation should return matching time and output arrays."""
    model = first_order_lowpass_tf()

    t, y = simulate_step_response(model, t_span=(0.0, 5.0), num_points=50)

    assert len(t) == len(y)
    assert len(t) == 50


def test_first_order_unit_gain_step_response_approaches_one():
    """A unit-gain first-order step response should approach 1."""
    model = first_order_lowpass_tf(K=1.0, tau=1.0)

    _, y = simulate_step_response(model, t_span=(0.0, 8.0), num_points=500)

    assert y[-1] == pytest.approx(1.0, rel=0.01)


def test_invalid_t_span_raises_value_error():
    """Final simulation time must be greater than initial time."""
    model = first_order_lowpass_tf()

    with pytest.raises(ValueError):
        simulate_step_response(model, t_span=(5.0, 0.0), num_points=50)


def test_invalid_num_points_raises_value_error():
    """Number of generated time samples must be positive."""
    model = first_order_lowpass_tf()

    with pytest.raises(ValueError):
        simulate_step_response(model, t_span=(0.0, 5.0), num_points=0)


def test_provided_time_array_must_be_one_dimensional():
    """Explicit time samples must be one-dimensional."""
    model = first_order_lowpass_tf()

    with pytest.raises(ValueError):
        simulate_step_response(model, t=[[0.0, 1.0], [2.0, 3.0]])


def test_provided_time_array_must_be_increasing():
    """Explicit time samples must be strictly increasing."""
    model = first_order_lowpass_tf()

    with pytest.raises(ValueError):
        simulate_step_response(model, t=[0.0, 1.0, 1.0, 2.0])
