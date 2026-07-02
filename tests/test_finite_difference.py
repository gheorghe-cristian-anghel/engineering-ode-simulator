import numpy as np
import pytest

from analysis.finite_difference import (
    backward_difference,
    central_difference,
    estimate_convergence_order,
    first_derivative_matrix,
    forward_difference,
    max_abs_error,
    rms_error,
    second_derivative_central,
    second_derivative_matrix,
    uniform_grid_1d,
)


def test_uniform_grid_1d_returns_correct_x_and_dx():
    """Uniform grid helper should return evenly spaced points and dx."""
    x, dx = uniform_grid_1d(0.0, 1.0, 6)

    assert x == pytest.approx([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    assert dx == pytest.approx(0.2)


def test_invalid_grid_parameters_raise_value_error():
    """Grid start/stop and point count should be validated."""
    with pytest.raises(ValueError):
        uniform_grid_1d(1.0, 0.0, 10)

    with pytest.raises(ValueError):
        uniform_grid_1d(0.0, 1.0, 1)


def test_forward_difference_returns_same_shape_as_input():
    """Forward difference helper should preserve input shape."""
    y = np.linspace(0.0, 1.0, 5)

    derivative = forward_difference(y, dx=0.25)

    assert derivative.shape == y.shape


def test_backward_difference_returns_same_shape_as_input():
    """Backward difference helper should preserve input shape."""
    y = np.linspace(0.0, 1.0, 5)

    derivative = backward_difference(y, dx=0.25)

    assert derivative.shape == y.shape


def test_central_difference_returns_same_shape_as_input():
    """Central difference helper should preserve input shape."""
    y = np.linspace(0.0, 1.0, 5)

    derivative = central_difference(y, dx=0.25)

    assert derivative.shape == y.shape


def test_second_derivative_central_returns_same_shape_as_input():
    """Second derivative helper should preserve input shape."""
    y = np.linspace(0.0, 1.0, 5)

    derivative = second_derivative_central(y, dx=0.25)

    assert derivative.shape == y.shape


def test_derivative_of_linear_function_is_constant():
    """Finite differences should exactly differentiate a linear function."""
    x, dx = uniform_grid_1d(-1.0, 1.0, 11)
    y = 3.0 * x - 2.0

    assert forward_difference(y, dx) == pytest.approx(np.full_like(x, 3.0))
    assert backward_difference(y, dx) == pytest.approx(np.full_like(x, 3.0))
    assert central_difference(y, dx) == pytest.approx(np.full_like(x, 3.0))


def test_second_derivative_of_quadratic_function_is_constant():
    """The central second derivative should exactly differentiate x^2."""
    x, dx = uniform_grid_1d(-1.0, 1.0, 21)
    y = 4.0 * x**2 + 3.0 * x - 1.0

    derivative = second_derivative_central(y, dx)

    assert derivative == pytest.approx(np.full_like(x, 8.0))


def test_central_difference_is_more_accurate_than_forward_for_sine():
    """Central difference should beat forward difference on a smooth sine."""
    x, dx = uniform_grid_1d(0.0, 1.0, 101)
    y = np.sin(2.0 * np.pi * x)
    exact = 2.0 * np.pi * np.cos(2.0 * np.pi * x)
    interior = slice(1, -1)

    forward_error = rms_error(forward_difference(y, dx)[interior], exact[interior])
    central_error = rms_error(central_difference(y, dx)[interior], exact[interior])

    assert central_error < forward_error


def test_max_abs_error_returns_expected_value():
    """Max absolute error should return the largest absolute difference."""
    numerical = np.array([1.0, 2.5, 4.0])
    exact = np.array([1.0, 2.0, 3.0])

    assert max_abs_error(numerical, exact) == pytest.approx(1.0)


def test_rms_error_returns_expected_value():
    """RMS error should return sqrt(mean(error^2))."""
    numerical = np.array([1.0, 3.0, 5.0])
    exact = np.array([1.0, 1.0, 1.0])

    assert rms_error(numerical, exact) == pytest.approx(np.sqrt(20.0 / 3.0))


def test_estimate_convergence_order_returns_one_for_linear_errors():
    """Errors proportional to dx should have order approximately one."""
    dx_values = np.array([0.2, 0.1, 0.05, 0.025])
    errors = 3.0 * dx_values

    assert estimate_convergence_order(dx_values, errors) == pytest.approx(1.0)


def test_estimate_convergence_order_returns_two_for_quadratic_errors():
    """Errors proportional to dx^2 should have order approximately two."""
    dx_values = np.array([0.2, 0.1, 0.05, 0.025])
    errors = 3.0 * dx_values**2

    assert estimate_convergence_order(dx_values, errors) == pytest.approx(2.0)


def test_invalid_dx_raises_value_error():
    """Derivative helpers should reject nonpositive dx."""
    y = np.linspace(0.0, 1.0, 5)

    with pytest.raises(ValueError):
        forward_difference(y, dx=0.0)

    with pytest.raises(ValueError):
        central_difference(y, dx=-1.0)


def test_non_1d_y_raises_value_error():
    """Derivative helpers should reject multidimensional inputs."""
    y = np.ones((2, 3))

    with pytest.raises(ValueError):
        forward_difference(y, dx=0.1)


def test_mismatched_error_inputs_raise_value_error():
    """Error helpers should require matching input shapes."""
    with pytest.raises(ValueError):
        max_abs_error([1.0, 2.0], [1.0])

    with pytest.raises(ValueError):
        rms_error([1.0, 2.0], [1.0])


def test_invalid_convergence_inputs_raise_value_error():
    """Convergence helper should validate positive matching arrays."""
    with pytest.raises(ValueError):
        estimate_convergence_order([0.1, 0.05], [0.01])

    with pytest.raises(ValueError):
        estimate_convergence_order([0.1, 0.05], [0.01, 0.0])


def test_first_derivative_matrix_has_correct_shape():
    """First derivative matrix should be square with grid size dimension."""
    matrix = first_derivative_matrix(num_points=5, dx=0.25)

    assert matrix.shape == (5, 5)


def test_second_derivative_matrix_has_correct_shape():
    """Second derivative matrix should be square with grid size dimension."""
    matrix = second_derivative_matrix(num_points=5, dx=0.25)

    assert matrix.shape == (5, 5)


def test_first_derivative_matrix_differentiates_linear_function():
    """First derivative matrix should exactly differentiate a linear function."""
    x, dx = uniform_grid_1d(0.0, 1.0, 6)
    y = 2.0 * x + 1.0
    matrix = first_derivative_matrix(num_points=len(x), dx=dx)

    assert matrix @ y == pytest.approx(np.full_like(x, 2.0))


def test_second_derivative_matrix_differentiates_quadratic_function():
    """Second derivative matrix should exactly differentiate x^2."""
    x, dx = uniform_grid_1d(0.0, 1.0, 6)
    y = 3.0 * x**2 + 2.0 * x + 1.0
    matrix = second_derivative_matrix(num_points=len(x), dx=dx)

    assert matrix @ y == pytest.approx(np.full_like(x, 6.0))
