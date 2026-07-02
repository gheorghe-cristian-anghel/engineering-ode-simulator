"""Reusable finite difference utilities for one-dimensional grids."""

import numpy as np


def uniform_grid_1d(start=0.0, stop=1.0, num_points=101):
    """Return a uniform 1D grid and spacing ``(x, dx)``."""
    start = float(start)
    stop = float(stop)
    num_points = int(num_points)

    if stop <= start:
        raise ValueError("stop must be greater than start")

    if num_points < 2:
        raise ValueError("num_points must be at least 2")

    x = np.linspace(start, stop, num_points)
    dx = x[1] - x[0]

    return x, dx


def forward_difference(y, dx):
    """Approximate first derivative with forward differences.

    The returned array has the same shape as ``y``. Forward differences are
    used up to the second-to-last point, and a backward difference is used at
    the final point.
    """
    values = _validate_1d_values(y, min_points=2)
    dx = _validate_dx(dx)

    derivative = np.empty_like(values, dtype=float)
    derivative[:-1] = (values[1:] - values[:-1]) / dx
    derivative[-1] = (values[-1] - values[-2]) / dx

    return derivative


def backward_difference(y, dx):
    """Approximate first derivative with backward differences.

    The returned array has the same shape as ``y``. A forward difference is
    used at the first point, and backward differences are used elsewhere.
    """
    values = _validate_1d_values(y, min_points=2)
    dx = _validate_dx(dx)

    derivative = np.empty_like(values, dtype=float)
    derivative[0] = (values[1] - values[0]) / dx
    derivative[1:] = (values[1:] - values[:-1]) / dx

    return derivative


def central_difference(y, dx):
    """Approximate first derivative with central differences.

    The returned array has the same shape as ``y``. Central differences are
    used in the interior, with one-sided differences at the boundaries.
    """
    values = _validate_1d_values(y, min_points=3)
    dx = _validate_dx(dx)

    derivative = np.empty_like(values, dtype=float)
    derivative[0] = (values[1] - values[0]) / dx
    derivative[1:-1] = (values[2:] - values[:-2]) / (2.0 * dx)
    derivative[-1] = (values[-1] - values[-2]) / dx

    return derivative


def second_derivative_central(y, dx):
    """Approximate second derivative with central differences.

    The returned array has the same shape as ``y``. Central differences are
    used in the interior. Boundary values copy the nearest interior estimate.
    """
    values = _validate_1d_values(y, min_points=3)
    dx = _validate_dx(dx)

    derivative = np.empty_like(values, dtype=float)
    derivative[1:-1] = (values[2:] - 2.0 * values[1:-1] + values[:-2]) / dx**2
    derivative[0] = derivative[1]
    derivative[-1] = derivative[-2]

    return derivative


def first_derivative_matrix(num_points, dx, scheme="central"):
    """Return a dense matrix for first-derivative finite differences."""
    num_points = _validate_num_points(num_points, min_points=2)
    dx = _validate_dx(dx)
    scheme = str(scheme).lower()

    if scheme not in {"forward", "backward", "central"}:
        raise ValueError("scheme must be 'forward', 'backward', or 'central'")

    matrix = np.zeros((num_points, num_points), dtype=float)

    if scheme == "forward":
        for row in range(num_points - 1):
            matrix[row, row] = -1.0 / dx
            matrix[row, row + 1] = 1.0 / dx
        matrix[-1, -2] = -1.0 / dx
        matrix[-1, -1] = 1.0 / dx
        return matrix

    if scheme == "backward":
        matrix[0, 0] = -1.0 / dx
        matrix[0, 1] = 1.0 / dx
        for row in range(1, num_points):
            matrix[row, row - 1] = -1.0 / dx
            matrix[row, row] = 1.0 / dx
        return matrix

    if num_points < 3:
        raise ValueError("central scheme requires at least 3 points")

    matrix[0, 0] = -1.0 / dx
    matrix[0, 1] = 1.0 / dx
    for row in range(1, num_points - 1):
        matrix[row, row - 1] = -0.5 / dx
        matrix[row, row + 1] = 0.5 / dx
    matrix[-1, -2] = -1.0 / dx
    matrix[-1, -1] = 1.0 / dx

    return matrix


def second_derivative_matrix(num_points, dx):
    """Return a dense matrix for central second-derivative differences."""
    num_points = _validate_num_points(num_points, min_points=3)
    dx = _validate_dx(dx)

    matrix = np.zeros((num_points, num_points), dtype=float)

    for row in range(1, num_points - 1):
        matrix[row, row - 1] = 1.0 / dx**2
        matrix[row, row] = -2.0 / dx**2
        matrix[row, row + 1] = 1.0 / dx**2

    matrix[0] = matrix[1]
    matrix[-1] = matrix[-2]

    return matrix


def max_abs_error(numerical, exact):
    """Return the maximum absolute error between numerical and exact values."""
    numerical, exact = _validate_matching_arrays(numerical, exact)
    return float(np.max(np.abs(numerical - exact)))


def rms_error(numerical, exact):
    """Return the root-mean-square error between numerical and exact values."""
    numerical, exact = _validate_matching_arrays(numerical, exact)
    return float(np.sqrt(np.mean((numerical - exact) ** 2)))


def estimate_convergence_order(dx_values, error_values):
    """Estimate convergence order from ``log(error)`` versus ``log(dx)``."""
    dx_values = _validate_positive_1d_array(dx_values, "dx_values")
    error_values = _validate_positive_1d_array(error_values, "error_values")

    if len(dx_values) != len(error_values):
        raise ValueError("dx_values and error_values must have the same length")

    if len(dx_values) < 2:
        raise ValueError("at least two dx/error pairs are required")

    slope, _ = np.polyfit(np.log(dx_values), np.log(error_values), 1)

    return float(slope)


def _validate_1d_values(values, min_points):
    """Return a validated one-dimensional float array."""
    array = np.asarray(values, dtype=float)

    if array.ndim != 1:
        raise ValueError("y must be a one-dimensional array")

    if len(array) < min_points:
        raise ValueError(f"y must contain at least {min_points} points")

    if not np.all(np.isfinite(array)):
        raise ValueError("y must contain only finite values")

    return array


def _validate_dx(dx):
    """Return a validated grid spacing."""
    dx = float(dx)

    if dx <= 0:
        raise ValueError("dx must be positive")

    if not np.isfinite(dx):
        raise ValueError("dx must be finite")

    return dx


def _validate_num_points(num_points, min_points):
    """Return a validated number of grid points."""
    num_points = int(num_points)

    if num_points < min_points:
        raise ValueError(f"num_points must be at least {min_points}")

    return num_points


def _validate_matching_arrays(numerical, exact):
    """Return validated numerical and exact arrays with matching shapes."""
    numerical = np.asarray(numerical, dtype=float)
    exact = np.asarray(exact, dtype=float)

    if numerical.shape != exact.shape:
        raise ValueError("numerical and exact must have the same shape")

    if numerical.ndim == 0:
        raise ValueError("numerical and exact must be arrays")

    if not np.all(np.isfinite(numerical)):
        raise ValueError("numerical must contain only finite values")

    if not np.all(np.isfinite(exact)):
        raise ValueError("exact must contain only finite values")

    return numerical, exact


def _validate_positive_1d_array(values, name):
    """Return a positive one-dimensional array for convergence fitting."""
    array = np.asarray(values, dtype=float)

    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")

    if np.any(array <= 0):
        raise ValueError(f"{name} must contain only positive values")

    return array
