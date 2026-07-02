"""Two-dimensional heat equation finite-difference solver.

This module solves the diffusion equation

    u_t = alpha * (u_xx + u_yy)

on a rectangular plate with an explicit finite-difference method. The grid is
created with ``X, Y = np.meshgrid(x, y)``, so temperature fields have shape
``(ny, nx)``.
"""

import numpy as np


def heat_stability_numbers_2d(alpha, dt, dx, dy):
    """Return ``(rx, ry, rx + ry)`` for the explicit 2D heat scheme."""
    alpha = float(alpha)
    dt = float(dt)
    dx = float(dx)
    dy = float(dy)

    if alpha <= 0:
        raise ValueError("alpha must be positive")

    if dt <= 0:
        raise ValueError("dt must be positive")

    if dx <= 0:
        raise ValueError("dx must be positive")

    if dy <= 0:
        raise ValueError("dy must be positive")

    if not np.isfinite([alpha, dt, dx, dy]).all():
        raise ValueError("alpha, dt, dx, and dy must be finite")

    rx = alpha * dt / dx**2
    ry = alpha * dt / dy**2

    return rx, ry, rx + ry


def check_heat_stability_2d(
    alpha,
    dt,
    dx,
    dy,
    max_sum=0.5,
    raise_error=False,
):
    """Check whether explicit 2D heat-equation parameters are stable."""
    max_sum = float(max_sum)

    if max_sum <= 0:
        raise ValueError("max_sum must be positive")

    if not np.isfinite(max_sum):
        raise ValueError("max_sum must be finite")

    rx, ry, stability_sum = heat_stability_numbers_2d(alpha, dt, dx, dy)
    is_stable = stability_sum <= max_sum

    if raise_error and not is_stable:
        raise ValueError(
            "Explicit 2D heat equation scheme is unstable: "
            f"rx + ry = {stability_sum:.6g}, but max_sum = {max_sum:.6g} "
            f"(rx = {rx:.6g}, ry = {ry:.6g})"
        )

    return is_stable


def create_2d_grid(width=1.0, height=1.0, nx=51, ny=51):
    """Return ``x, y, X, Y, dx, dy`` for a uniform rectangular grid."""
    width = float(width)
    height = float(height)
    nx = int(nx)
    ny = int(ny)

    if width <= 0:
        raise ValueError("width must be positive")

    if height <= 0:
        raise ValueError("height must be positive")

    if nx < 3:
        raise ValueError("nx must be at least 3")

    if ny < 3:
        raise ValueError("ny must be at least 3")

    if not np.isfinite([width, height]).all():
        raise ValueError("width and height must be finite")

    x = np.linspace(0.0, width, nx)
    y = np.linspace(0.0, height, ny)
    X, Y = np.meshgrid(x, y)
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    return x, y, X, Y, dx, dy


def gaussian_hotspot_2d(
    X,
    Y,
    center=(0.5, 0.5),
    width=0.1,
    amplitude=1.0,
):
    """Return a Gaussian hot spot on a 2D grid."""
    X, Y = _validate_grid_arrays(X, Y)
    center = _validate_pair(center, "center")
    width = float(width)
    amplitude = float(amplitude)

    if width <= 0:
        raise ValueError("width must be positive")

    if not np.isfinite([width, amplitude]).all():
        raise ValueError("width and amplitude must be finite")

    radius_squared = (X - center[0]) ** 2 + (Y - center[1]) ** 2
    return amplitude * np.exp(-radius_squared / (2.0 * width**2))


def rectangular_hot_region_2d(
    X,
    Y,
    x_range=(0.4, 0.6),
    y_range=(0.4, 0.6),
    value=1.0,
    background=0.0,
):
    """Return a rectangular hot region on a 2D grid."""
    X, Y = _validate_grid_arrays(X, Y)
    x_min, x_max = _validate_range(x_range, "x_range")
    y_min, y_max = _validate_range(y_range, "y_range")
    value = float(value)
    background = float(background)

    if not np.isfinite([value, background]).all():
        raise ValueError("value and background must be finite")

    temperature = np.full_like(X, background, dtype=float)
    mask = (X >= x_min) & (X <= x_max) & (Y >= y_min) & (Y <= y_max)
    temperature[mask] = value

    return temperature


def sine_initial_condition_2d(
    X,
    Y,
    amplitude=1.0,
    mode_x=1,
    mode_y=1,
    width=1.0,
    height=1.0,
):
    """Return a smooth sine mode temperature field on a rectangular plate."""
    X, Y = _validate_grid_arrays(X, Y)
    amplitude = float(amplitude)
    mode_x = int(mode_x)
    mode_y = int(mode_y)
    width = float(width)
    height = float(height)

    if mode_x <= 0:
        raise ValueError("mode_x must be positive")

    if mode_y <= 0:
        raise ValueError("mode_y must be positive")

    if width <= 0:
        raise ValueError("width must be positive")

    if height <= 0:
        raise ValueError("height must be positive")

    if not np.isfinite([amplitude, width, height]).all():
        raise ValueError("amplitude, width, and height must be finite")

    return amplitude * np.sin(mode_x * np.pi * X / width) * np.sin(
        mode_y * np.pi * Y / height
    )


def apply_2d_boundary_conditions(
    U,
    boundary_type="dirichlet",
    boundary_values=0.0,
):
    """Apply 2D boundary conditions to a temperature field in-place."""
    boundary_type = _validate_boundary_type(boundary_type)
    U = _validate_temperature_field(U, "U")

    if boundary_type == "dirichlet":
        left, right, bottom, top = _validate_boundary_values(boundary_values)
        U[:, 0] = left
        U[:, -1] = right
        U[0, :] = bottom
        U[-1, :] = top
        return U

    U[:, 0] = U[:, 1]
    U[:, -1] = U[:, -2]
    U[0, :] = U[1, :]
    U[-1, :] = U[-2, :]
    return U


def simulate_heat_equation_2d(
    width=1.0,
    height=1.0,
    alpha=0.01,
    t_final=1.0,
    nx=51,
    ny=51,
    dt=None,
    initial_condition=None,
    boundary_type="dirichlet",
    boundary_values=0.0,
    enforce_stability=True,
    store_every=1,
):
    """Simulate the 2D heat equation using explicit finite differences."""
    alpha = float(alpha)
    t_final = float(t_final)
    store_every = int(store_every)

    if alpha <= 0:
        raise ValueError("alpha must be positive")

    if t_final <= 0:
        raise ValueError("t_final must be positive")

    if store_every < 1:
        raise ValueError("store_every must be at least 1")

    if not np.isfinite([alpha, t_final]).all():
        raise ValueError("alpha and t_final must be finite")

    boundary_type = _validate_boundary_type(boundary_type)
    x, y, X, Y, dx, dy = create_2d_grid(width, height, nx, ny)
    requested_dt = _choose_time_step(alpha, dx, dy, dt)

    if enforce_stability:
        check_heat_stability_2d(
            alpha,
            requested_dt,
            dx,
            dy,
            raise_error=True,
        )

    num_intervals = max(1, int(np.ceil(t_final / requested_dt)))
    actual_dt = t_final / num_intervals
    rx, ry, stability_sum = heat_stability_numbers_2d(alpha, actual_dt, dx, dy)

    if enforce_stability:
        check_heat_stability_2d(alpha, actual_dt, dx, dy, raise_error=True)

    initial_temperature = _initial_temperature_2d(
        X,
        Y,
        initial_condition,
        width,
        height,
    )
    apply_2d_boundary_conditions(
        initial_temperature,
        boundary_type,
        boundary_values,
    )

    stored_times = [0.0]
    stored_temperatures = [initial_temperature.copy()]
    current = initial_temperature

    for step_index in range(1, num_intervals + 1):
        previous = current
        next_temperature = previous.copy()

        next_temperature[1:-1, 1:-1] = (
            previous[1:-1, 1:-1]
            + rx
            * (
                previous[1:-1, 2:]
                - 2.0 * previous[1:-1, 1:-1]
                + previous[1:-1, :-2]
            )
            + ry
            * (
                previous[2:, 1:-1]
                - 2.0 * previous[1:-1, 1:-1]
                + previous[:-2, 1:-1]
            )
        )
        apply_2d_boundary_conditions(
            next_temperature,
            boundary_type,
            boundary_values,
        )
        current = next_temperature

        if step_index % store_every == 0 or step_index == num_intervals:
            stored_times.append(step_index * actual_dt)
            stored_temperatures.append(current.copy())

    return {
        "x": x,
        "y": y,
        "X": X,
        "Y": Y,
        "t": np.array(stored_times, dtype=float),
        "temperature": np.array(stored_temperatures, dtype=float),
        "alpha": alpha,
        "dx": dx,
        "dy": dy,
        "dt": actual_dt,
        "rx": rx,
        "ry": ry,
        "stability_sum": stability_sum,
        "boundary_type": boundary_type,
        "boundary_values": boundary_values,
        "store_every": store_every,
    }


def _choose_time_step(alpha, dx, dy, dt):
    """Return a validated requested time step."""
    if dt is None:
        return 0.4 / (alpha * (1.0 / dx**2 + 1.0 / dy**2))

    dt = float(dt)

    if dt <= 0:
        raise ValueError("dt must be positive")

    if not np.isfinite(dt):
        raise ValueError("dt must be finite")

    return dt


def _initial_temperature_2d(X, Y, initial_condition, width, height):
    """Return a validated initial temperature field."""
    if initial_condition is None:
        values = gaussian_hotspot_2d(
            X,
            Y,
            center=(0.5 * width, 0.5 * height),
            width=0.1 * min(width, height),
        )
    elif callable(initial_condition):
        values = initial_condition(X, Y)
    else:
        values = initial_condition

    values = np.asarray(values, dtype=float)

    if values.shape != X.shape:
        raise ValueError("initial_condition must return an array with shape (ny, nx)")

    if not np.all(np.isfinite(values)):
        raise ValueError("initial_condition must contain only finite values")

    return values


def _validate_boundary_type(boundary_type):
    """Return a normalized boundary type."""
    boundary_type = str(boundary_type).lower()

    if boundary_type not in {"dirichlet", "neumann"}:
        raise ValueError("boundary_type must be 'dirichlet' or 'neumann'")

    return boundary_type


def _validate_boundary_values(boundary_values):
    """Return boundary values as ``(left, right, bottom, top)``."""
    values = np.asarray(boundary_values, dtype=float)

    if values.ndim == 0:
        values = np.full(4, float(values), dtype=float)
    elif values.ndim == 1 and len(values) == 4:
        values = values.astype(float)
    else:
        raise ValueError(
            "boundary_values must be a scalar or four values "
            "(left, right, bottom, top)"
        )

    if not np.all(np.isfinite(values)):
        raise ValueError("boundary_values must contain only finite values")

    return tuple(values)


def _validate_grid_arrays(X, Y):
    """Return validated matching 2D grid arrays."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    if X.shape != Y.shape:
        raise ValueError("X and Y must have the same shape")

    if X.ndim != 2:
        raise ValueError("X and Y must be two-dimensional")

    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain only finite values")

    if not np.all(np.isfinite(Y)):
        raise ValueError("Y must contain only finite values")

    return X, Y


def _validate_temperature_field(U, name):
    """Return a validated 2D temperature field."""
    U = np.asarray(U, dtype=float)

    if U.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")

    if U.shape[0] < 3 or U.shape[1] < 3:
        raise ValueError(f"{name} must have at least 3 rows and 3 columns")

    if not np.all(np.isfinite(U)):
        raise ValueError(f"{name} must contain only finite values")

    return U


def _validate_pair(values, name):
    """Return two finite values."""
    values = np.asarray(values, dtype=float)

    if values.shape != (2,):
        raise ValueError(f"{name} must contain two values")

    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")

    return tuple(values)


def _validate_range(values, name):
    """Return a finite increasing range."""
    value_min, value_max = _validate_pair(values, name)

    if value_max <= value_min:
        raise ValueError(f"{name} maximum must be greater than minimum")

    return value_min, value_max
