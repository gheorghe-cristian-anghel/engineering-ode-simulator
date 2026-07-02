"""Two-dimensional wave equation finite-difference solver.

This module solves the membrane wave equation

    u_tt = c^2 * (u_xx + u_yy)

on a rectangular domain with an explicit central finite-difference method.
The grid is created with ``X, Y = np.meshgrid(x, y)``, so displacement fields
have shape ``(ny, nx)``.
"""

import numpy as np


def wave_stability_numbers_2d(c, dt, dx, dy):
    """Return ``(lambda_x, lambda_y, rx, ry, rx + ry)`` for the 2D wave scheme."""
    c = float(c)
    dt = float(dt)
    dx = float(dx)
    dy = float(dy)

    if c <= 0:
        raise ValueError("c must be positive")

    if dt <= 0:
        raise ValueError("dt must be positive")

    if dx <= 0:
        raise ValueError("dx must be positive")

    if dy <= 0:
        raise ValueError("dy must be positive")

    if not np.isfinite([c, dt, dx, dy]).all():
        raise ValueError("c, dt, dx, and dy must be finite")

    lambda_x = c * dt / dx
    lambda_y = c * dt / dy
    rx = lambda_x**2
    ry = lambda_y**2

    return lambda_x, lambda_y, rx, ry, rx + ry


def check_wave_stability_2d(
    c,
    dt,
    dx,
    dy,
    max_sum=1.0,
    raise_error=False,
):
    """Check whether explicit 2D wave-equation parameters are stable."""
    max_sum = float(max_sum)

    if max_sum <= 0:
        raise ValueError("max_sum must be positive")

    if not np.isfinite(max_sum):
        raise ValueError("max_sum must be finite")

    lambda_x, lambda_y, rx, ry, stability_sum = wave_stability_numbers_2d(
        c,
        dt,
        dx,
        dy,
    )
    is_stable = stability_sum <= max_sum

    if raise_error and not is_stable:
        raise ValueError(
            "Explicit 2D wave equation scheme is unstable: "
            f"rx + ry = {stability_sum:.6g}, but max_sum = {max_sum:.6g} "
            f"(lambda_x = {lambda_x:.6g}, lambda_y = {lambda_y:.6g}, "
            f"rx = {rx:.6g}, ry = {ry:.6g})"
        )

    return is_stable


def create_2d_grid(width=1.0, height=1.0, nx=61, ny=61):
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


def gaussian_displacement_2d(
    X,
    Y,
    center=(0.5, 0.5),
    width=0.08,
    amplitude=1.0,
):
    """Return a Gaussian initial displacement on a 2D grid."""
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


def circular_ring_displacement_2d(
    X,
    Y,
    center=(0.5, 0.5),
    radius=0.25,
    width=0.04,
    amplitude=1.0,
):
    """Return a circular ring displacement pulse on a 2D grid."""
    X, Y = _validate_grid_arrays(X, Y)
    center = _validate_pair(center, "center")
    radius = float(radius)
    width = float(width)
    amplitude = float(amplitude)

    if radius < 0:
        raise ValueError("radius must be nonnegative")

    if width <= 0:
        raise ValueError("width must be positive")

    if not np.isfinite([radius, width, amplitude]).all():
        raise ValueError("radius, width, and amplitude must be finite")

    distance = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    return amplitude * np.exp(-((distance - radius) ** 2) / (2.0 * width**2))


def sine_displacement_2d(
    X,
    Y,
    amplitude=1.0,
    mode_x=1,
    mode_y=1,
    width=1.0,
    height=1.0,
):
    """Return a fixed-edge sine-mode displacement field."""
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


def zero_initial_velocity_2d(X, Y):
    """Return a zero initial velocity field with shape ``(ny, nx)``."""
    X, Y = _validate_grid_arrays(X, Y)
    return np.zeros_like(X, dtype=float)


def apply_2d_wave_boundary_conditions(
    U,
    boundary_type="dirichlet",
    boundary_values=0.0,
):
    """Apply fixed 2D wave boundary conditions to a displacement field in-place."""
    boundary_type = _validate_boundary_type(boundary_type)
    U = _validate_displacement_field(U, "U")

    left, right, bottom, top = _validate_boundary_values(boundary_values)
    U[:, 0] = left
    U[:, -1] = right
    U[0, :] = bottom
    U[-1, :] = top

    return U


def simulate_wave_equation_2d(
    width=1.0,
    height=1.0,
    c=1.0,
    t_final=1.0,
    nx=61,
    ny=61,
    dt=None,
    initial_displacement=None,
    initial_velocity=None,
    boundary_type="dirichlet",
    boundary_values=0.0,
    enforce_stability=True,
    store_every=1,
):
    """Simulate the 2D wave equation using explicit finite differences."""
    c = float(c)
    t_final = float(t_final)
    store_every = int(store_every)

    if c <= 0:
        raise ValueError("c must be positive")

    if t_final <= 0:
        raise ValueError("t_final must be positive")

    if store_every < 1:
        raise ValueError("store_every must be at least 1")

    if not np.isfinite([c, t_final]).all():
        raise ValueError("c and t_final must be finite")

    boundary_type = _validate_boundary_type(boundary_type)
    boundary_values = _validate_boundary_values(boundary_values)
    x, y, X, Y, dx, dy = create_2d_grid(width, height, nx, ny)
    requested_dt = _choose_time_step(c, dx, dy, dt)

    if enforce_stability:
        check_wave_stability_2d(
            c,
            requested_dt,
            dx,
            dy,
            raise_error=True,
        )

    num_intervals = max(1, int(np.ceil(t_final / requested_dt)))
    actual_dt = t_final / num_intervals
    lambda_x, lambda_y, rx, ry, stability_sum = wave_stability_numbers_2d(
        c,
        actual_dt,
        dx,
        dy,
    )

    if enforce_stability:
        check_wave_stability_2d(c, actual_dt, dx, dy, raise_error=True)

    initial_u = _initial_field_2d(
        X,
        Y,
        initial_displacement,
        default_field=lambda grid_x, grid_y: gaussian_displacement_2d(
            grid_x,
            grid_y,
            center=(0.5 * width, 0.5 * height),
            width=0.08 * min(width, height),
        ),
        name="initial_displacement",
    )
    initial_v = _initial_field_2d(
        X,
        Y,
        initial_velocity,
        default_field=zero_initial_velocity_2d,
        name="initial_velocity",
    )

    apply_2d_wave_boundary_conditions(
        initial_u,
        boundary_type,
        boundary_values,
    )

    stored_times = [0.0]
    stored_displacements = [initial_u.copy()]

    previous = initial_u
    current = initial_u.copy()

    if num_intervals >= 1:
        current = previous.copy()
        current[1:-1, 1:-1] = (
            previous[1:-1, 1:-1]
            + actual_dt * initial_v[1:-1, 1:-1]
            + 0.5
            * (
                rx
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
        )
        apply_2d_wave_boundary_conditions(
            current,
            boundary_type,
            boundary_values,
        )

        if 1 % store_every == 0 or num_intervals == 1:
            stored_times.append(actual_dt)
            stored_displacements.append(current.copy())

    for step_index in range(2, num_intervals + 1):
        next_displacement = current.copy()

        next_displacement[1:-1, 1:-1] = (
            2.0 * current[1:-1, 1:-1]
            - previous[1:-1, 1:-1]
            + rx
            * (
                current[1:-1, 2:]
                - 2.0 * current[1:-1, 1:-1]
                + current[1:-1, :-2]
            )
            + ry
            * (
                current[2:, 1:-1]
                - 2.0 * current[1:-1, 1:-1]
                + current[:-2, 1:-1]
            )
        )
        apply_2d_wave_boundary_conditions(
            next_displacement,
            boundary_type,
            boundary_values,
        )

        previous, current = current, next_displacement

        if step_index % store_every == 0 or step_index == num_intervals:
            stored_times.append(step_index * actual_dt)
            stored_displacements.append(current.copy())

    return {
        "x": x,
        "y": y,
        "X": X,
        "Y": Y,
        "t": np.array(stored_times, dtype=float),
        "displacement": np.array(stored_displacements, dtype=float),
        "c": c,
        "dx": dx,
        "dy": dy,
        "dt": actual_dt,
        "lambda_x": lambda_x,
        "lambda_y": lambda_y,
        "rx": rx,
        "ry": ry,
        "stability_sum": stability_sum,
        "boundary_type": boundary_type,
        "boundary_values": boundary_values,
        "store_every": store_every,
    }


def _choose_time_step(c, dx, dy, dt):
    """Return a validated requested time step."""
    if dt is None:
        return 0.8 / (c * np.sqrt(1.0 / dx**2 + 1.0 / dy**2))

    dt = float(dt)

    if dt <= 0:
        raise ValueError("dt must be positive")

    if not np.isfinite(dt):
        raise ValueError("dt must be finite")

    return dt


def _initial_field_2d(X, Y, values_or_function, default_field, name):
    """Return a validated initial displacement or velocity field."""
    if values_or_function is None:
        values = default_field(X, Y)
    elif callable(values_or_function):
        values = values_or_function(X, Y)
    else:
        values = values_or_function

    values = np.asarray(values, dtype=float)

    if values.shape != X.shape:
        raise ValueError(f"{name} must return an array with shape (ny, nx)")

    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")

    return values


def _validate_boundary_type(boundary_type):
    """Return a normalized boundary type."""
    boundary_type = str(boundary_type).lower()

    if boundary_type != "dirichlet":
        raise ValueError("boundary_type must be 'dirichlet'")

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

    return tuple(float(value) for value in values)


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


def _validate_displacement_field(U, name):
    """Return a validated 2D displacement field."""
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

    return tuple(float(value) for value in values)
