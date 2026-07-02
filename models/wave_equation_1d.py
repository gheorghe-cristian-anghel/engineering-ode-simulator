"""One-dimensional wave equation finite-difference solver.

This module solves the wave equation

    u_tt = c^2 * u_xx

with an explicit central finite-difference method. It is intended as an
educational numerical-methods example, so the API exposes the grid spacing,
time step, and CFL number used by the simulation.
"""

import numpy as np


def wave_cfl_number(c, dt, dx):
    """Return the explicit wave-equation CFL number ``lambda``."""
    c = float(c)
    dt = float(dt)
    dx = float(dx)

    if c <= 0:
        raise ValueError("c must be positive")

    if dt <= 0:
        raise ValueError("dt must be positive")

    if dx <= 0:
        raise ValueError("dx must be positive")

    return c * dt / dx


def check_wave_stability(c, dt, dx, max_cfl=1.0, raise_error=False):
    """Check whether explicit wave-equation parameters satisfy CFL stability."""
    max_cfl = float(max_cfl)

    if max_cfl <= 0:
        raise ValueError("max_cfl must be positive")

    cfl = wave_cfl_number(c, dt, dx)
    is_stable = cfl <= max_cfl

    if raise_error and not is_stable:
        raise ValueError(
            "Explicit wave equation scheme is unstable: "
            f"CFL = {cfl:.6g}, but max_cfl = {max_cfl:.6g}"
        )

    return is_stable


def gaussian_displacement(x, center=0.5, width=0.08, amplitude=1.0):
    """Return a Gaussian initial displacement profile."""
    x = np.asarray(x, dtype=float)
    center = float(center)
    width = float(width)
    amplitude = float(amplitude)

    if width <= 0:
        raise ValueError("width must be positive")

    return amplitude * np.exp(-((x - center) ** 2) / (2 * width**2))


def sine_displacement(x, amplitude=1.0, mode=1, length=1.0):
    """Return a fixed-end sine-mode initial displacement profile."""
    x = np.asarray(x, dtype=float)
    mode = int(mode)
    length = float(length)

    if mode <= 0:
        raise ValueError("mode must be positive")

    if length <= 0:
        raise ValueError("length must be positive")

    return float(amplitude) * np.sin(mode * np.pi * x / length)


def triangular_displacement(x, center=0.5, width=0.4, amplitude=1.0):
    """Return a triangular plucked-string displacement profile."""
    x = np.asarray(x, dtype=float)
    center = float(center)
    width = float(width)
    amplitude = float(amplitude)

    if width <= 0:
        raise ValueError("width must be positive")

    half_width = 0.5 * width
    distance = np.abs(x - center)
    shape = np.maximum(1.0 - distance / half_width, 0.0)

    return amplitude * shape


def zero_initial_velocity(x):
    """Return a zero initial velocity profile."""
    x = np.asarray(x, dtype=float)
    return np.zeros_like(x, dtype=float)


def simulate_wave_equation_1d(
    length=1.0,
    c=1.0,
    t_final=2.0,
    num_points=201,
    dt=None,
    initial_displacement=None,
    initial_velocity=None,
    boundary_type="dirichlet",
    boundary_values=(0.0, 0.0),
    enforce_stability=True,
):
    """Simulate the 1D wave equation using explicit finite differences.

    Parameters
    ----------
    length : float, optional
        String or rod length in meters.
    c : float, optional
        Wave speed in meters per second.
    t_final : float, optional
        Final simulation time in seconds.
    num_points : int, optional
        Number of spatial grid points.
    dt : float, optional
        Requested time step. If None, a stable time step is chosen
        automatically.
    initial_displacement : callable or array-like, optional
        Initial displacement profile. Callables are evaluated as
        ``initial_displacement(x)``. If None, a Gaussian pulse is used.
    initial_velocity : callable or array-like, optional
        Initial velocity profile. Callables are evaluated as
        ``initial_velocity(x)``. If None, zero velocity is used.
    boundary_type : {"dirichlet", "neumann"}, optional
        Boundary condition type. Dirichlet fixes end displacement. Neumann
        uses zero-gradient free ends.
    boundary_values : tuple, optional
        Left and right boundary displacements for Dirichlet conditions.
    enforce_stability : bool, optional
        If True, unstable explicit time steps raise ``ValueError``.

    Returns
    -------
    dict
        Dictionary containing ``x``, ``t``, ``displacement``, ``c``, ``dx``,
        ``dt``, ``cfl_number``, ``boundary_type``, and ``boundary_values``.
    """
    length, c, t_final, num_points = _validate_simulation_parameters(
        length,
        c,
        t_final,
        num_points,
    )
    boundary_type = _validate_boundary_type(boundary_type)
    boundary_values = _validate_boundary_values(boundary_values)

    x = np.linspace(0.0, length, num_points)
    dx = x[1] - x[0]
    requested_dt = _choose_time_step(c, dx, dt)

    if enforce_stability:
        check_wave_stability(c, requested_dt, dx, raise_error=True)

    num_intervals = max(1, int(np.ceil(t_final / requested_dt)))
    actual_dt = t_final / num_intervals
    t = np.linspace(0.0, t_final, num_intervals + 1)
    cfl = wave_cfl_number(c, actual_dt, dx)
    cfl_squared = cfl**2

    displacement = np.empty((len(t), num_points), dtype=float)
    displacement[0] = _initial_profile(
        x,
        initial_displacement,
        default_profile=lambda grid: gaussian_displacement(
            grid,
            center=0.5 * length,
        ),
        name="initial_displacement",
    )
    velocity = _initial_profile(
        x,
        initial_velocity,
        default_profile=zero_initial_velocity,
        name="initial_velocity",
    )
    _apply_boundary_conditions(
        displacement[0],
        boundary_type,
        boundary_values,
    )

    if len(t) > 1:
        displacement[1] = displacement[0].copy()
        previous = displacement[0]
        displacement[1][1:-1] = (
            previous[1:-1]
            + actual_dt * velocity[1:-1]
            + 0.5
            * cfl_squared
            * (previous[2:] - 2.0 * previous[1:-1] + previous[:-2])
        )
        _apply_boundary_conditions(
            displacement[1],
            boundary_type,
            boundary_values,
        )

    for step_index in range(2, len(t)):
        current = displacement[step_index - 1]
        previous = displacement[step_index - 2]
        next_profile = current.copy()

        next_profile[1:-1] = (
            2.0 * current[1:-1]
            - previous[1:-1]
            + cfl_squared * (current[2:] - 2.0 * current[1:-1] + current[:-2])
        )
        _apply_boundary_conditions(
            next_profile,
            boundary_type,
            boundary_values,
        )
        displacement[step_index] = next_profile

    return {
        "x": x,
        "t": t,
        "displacement": displacement,
        "c": c,
        "dx": dx,
        "dt": actual_dt,
        "cfl_number": cfl,
        "boundary_type": boundary_type,
        "boundary_values": boundary_values,
    }


def _validate_simulation_parameters(length, c, t_final, num_points):
    """Return validated scalar simulation parameters."""
    length = float(length)
    c = float(c)
    t_final = float(t_final)
    num_points = int(num_points)

    if length <= 0:
        raise ValueError("length must be positive")

    if c <= 0:
        raise ValueError("c must be positive")

    if t_final <= 0:
        raise ValueError("t_final must be positive")

    if num_points < 3:
        raise ValueError("num_points must be at least 3")

    return length, c, t_final, num_points


def _choose_time_step(c, dx, dt):
    """Return a validated requested time step."""
    if dt is None:
        return 0.8 * dx / c

    dt = float(dt)

    if dt <= 0:
        raise ValueError("dt must be positive")

    return dt


def _validate_boundary_type(boundary_type):
    """Return a normalized boundary type."""
    boundary_type = str(boundary_type).lower()

    if boundary_type not in {"dirichlet", "neumann"}:
        raise ValueError("boundary_type must be 'dirichlet' or 'neumann'")

    return boundary_type


def _validate_boundary_values(boundary_values):
    """Return validated left and right boundary values."""
    if len(boundary_values) != 2:
        raise ValueError("boundary_values must contain two values")

    left, right = float(boundary_values[0]), float(boundary_values[1])

    if not np.isfinite([left, right]).all():
        raise ValueError("boundary_values must be finite")

    return (left, right)


def _initial_profile(x, values_or_function, default_profile, name):
    """Return a validated initial displacement or velocity profile."""
    if values_or_function is None:
        values = default_profile(x)
    elif callable(values_or_function):
        values = values_or_function(x)
    else:
        values = values_or_function

    values = np.asarray(values, dtype=float)

    if values.shape != x.shape:
        raise ValueError(f"{name} must return one value per x point")

    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")

    return values


def _apply_boundary_conditions(displacement, boundary_type, boundary_values):
    """Apply boundary conditions to one displacement profile in-place."""
    if boundary_type == "dirichlet":
        displacement[0] = boundary_values[0]
        displacement[-1] = boundary_values[1]
        return

    displacement[0] = displacement[1]
    displacement[-1] = displacement[-2]
