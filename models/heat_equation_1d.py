"""One-dimensional heat equation finite-difference solver.

This module solves the diffusion equation

    du/dt = alpha * d2u/dx2

with an explicit finite-difference method. It is intended as an educational
numerical-methods example, so the API exposes the grid spacing, time step, and
stability number used by the simulation.
"""

import numpy as np


def heat_stability_number(alpha, dt, dx):
    """Return the explicit heat-equation stability number ``r``.

    The explicit 1D heat equation scheme is stable for ``r <= 0.5``.
    """
    alpha = float(alpha)
    dt = float(dt)
    dx = float(dx)

    if alpha <= 0:
        raise ValueError("alpha must be positive")

    if dt <= 0:
        raise ValueError("dt must be positive")

    if dx <= 0:
        raise ValueError("dx must be positive")

    return alpha * dt / dx**2


def check_heat_stability(alpha, dt, dx, max_r=0.5, raise_error=False):
    """Check whether explicit heat-equation parameters satisfy stability.

    Parameters
    ----------
    alpha : float
        Thermal diffusivity.
    dt : float
        Time step.
    dx : float
        Spatial grid spacing.
    max_r : float, optional
        Maximum allowed stability number. For the explicit 1D scheme this is
        usually 0.5.
    raise_error : bool, optional
        If True, raise ``ValueError`` when the parameters are unstable.
        Otherwise return True or False.
    """
    max_r = float(max_r)

    if max_r <= 0:
        raise ValueError("max_r must be positive")

    r = heat_stability_number(alpha, dt, dx)
    is_stable = r <= max_r

    if raise_error and not is_stable:
        raise ValueError(
            "Explicit heat equation scheme is unstable: "
            f"r = {r:.6g}, but max_r = {max_r:.6g}"
        )

    return is_stable


def gaussian_initial_condition(x, center=0.5, width=0.08, amplitude=1.0):
    """Return a Gaussian pulse temperature profile."""
    x = np.asarray(x, dtype=float)
    center = float(center)
    width = float(width)
    amplitude = float(amplitude)

    if width <= 0:
        raise ValueError("width must be positive")

    return amplitude * np.exp(-((x - center) ** 2) / (2 * width**2))


def step_initial_condition(x, left_value=1.0, right_value=0.0, split=0.5):
    """Return a step temperature profile."""
    x = np.asarray(x, dtype=float)
    split = float(split)

    return np.where(x < split, float(left_value), float(right_value))


def sine_initial_condition(x, amplitude=1.0, mode=1, length=1.0):
    """Return a sine-wave temperature profile on a rod of given length."""
    x = np.asarray(x, dtype=float)
    mode = int(mode)
    length = float(length)

    if mode <= 0:
        raise ValueError("mode must be positive")

    if length <= 0:
        raise ValueError("length must be positive")

    return float(amplitude) * np.sin(mode * np.pi * x / length)


def simulate_heat_equation_1d(
    length=1.0,
    alpha=0.01,
    t_final=2.0,
    num_points=101,
    dt=None,
    initial_condition=None,
    boundary_type="dirichlet",
    boundary_values=(0.0, 0.0),
    enforce_stability=True,
):
    """Simulate the 1D heat equation using explicit finite differences.

    Parameters
    ----------
    length : float, optional
        Rod length in meters.
    alpha : float, optional
        Thermal diffusivity.
    t_final : float, optional
        Final simulation time in seconds.
    num_points : int, optional
        Number of spatial grid points.
    dt : float, optional
        Requested time step. If None, a stable time step is chosen
        automatically.
    initial_condition : callable or array-like, optional
        Initial temperature profile. Callables are evaluated as
        ``initial_condition(x)``. If None, a Gaussian pulse is used.
    boundary_type : {"dirichlet", "neumann"}, optional
        Boundary condition type. Dirichlet fixes end temperatures. Neumann
        uses zero-gradient insulated ends.
    boundary_values : tuple, optional
        Left and right boundary temperatures for Dirichlet conditions.
    enforce_stability : bool, optional
        If True, unstable explicit time steps raise ``ValueError``.

    Returns
    -------
    dict
        Dictionary containing ``x``, ``t``, ``temperature``, ``alpha``, ``dx``,
        ``dt``, ``stability_number``, ``boundary_type``, and
        ``boundary_values``.
    """
    length, alpha, t_final, num_points = _validate_simulation_parameters(
        length,
        alpha,
        t_final,
        num_points,
    )
    boundary_type = _validate_boundary_type(boundary_type)
    boundary_values = _validate_boundary_values(boundary_values)

    x = np.linspace(0.0, length, num_points)
    dx = x[1] - x[0]

    requested_dt = _choose_time_step(alpha, dx, dt)

    if enforce_stability:
        check_heat_stability(
            alpha,
            requested_dt,
            dx,
            raise_error=True,
        )

    num_intervals = max(1, int(np.ceil(t_final / requested_dt)))
    actual_dt = t_final / num_intervals
    t = np.linspace(0.0, t_final, num_intervals + 1)
    r = heat_stability_number(alpha, actual_dt, dx)

    temperature = np.empty((len(t), num_points), dtype=float)
    temperature[0] = _initial_temperature(
        x,
        length,
        initial_condition,
    )
    _apply_boundary_conditions(
        temperature[0],
        boundary_type,
        boundary_values,
    )

    for step_index in range(1, len(t)):
        previous = temperature[step_index - 1]
        current = previous.copy()

        current[1:-1] = previous[1:-1] + r * (
            previous[2:] - 2.0 * previous[1:-1] + previous[:-2]
        )
        _apply_boundary_conditions(current, boundary_type, boundary_values)
        temperature[step_index] = current

    return {
        "x": x,
        "t": t,
        "temperature": temperature,
        "alpha": alpha,
        "dx": dx,
        "dt": actual_dt,
        "stability_number": r,
        "boundary_type": boundary_type,
        "boundary_values": boundary_values,
    }


def _validate_simulation_parameters(length, alpha, t_final, num_points):
    """Return validated scalar simulation parameters."""
    length = float(length)
    alpha = float(alpha)
    t_final = float(t_final)
    num_points = int(num_points)

    if length <= 0:
        raise ValueError("length must be positive")

    if alpha <= 0:
        raise ValueError("alpha must be positive")

    if t_final <= 0:
        raise ValueError("t_final must be positive")

    if num_points < 3:
        raise ValueError("num_points must be at least 3")

    return length, alpha, t_final, num_points


def _choose_time_step(alpha, dx, dt):
    """Return a validated requested time step."""
    if dt is None:
        return 0.4 * dx**2 / alpha

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


def _initial_temperature(x, length, initial_condition):
    """Return a validated initial temperature profile."""
    if initial_condition is None:
        values = gaussian_initial_condition(x, center=0.5 * length)
    elif callable(initial_condition):
        values = initial_condition(x)
    else:
        values = initial_condition

    values = np.asarray(values, dtype=float)

    if values.shape != x.shape:
        raise ValueError("initial_condition must return one value per x point")

    if not np.all(np.isfinite(values)):
        raise ValueError("initial_condition must contain only finite values")

    return values


def _apply_boundary_conditions(temperature, boundary_type, boundary_values):
    """Apply boundary conditions to one temperature profile in-place."""
    if boundary_type == "dirichlet":
        temperature[0] = boundary_values[0]
        temperature[-1] = boundary_values[1]
        return

    temperature[0] = temperature[1]
    temperature[-1] = temperature[-2]
