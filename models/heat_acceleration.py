"""Optional Numba kernels for the explicit two-dimensional heat solver.

The public solver remains usable without Numba.  This module contains only the
time-stepping kernel so validation and result construction stay in Python.
"""

import numpy as np

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised on installations without extras
    NUMBA_AVAILABLE = False


if NUMBA_AVAILABLE:

    @njit(cache=True)
    def _advance_dirichlet(current, next_temperature, rx, ry, steps, boundaries):
        for _ in range(steps):
            for row in range(1, current.shape[0] - 1):
                for column in range(1, current.shape[1] - 1):
                    next_temperature[row, column] = (
                        current[row, column]
                        + rx
                        * (
                            current[row, column + 1]
                            - 2.0 * current[row, column]
                            + current[row, column - 1]
                        )
                        + ry
                        * (
                            current[row + 1, column]
                            - 2.0 * current[row, column]
                            + current[row - 1, column]
                        )
                    )
            for row in range(1, current.shape[0] - 1):
                next_temperature[row, 0] = boundaries[0]
                next_temperature[row, -1] = boundaries[1]
            for column in range(current.shape[1]):
                next_temperature[0, column] = boundaries[2]
                next_temperature[-1, column] = boundaries[3]
            current, next_temperature = next_temperature, current
        return current


    @njit(cache=True)
    def _advance_neumann(current, next_temperature, rx, ry, steps):
        for _ in range(steps):
            for row in range(1, current.shape[0] - 1):
                for column in range(1, current.shape[1] - 1):
                    next_temperature[row, column] = (
                        current[row, column]
                        + rx
                        * (
                            current[row, column + 1]
                            - 2.0 * current[row, column]
                            + current[row, column - 1]
                        )
                        + ry
                        * (
                            current[row + 1, column]
                            - 2.0 * current[row, column]
                            + current[row - 1, column]
                        )
                    )
            for row in range(current.shape[0]):
                next_temperature[row, 0] = next_temperature[row, 1]
                next_temperature[row, -1] = next_temperature[row, -2]
            for column in range(current.shape[1]):
                next_temperature[0, column] = next_temperature[1, column]
                next_temperature[-1, column] = next_temperature[-2, column]
            current, next_temperature = next_temperature, current
        return current


def advance_heat_steps(current, rx, ry, steps, boundary_type, boundary_values):
    """Return ``current`` advanced by ``steps`` with the native kernel.

    Raises a clear error when the optional Numba extra is absent.  The caller
    validates inputs first, allowing these kernels to remain focused and fast.
    """
    if not NUMBA_AVAILABLE:
        raise RuntimeError(
            "Numba acceleration is unavailable; install "
            "engineering-simulation-toolkit[acceleration] or use acceleration='python'."
        )

    next_temperature = np.empty_like(current)
    if boundary_type == "dirichlet":
        boundaries = np.asarray(boundary_values, dtype=np.float64)
        return _advance_dirichlet(current, next_temperature, rx, ry, steps, boundaries)
    return _advance_neumann(current, next_temperature, rx, ry, steps)
