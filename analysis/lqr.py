"""Continuous-time Linear Quadratic Regulator utilities."""

import numpy as np
from scipy.linalg import solve_continuous_are


def _as_2d_matrix(matrix, name):
    """Return a validated two-dimensional float matrix."""
    matrix_array = np.asarray(matrix, dtype=float)

    if matrix_array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix")

    if not np.all(np.isfinite(matrix_array)):
        raise ValueError(f"{name} must contain only finite values")

    return matrix_array


def _validate_symmetric(matrix, name):
    """Validate that a matrix is symmetric within numerical tolerance."""
    if not np.allclose(matrix, matrix.T):
        raise ValueError(f"{name} must be symmetric")


def _validate_positive_semidefinite(matrix, name):
    """Validate that a symmetric matrix is positive semidefinite."""
    eigenvalues = np.linalg.eigvalsh(matrix)

    if np.min(eigenvalues) < -1e-10:
        raise ValueError(f"{name} must be positive semidefinite")


def validate_lqr_matrices(A, B, Q, R):
    """Validate continuous-time LQR matrix dimensions and properties."""
    A = _as_2d_matrix(A, "A")
    B = _as_2d_matrix(B, "B")
    Q = _as_2d_matrix(Q, "Q")
    R = _as_2d_matrix(R, "R")

    n_states = A.shape[0]
    n_inputs = B.shape[1]

    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")

    if B.shape[0] != n_states:
        raise ValueError("B row count must match A state count")

    if Q.shape != A.shape:
        raise ValueError("Q shape must match A shape")

    if R.shape != (n_inputs, n_inputs):
        raise ValueError("R shape must be (number of inputs, number of inputs)")

    _validate_symmetric(Q, "Q")
    _validate_symmetric(R, "R")
    _validate_positive_semidefinite(Q, "Q")

    try:
        np.linalg.cholesky(R)
    except np.linalg.LinAlgError as exc:
        raise ValueError("R must be symmetric positive definite") from exc

    return A, B, Q, R


def lqr(A, B, Q, R):
    """Compute a continuous-time Linear Quadratic Regulator gain.

    The control law convention is ``u = -K*x``. The gain minimizes the
    continuous-time cost integral ``x.T Q x + u.T R u`` for the system
    ``x_dot = A*x + B*u``.

    Returns
    -------
    tuple
        ``(K, P, closed_loop_eigenvalues)`` where ``P`` is the solution to the
        continuous algebraic Riccati equation and the eigenvalues are from
        ``A - B @ K``.
    """
    A, B, Q, R = validate_lqr_matrices(A, B, Q, R)

    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)
    closed_loop_eigenvalues = np.linalg.eigvals(A - B @ K)

    return K, P, closed_loop_eigenvalues
