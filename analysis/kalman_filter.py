"""Reusable discrete-time Kalman filter utilities."""

from dataclasses import dataclass

import numpy as np
from scipy.signal import cont2discrete


def _as_2d_matrix(matrix, name):
    """Return a validated two-dimensional float matrix."""
    matrix_array = np.asarray(matrix, dtype=float)

    if matrix_array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix")

    if not np.all(np.isfinite(matrix_array)):
        raise ValueError(f"{name} must contain only finite values")

    return matrix_array


def _as_vector(vector, name, expected_length):
    """Return a validated one-dimensional float vector."""
    vector_array = np.asarray(vector, dtype=float)

    if vector_array.ndim == 0:
        vector_array = vector_array.reshape(1)

    if vector_array.ndim != 1:
        raise ValueError(f"{name} must be a scalar or one-dimensional vector")

    if len(vector_array) != expected_length:
        raise ValueError(f"{name} length must be {expected_length}")

    if not np.all(np.isfinite(vector_array)):
        raise ValueError(f"{name} must contain only finite values")

    return vector_array


def _validate_covariance_matrix(matrix, name, expected_shape):
    """Validate a symmetric positive-semidefinite covariance matrix."""
    matrix = _as_2d_matrix(matrix, name)

    if matrix.shape != expected_shape:
        raise ValueError(f"{name} shape must be {expected_shape}")

    if not np.allclose(matrix, matrix.T):
        raise ValueError(f"{name} must be symmetric")

    if np.min(np.linalg.eigvalsh(matrix)) < -1e-10:
        raise ValueError(f"{name} must be positive semidefinite")

    return matrix


def validate_kalman_matrices(A, B, C, Q, R, x_hat, P):
    """Validate discrete-time Kalman filter matrix dimensions."""
    A = _as_2d_matrix(A, "A")
    B = _as_2d_matrix(B, "B")
    C = _as_2d_matrix(C, "C")
    Q = _as_2d_matrix(Q, "Q")
    R = _as_2d_matrix(R, "R")
    P = _as_2d_matrix(P, "P")

    n_states = A.shape[0]
    n_measurements = C.shape[0]

    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")

    if B.shape[0] != n_states:
        raise ValueError("B row count must match A state count")

    if C.shape[1] != n_states:
        raise ValueError("C column count must match A state count")

    Q = _validate_covariance_matrix(Q, "Q", A.shape)
    R = _validate_covariance_matrix(R, "R", (n_measurements, n_measurements))
    P = _validate_covariance_matrix(P, "P", A.shape)

    x_hat = _as_vector(x_hat, "x_hat", n_states)

    return A, B, C, Q, R, x_hat, P


def discretize_state_space(A, B, dt):
    """Discretize continuous-time ``x_dot = A*x + B*u`` matrices.

    Parameters
    ----------
    A, B : array-like
        Continuous-time state and input matrices.
    dt : float
        Sample time in seconds.

    Returns
    -------
    tuple
        ``(A_d, B_d)`` for the discrete-time model
        ``x[k+1] = A_d*x[k] + B_d*u[k]``.
    """
    A = _as_2d_matrix(A, "A")
    B = _as_2d_matrix(B, "B")

    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")

    if B.shape[0] != A.shape[0]:
        raise ValueError("B row count must match A state count")

    if dt <= 0:
        raise ValueError("dt must be positive")

    C_dummy = np.eye(A.shape[0])
    D_dummy = np.zeros((A.shape[0], B.shape[1]))
    A_d, B_d, _, _, _ = cont2discrete((A, B, C_dummy, D_dummy), dt)

    return A_d, B_d


@dataclass
class KalmanFilter:
    """Linear discrete-time Kalman filter.

    The model is:

    ``x[k+1] = A*x[k] + B*u[k] + w[k]``

    ``y[k] = C*x[k] + v[k]``

    where ``Q`` is the process-noise covariance and ``R`` is the
    measurement-noise covariance.
    """

    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    x_hat: np.ndarray
    P: np.ndarray
    D: np.ndarray | None = None
    name: str = "KalmanFilter"

    def __post_init__(self):
        """Validate and store arrays as floats."""
        (
            self.A,
            self.B,
            self.C,
            self.Q,
            self.R,
            self.x_hat,
            self.P,
        ) = validate_kalman_matrices(
            self.A,
            self.B,
            self.C,
            self.Q,
            self.R,
            self.x_hat,
            self.P,
        )
        self.D = self._validate_d_matrix(self.D)
        self.last_gain = np.zeros((self.A.shape[0], self.C.shape[0]))

    def _validate_d_matrix(self, D):
        """Validate optional measurement feedthrough matrix."""
        if D is None:
            return np.zeros((self.C.shape[0], self.B.shape[1]))

        D = _as_2d_matrix(D, "D")

        if D.shape != (self.C.shape[0], self.B.shape[1]):
            raise ValueError("D shape must be (number of measurements, number of inputs)")

        return D

    def _input_vector(self, u):
        """Return an input vector with the correct input dimension."""
        n_inputs = self.B.shape[1]

        if u is None:
            return np.zeros(n_inputs)

        return _as_vector(u, "u", n_inputs)

    def _measurement_vector(self, y):
        """Return a measurement vector with the correct output dimension."""
        return _as_vector(y, "y", self.C.shape[0])

    def predict(self, u=None):
        """Run the Kalman prediction step and return the predicted state."""
        input_vector = self._input_vector(u)

        self.x_hat = self.A @ self.x_hat + self.B @ input_vector
        self.P = self.A @ self.P @ self.A.T + self.Q

        return self.x_hat

    def update(self, y, u=None):
        """Run the measurement update step.

        Returns
        -------
        tuple
            ``(x_hat, K)`` where ``K`` is the Kalman gain.
        """
        measurement = self._measurement_vector(y)
        input_vector = self._input_vector(u)
        innovation = measurement - (self.C @ self.x_hat + self.D @ input_vector)
        innovation_covariance = self.C @ self.P @ self.C.T + self.R
        kalman_gain = np.linalg.solve(
            innovation_covariance.T,
            (self.P @ self.C.T).T,
        ).T

        self.x_hat = self.x_hat + kalman_gain @ innovation
        identity = np.eye(self.A.shape[0])
        innovation_factor = identity - kalman_gain @ self.C
        self.P = (
            innovation_factor @ self.P @ innovation_factor.T
            + kalman_gain @ self.R @ kalman_gain.T
        )
        self.P = 0.5 * (self.P + self.P.T)
        self.last_gain = kalman_gain

        return self.x_hat, kalman_gain

    def step(self, y, u=None):
        """Run prediction and update steps, returning ``(x_hat, K)``."""
        self.predict(u)
        return self.update(y, u)
