"""Reusable Extended Kalman Filter utilities for nonlinear systems."""

from dataclasses import dataclass

import numpy as np


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


def _validate_callable(function, name):
    """Validate that a model or Jacobian callback is callable."""
    if not callable(function):
        raise ValueError(f"{name} must be callable")


def _validate_dt(dt):
    """Return a validated positive sample time."""
    dt = float(dt)

    if dt <= 0:
        raise ValueError("dt must be positive")

    return dt


def _validate_covariance_matrix(matrix, name):
    """Validate a square symmetric positive-semidefinite covariance matrix."""
    matrix = _as_2d_matrix(matrix, name)

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be square")

    if not np.allclose(matrix, matrix.T):
        raise ValueError(f"{name} must be symmetric")

    if np.min(np.linalg.eigvalsh(matrix)) < -1e-10:
        raise ValueError(f"{name} must be positive semidefinite")

    return matrix


def validate_ekf_matrices(Q, R, x_hat, P):
    """Validate Extended Kalman Filter covariance and state dimensions."""
    Q = _validate_covariance_matrix(Q, "Q")
    R = _validate_covariance_matrix(R, "R")
    P = _validate_covariance_matrix(P, "P")

    n_states = Q.shape[0]
    x_hat = _as_vector(x_hat, "x_hat", n_states)

    if P.shape != Q.shape:
        raise ValueError("P shape must match Q shape")

    return Q, R, x_hat, P


@dataclass
class ExtendedKalmanFilter:
    """Extended Kalman Filter for discrete-time nonlinear systems.

    The model is:

    ``x[k+1] = f(x[k], u[k], dt) + w[k]``

    ``y[k] = h(x[k]) + v[k]``

    where ``F_jacobian`` is the Jacobian of ``f`` with respect to the state,
    ``H_jacobian`` is the Jacobian of ``h`` with respect to the state, ``Q``
    is process-noise covariance, and ``R`` is measurement-noise covariance.
    """

    f: object
    h: object
    F_jacobian: object
    H_jacobian: object
    Q: np.ndarray
    R: np.ndarray
    x_hat: np.ndarray
    P: np.ndarray
    name: str = "ExtendedKalmanFilter"

    def __post_init__(self):
        """Validate callbacks, covariances, and state estimate."""
        _validate_callable(self.f, "f")
        _validate_callable(self.h, "h")
        _validate_callable(self.F_jacobian, "F_jacobian")
        _validate_callable(self.H_jacobian, "H_jacobian")

        self.Q, self.R, self.x_hat, self.P = validate_ekf_matrices(
            self.Q,
            self.R,
            self.x_hat,
            self.P,
        )
        initial_measurement = _as_vector(self.h(self.x_hat), "h output", self.R.shape[0])
        self._measurement_vector(initial_measurement, "h output")
        self.last_gain = np.zeros((len(self.x_hat), self.R.shape[0]))

    def _state_vector(self, state, name):
        """Return a validated state vector."""
        return _as_vector(state, name, len(self.x_hat))

    def _measurement_vector(self, measurement, name="y"):
        """Return a validated measurement vector."""
        return _as_vector(measurement, name, self.R.shape[0])

    def _state_jacobian(self, u, dt):
        """Evaluate and validate the transition Jacobian."""
        F = _as_2d_matrix(self.F_jacobian(self.x_hat, u, dt), "F")

        if F.shape != self.Q.shape:
            raise ValueError("F_jacobian must return a matrix with shape matching Q")

        return F

    def _measurement_jacobian(self):
        """Evaluate and validate the measurement Jacobian."""
        H = _as_2d_matrix(self.H_jacobian(self.x_hat), "H")

        if H.shape != (self.R.shape[0], len(self.x_hat)):
            raise ValueError(
                "H_jacobian must return shape "
                "(number of measurements, number of states)"
            )

        return H

    def predict(self, u=None, dt=0.01):
        """Run the EKF prediction step and return the predicted state."""
        dt = _validate_dt(dt)
        F = self._state_jacobian(u, dt)
        predicted_state = self._state_vector(self.f(self.x_hat, u, dt), "f output")

        self.x_hat = predicted_state
        self.P = F @ self.P @ F.T + self.Q

        return self.x_hat

    def update(self, y):
        """Run the EKF measurement update step.

        Returns
        -------
        tuple
            ``(x_hat, K)`` where ``K`` is the Kalman gain.
        """
        measurement = self._measurement_vector(y)
        H = self._measurement_jacobian()
        predicted_measurement = self._measurement_vector(self.h(self.x_hat), "h output")
        innovation = measurement - predicted_measurement
        innovation_covariance = H @ self.P @ H.T + self.R
        kalman_gain = np.linalg.solve(
            innovation_covariance.T,
            (self.P @ H.T).T,
        ).T

        self.x_hat = self.x_hat + kalman_gain @ innovation
        identity = np.eye(len(self.x_hat))
        innovation_factor = identity - kalman_gain @ H
        self.P = (
            innovation_factor @ self.P @ innovation_factor.T
            + kalman_gain @ self.R @ kalman_gain.T
        )
        self.P = 0.5 * (self.P + self.P.T)
        self.last_gain = kalman_gain

        return self.x_hat, kalman_gain

    def step(self, y, u=None, dt=0.01):
        """Run prediction and update steps, returning ``(x_hat, K)``."""
        self.predict(u, dt)
        return self.update(y)


def pendulum_ekf_state_transition(x, u=None, dt=0.01, L=1.0, g=9.81):
    """Return one Euler prediction step for nonlinear pendulum state."""
    dt = _validate_dt(dt)
    if L <= 0:
        raise ValueError("L must be positive")

    if g <= 0:
        raise ValueError("g must be positive")

    state = _as_vector(x, "x", 2)
    theta, omega = state

    theta_next = theta + omega * dt
    omega_next = omega - (g / L) * np.sin(theta) * dt

    return np.array([theta_next, omega_next])


def pendulum_ekf_state_jacobian(x, u=None, dt=0.01, L=1.0, g=9.81):
    """Return the pendulum EKF transition Jacobian."""
    dt = _validate_dt(dt)
    if L <= 0:
        raise ValueError("L must be positive")

    if g <= 0:
        raise ValueError("g must be positive")

    state = _as_vector(x, "x", 2)
    theta = state[0]

    return np.array(
        [
            [1.0, dt],
            [-(g / L) * np.cos(theta) * dt, 1.0],
        ]
    )


def pendulum_angle_measurement(x):
    """Return pendulum angle measurement ``y = theta``."""
    state = _as_vector(x, "x", 2)

    return np.array([state[0]])


def pendulum_angle_measurement_jacobian(x):
    """Return the Jacobian for measuring pendulum angle only."""
    _as_vector(x, "x", 2)

    return np.array([[1.0, 0.0]])
