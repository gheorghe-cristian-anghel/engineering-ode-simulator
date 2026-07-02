"""Reusable Unscented Kalman Filter utilities for nonlinear systems."""

import numpy as np


def _as_2d_matrix(matrix, name):
    """Return a validated two-dimensional float matrix."""
    matrix_array = np.asarray(matrix, dtype=float)

    if matrix_array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix")

    if not np.all(np.isfinite(matrix_array)):
        raise ValueError(f"{name} must contain only finite values")

    return matrix_array


def _as_1d_vector(vector, name):
    """Return a validated one-dimensional float vector."""
    vector_array = np.asarray(vector, dtype=float)

    if vector_array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")

    if not np.all(np.isfinite(vector_array)):
        raise ValueError(f"{name} must contain only finite values")

    return vector_array


def _as_measurement_vector(vector, name, expected_length):
    """Return a scalar or one-dimensional measurement as a vector."""
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
    """Validate that a model callback is callable."""
    if not callable(function):
        raise ValueError(f"{name} must be callable")


def _validate_dt(dt):
    """Return a validated positive sample time."""
    dt = float(dt)

    if dt <= 0:
        raise ValueError("dt must be positive")

    return dt


def _symmetric_matrix(matrix):
    """Return a numerically symmetrized copy of a covariance matrix."""
    return 0.5 * (matrix + matrix.T)


def _cholesky_with_jitter(matrix):
    """Return a Cholesky factor, adding small diagonal jitter if needed."""
    matrix = _symmetric_matrix(matrix)
    identity = np.eye(matrix.shape[0])
    jitter = 0.0

    for _ in range(8):
        try:
            return np.linalg.cholesky(matrix + jitter * identity)
        except np.linalg.LinAlgError:
            jitter = 1e-12 if jitter == 0.0 else jitter * 10.0

    return np.linalg.cholesky(matrix + jitter * identity)


class UnscentedKalmanFilter:
    """Unscented Kalman Filter for discrete-time nonlinear systems.

    The model is:

    ``x[k+1] = process_model(x[k], dt) + w[k]``

    ``z[k] = measurement_model(x[k]) + v[k]``

    where ``Q`` is process-noise covariance and ``R`` is measurement-noise
    covariance. The UKF estimates nonlinear systems without requiring
    manually derived Jacobian matrices.
    """

    def __init__(
        self,
        x0,
        P0,
        Q,
        R,
        process_model,
        measurement_model,
        dt,
        alpha=1e-3,
        beta=2.0,
        kappa=0.0,
    ):
        """Initialize the UKF state, covariance, models, and sigma weights."""
        _validate_callable(process_model, "process_model")
        _validate_callable(measurement_model, "measurement_model")

        self.x = _as_1d_vector(x0, "x0")
        self.P = _as_2d_matrix(P0, "P0")
        self.Q = _as_2d_matrix(Q, "Q")
        self.R = _as_2d_matrix(R, "R")
        self.process_model = process_model
        self.measurement_model = measurement_model
        self.dt = _validate_dt(dt)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.kappa = float(kappa)

        self.state_dim = len(self.x)
        self._validate_covariance_shapes()
        self._validate_parameters()

        initial_measurement = _as_measurement_vector(
            self.measurement_model(self.x),
            "measurement_model output",
            self.R.shape[0],
        )
        self.measurement_dim = len(initial_measurement)
        self.weights_mean, self.weights_covariance = self._compute_weights()
        self.P = _symmetric_matrix(self.P)
        self.last_gain = np.zeros((self.state_dim, self.measurement_dim))
        self._sigma_points_for_update = None

    def _validate_covariance_shapes(self):
        """Validate state, process, and measurement covariance shapes."""
        expected_state_shape = (self.state_dim, self.state_dim)

        if self.P.shape != expected_state_shape:
            raise ValueError("P0 shape must be (state_dim, state_dim)")

        if self.Q.shape != expected_state_shape:
            raise ValueError("Q shape must be (state_dim, state_dim)")

        if self.R.shape[0] != self.R.shape[1]:
            raise ValueError("R must be square")

    def _validate_parameters(self):
        """Validate UKF scaling parameters."""
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")

        if self.beta < 0:
            raise ValueError("beta must be nonnegative")

        self.lambda_ = self.alpha**2 * (self.state_dim + self.kappa) - self.state_dim
        self.scaling = self.state_dim + self.lambda_

        if self.scaling <= 0:
            raise ValueError("state_dim + lambda must be positive")

    def _compute_weights(self):
        """Return unscented transform mean and covariance weights."""
        n_sigma_points = 2 * self.state_dim + 1
        weights_mean = np.full(n_sigma_points, 1.0 / (2.0 * self.scaling))
        weights_covariance = np.full(n_sigma_points, 1.0 / (2.0 * self.scaling))

        weights_mean[0] = self.lambda_ / self.scaling
        weights_covariance[0] = (
            self.lambda_ / self.scaling + (1.0 - self.alpha**2 + self.beta)
        )

        return weights_mean, weights_covariance

    def generate_sigma_points(self):
        """Generate sigma points around the current state estimate."""
        sigma_points = np.zeros((2 * self.state_dim + 1, self.state_dim))
        sigma_points[0] = self.x
        covariance_sqrt = _cholesky_with_jitter(self.scaling * self.P)

        for index in range(self.state_dim):
            sigma_points[index + 1] = self.x + covariance_sqrt[:, index]
            sigma_points[self.state_dim + index + 1] = (
                self.x - covariance_sqrt[:, index]
            )

        return sigma_points

    def _unscented_transform(self, transformed_sigma_points, noise_covariance):
        """Return mean and covariance from transformed sigma points."""
        transformed_sigma_points = np.asarray(transformed_sigma_points, dtype=float)
        noise_covariance = _as_2d_matrix(noise_covariance, "noise_covariance")

        mean = self.weights_mean @ transformed_sigma_points
        covariance = np.array(noise_covariance, copy=True)

        for index, sigma_point in enumerate(transformed_sigma_points):
            deviation = sigma_point - mean
            covariance += self.weights_covariance[index] * np.outer(
                deviation,
                deviation,
            )

        return mean, _symmetric_matrix(covariance)

    def predict(self):
        """Run the UKF prediction step and return the predicted state."""
        sigma_points = self.generate_sigma_points()
        predicted_sigma_points = np.array(
            [
                _as_1d_vector(
                    self.process_model(sigma_point, self.dt),
                    "process_model output",
                )
                for sigma_point in sigma_points
            ]
        )

        if predicted_sigma_points.shape != sigma_points.shape:
            raise ValueError("process_model output must match state dimension")

        self.x, self.P = self._unscented_transform(predicted_sigma_points, self.Q)
        self._sigma_points_for_update = predicted_sigma_points

        return self.x

    def update(self, z):
        """Run the UKF measurement update step and return the updated state."""
        measurement = _as_measurement_vector(z, "z", self.measurement_dim)
        if self._sigma_points_for_update is None:
            sigma_points = self.generate_sigma_points()
        else:
            sigma_points = self._sigma_points_for_update

        measurement_sigma_points = np.array(
            [
                _as_measurement_vector(
                    self.measurement_model(sigma_point),
                    "measurement_model output",
                    self.measurement_dim,
                )
                for sigma_point in sigma_points
            ]
        )
        predicted_measurement, innovation_covariance = self._unscented_transform(
            measurement_sigma_points,
            self.R,
        )
        cross_covariance = np.zeros((self.state_dim, self.measurement_dim))

        for index in range(len(sigma_points)):
            state_deviation = sigma_points[index] - self.x
            measurement_deviation = measurement_sigma_points[index] - predicted_measurement
            cross_covariance += self.weights_covariance[index] * np.outer(
                state_deviation,
                measurement_deviation,
            )

        kalman_gain = np.linalg.solve(
            innovation_covariance.T,
            cross_covariance.T,
        ).T
        innovation = measurement - predicted_measurement

        self.x = self.x + kalman_gain @ innovation
        self.P = self.P - kalman_gain @ innovation_covariance @ kalman_gain.T
        self.P = _symmetric_matrix(self.P)
        self.last_gain = kalman_gain
        self._sigma_points_for_update = None

        return self.x

    def step(self, z):
        """Run prediction and update steps, returning the updated state."""
        self.predict()
        return self.update(z)
