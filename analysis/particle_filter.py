"""Reusable bootstrap Particle Filter utilities for nonlinear systems."""

import numpy as np


def _as_2d_array(array, name):
    """Return a validated two-dimensional float array."""
    array = np.asarray(array, dtype=float)

    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")

    return array


def _as_1d_array(array, name):
    """Return a validated one-dimensional float array."""
    array = np.asarray(array, dtype=float)

    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")

    return array


def _validate_callable(function, name):
    """Validate that a callback is callable."""
    if not callable(function):
        raise ValueError(f"{name} must be callable")


def _validate_dt(dt):
    """Return a validated positive sample time."""
    dt = float(dt)

    if dt <= 0:
        raise ValueError("dt must be positive")

    return dt


class ParticleFilter:
    """Bootstrap particle filter for nonlinear state estimation.

    The filter represents uncertainty with many weighted particles. Prediction
    propagates each particle through a nonlinear process model, and update
    reweights particles using a measurement likelihood.
    """

    def __init__(
        self,
        particles,
        weights=None,
        process_model=None,
        measurement_likelihood=None,
        process_noise_sampler=None,
        rng=None,
    ):
        """Initialize particles, weights, model callbacks, and random generator."""
        _validate_callable(process_model, "process_model")
        _validate_callable(measurement_likelihood, "measurement_likelihood")

        self.particles = _as_2d_array(particles, "particles")
        self.num_particles, self.state_dim = self.particles.shape

        if self.num_particles <= 0:
            raise ValueError("particles must contain at least one particle")

        self.weights = self._validate_weights(weights)
        self.process_model = process_model
        self.measurement_likelihood = measurement_likelihood
        self.process_noise_sampler = process_noise_sampler
        self.rng = self._make_rng(rng)

    def _make_rng(self, rng):
        """Return a NumPy random generator from a generator or seed-like value."""
        if isinstance(rng, np.random.Generator):
            return rng

        return np.random.default_rng(rng)

    def _validate_weights(self, weights):
        """Return validated and normalized particle weights."""
        if weights is None:
            return np.full(self.num_particles, 1.0 / self.num_particles)

        weights = _as_1d_array(weights, "weights")

        if len(weights) != self.num_particles:
            raise ValueError("weights length must match number of particles")

        if np.any(weights < 0):
            raise ValueError("weights must be nonnegative")

        weights_sum = np.sum(weights)
        if weights_sum <= 0:
            raise ValueError("weights sum must be positive")

        return weights / weights_sum

    def normalize_weights(self):
        """Normalize particle weights and return the normalized weights."""
        if np.any(self.weights < 0) or not np.all(np.isfinite(self.weights)):
            raise ValueError("weights must be finite and nonnegative")

        weights_sum = np.sum(self.weights)
        if weights_sum <= 0:
            self.weights = np.full(self.num_particles, 1.0 / self.num_particles)
        else:
            self.weights = self.weights / weights_sum

        return self.weights

    def predict(self, dt):
        """Propagate particles through the process model and process noise."""
        dt = _validate_dt(dt)
        predicted_particles = _as_2d_array(
            self.process_model(self.particles, dt),
            "process_model output",
        )

        if predicted_particles.shape != self.particles.shape:
            raise ValueError("process_model output shape must match particles")

        if self.process_noise_sampler is not None:
            process_noise = _as_2d_array(
                self.process_noise_sampler(
                    self.num_particles,
                    self.state_dim,
                    self.rng,
                ),
                "process noise",
            )

            if process_noise.shape != self.particles.shape:
                raise ValueError("process noise shape must match particles")

            predicted_particles = predicted_particles + process_noise

        self.particles = predicted_particles
        return self.particles

    def update(self, z):
        """Update particle weights using measurement likelihoods."""
        likelihoods = _as_1d_array(
            self.measurement_likelihood(z, self.particles),
            "likelihoods",
        )

        if len(likelihoods) != self.num_particles:
            raise ValueError("likelihoods length must match number of particles")

        if np.any(likelihoods < 0):
            raise ValueError("likelihoods must be nonnegative")

        epsilon = np.finfo(float).tiny
        self.weights = self.weights * (likelihoods + epsilon)
        self.normalize_weights()

        return self.weights

    def estimate(self):
        """Return the weighted mean state estimate."""
        return np.average(self.particles, axis=0, weights=self.weights)

    def effective_sample_size(self):
        """Return the effective number of particles."""
        return 1.0 / np.sum(self.weights**2)

    def systematic_resample(self):
        """Resample particles using systematic resampling."""
        positions = (self.rng.random() + np.arange(self.num_particles)) / (
            self.num_particles
        )
        cumulative_sum = np.cumsum(self.weights)
        cumulative_sum[-1] = 1.0
        indexes = np.searchsorted(cumulative_sum, positions)

        self.particles = self.particles[indexes].copy()
        self.weights = np.full(self.num_particles, 1.0 / self.num_particles)

        return self.particles

    def step(self, z, dt, resample_threshold=0.5):
        """Run predict, update, optional resampling, and return the estimate."""
        if resample_threshold < 0:
            raise ValueError("resample_threshold must be nonnegative")

        self.predict(dt)
        self.update(z)

        if self.effective_sample_size() < resample_threshold * self.num_particles:
            self.systematic_resample()

        return self.estimate()
