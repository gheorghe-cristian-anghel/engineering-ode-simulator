import numpy as np
import pytest

from analysis.particle_filter import ParticleFilter


def _identity_process_model(particles, dt):
    """Return particles unchanged."""
    return np.array(particles, dtype=float)


def _direct_measurement_likelihood(z, particles):
    """Return Gaussian likelihoods for direct scalar state measurements."""
    measurement = np.asarray(z, dtype=float).reshape(-1)[0]
    error = measurement - particles[:, 0]
    measurement_std = 0.5

    return np.exp(-0.5 * (error / measurement_std) ** 2)


def _simple_particle_filter():
    """Return a small one-state particle filter for tests."""
    particles = np.array([[-1.0], [0.0], [1.0]])

    return ParticleFilter(
        particles=particles,
        process_model=_identity_process_model,
        measurement_likelihood=_direct_measurement_likelihood,
        rng=0,
    )


def test_initializes_particles_and_uniform_weights_correctly():
    """ParticleFilter should store particles and default uniform weights."""
    particle_filter = _simple_particle_filter()

    assert particle_filter.particles.shape == (3, 1)
    assert particle_filter.weights.shape == (3,)
    assert particle_filter.num_particles == 3
    assert particle_filter.state_dim == 1
    assert np.allclose(particle_filter.weights, np.full(3, 1.0 / 3.0))


def test_estimate_returns_weighted_mean_with_correct_shape():
    """Estimate should return the weighted particle mean."""
    particles = np.array([[0.0], [2.0], [4.0]])
    weights = np.array([0.25, 0.25, 0.5])
    particle_filter = ParticleFilter(
        particles=particles,
        weights=weights,
        process_model=_identity_process_model,
        measurement_likelihood=_direct_measurement_likelihood,
        rng=0,
    )

    estimate = particle_filter.estimate()

    assert estimate.shape == (1,)
    assert estimate[0] == pytest.approx(2.5)


def test_effective_sample_size_returns_num_particles_for_uniform_weights():
    """Uniform weights should use all particles effectively."""
    particle_filter = _simple_particle_filter()

    assert particle_filter.effective_sample_size() == pytest.approx(3.0)


def test_nonuniform_weights_reduce_effective_sample_size():
    """Concentrated weights should lower effective sample size."""
    particle_filter = ParticleFilter(
        particles=np.array([[0.0], [1.0], [2.0]]),
        weights=np.array([0.8, 0.1, 0.1]),
        process_model=_identity_process_model,
        measurement_likelihood=_direct_measurement_likelihood,
        rng=0,
    )

    assert particle_filter.effective_sample_size() < particle_filter.num_particles


def test_normalize_weights_makes_weights_sum_to_one():
    """normalize_weights should scale valid weights to unit sum."""
    particle_filter = _simple_particle_filter()
    particle_filter.weights = np.array([1.0, 2.0, 3.0])

    particle_filter.normalize_weights()

    assert np.sum(particle_filter.weights) == pytest.approx(1.0)


def test_update_with_valid_likelihoods_keeps_weights_normalized():
    """Measurement updates should keep weights normalized."""
    particle_filter = _simple_particle_filter()

    particle_filter.update(1.0)

    assert np.sum(particle_filter.weights) == pytest.approx(1.0)
    assert np.all(particle_filter.weights >= 0.0)


def test_systematic_resample_keeps_shape_and_resets_weights_uniform():
    """Systematic resampling should preserve shape and reset weights."""
    particle_filter = ParticleFilter(
        particles=np.array([[0.0], [1.0], [2.0], [3.0]]),
        weights=np.array([0.7, 0.1, 0.1, 0.1]),
        process_model=_identity_process_model,
        measurement_likelihood=_direct_measurement_likelihood,
        rng=0,
    )

    resampled_particles = particle_filter.systematic_resample()

    assert resampled_particles.shape == (4, 1)
    assert particle_filter.particles.shape == (4, 1)
    assert np.allclose(particle_filter.weights, np.full(4, 0.25))


def test_step_returns_estimate_with_correct_shape():
    """A full particle filter step should return a state estimate."""
    particle_filter = _simple_particle_filter()

    estimate = particle_filter.step(1.0, dt=0.1)

    assert estimate.shape == (1,)


def test_prediction_preserves_particle_shape():
    """Prediction should preserve the particle array shape."""
    particle_filter = _simple_particle_filter()

    predicted_particles = particle_filter.predict(dt=0.1)

    assert predicted_particles.shape == (3, 1)


def test_invalid_particles_dimension_raises_value_error():
    """Particles must be a two-dimensional array."""
    with pytest.raises(ValueError):
        ParticleFilter(
            particles=np.array([0.0, 1.0]),
            process_model=_identity_process_model,
            measurement_likelihood=_direct_measurement_likelihood,
        )


def test_invalid_weights_shape_raises_value_error():
    """Weights length must match the number of particles."""
    with pytest.raises(ValueError):
        ParticleFilter(
            particles=np.array([[0.0], [1.0]]),
            weights=np.array([0.5, 0.25, 0.25]),
            process_model=_identity_process_model,
            measurement_likelihood=_direct_measurement_likelihood,
        )


def test_negative_weights_raise_value_error():
    """Weights must be nonnegative."""
    with pytest.raises(ValueError):
        ParticleFilter(
            particles=np.array([[0.0], [1.0]]),
            weights=np.array([1.0, -0.5]),
            process_model=_identity_process_model,
            measurement_likelihood=_direct_measurement_likelihood,
        )


def test_invalid_likelihood_shape_raises_value_error():
    """Likelihood output length must match the number of particles."""
    particle_filter = ParticleFilter(
        particles=np.array([[0.0], [1.0]]),
        process_model=_identity_process_model,
        measurement_likelihood=lambda z, particles: np.array([1.0]),
    )

    with pytest.raises(ValueError):
        particle_filter.update(0.0)


def test_all_zero_likelihoods_are_handled_robustly():
    """All-zero likelihoods should not create invalid weights."""
    particle_filter = ParticleFilter(
        particles=np.array([[0.0], [1.0], [2.0]]),
        process_model=_identity_process_model,
        measurement_likelihood=lambda z, particles: np.zeros(len(particles)),
    )

    particle_filter.update(0.0)

    assert np.sum(particle_filter.weights) == pytest.approx(1.0)
    assert np.all(np.isfinite(particle_filter.weights))
    assert np.all(particle_filter.weights >= 0.0)


def test_invalid_dt_raises_value_error():
    """Sample time must be positive."""
    particle_filter = _simple_particle_filter()

    with pytest.raises(ValueError):
        particle_filter.predict(dt=0.0)


def test_process_noise_shape_mismatch_raises_value_error():
    """Process noise shape must match particle shape."""
    particle_filter = ParticleFilter(
        particles=np.array([[0.0], [1.0]]),
        process_model=_identity_process_model,
        measurement_likelihood=_direct_measurement_likelihood,
        process_noise_sampler=lambda num_particles, state_dim, rng: np.zeros(
            (num_particles, state_dim + 1)
        ),
    )

    with pytest.raises(ValueError):
        particle_filter.predict(dt=0.1)


def test_simple_scalar_tracking_case_improves_estimate_toward_measurement():
    """A direct measurement should pull the particle estimate toward it."""
    particles = np.linspace(-2.0, 2.0, 41).reshape(-1, 1)
    particle_filter = ParticleFilter(
        particles=particles,
        process_model=_identity_process_model,
        measurement_likelihood=_direct_measurement_likelihood,
        rng=1,
    )
    initial_error = abs(1.5 - particle_filter.estimate()[0])

    estimate = particle_filter.step(1.5, dt=0.1, resample_threshold=0.0)
    updated_error = abs(1.5 - estimate[0])

    assert estimate[0] > 0.0
    assert updated_error < initial_error


def test_particle_filter_outputs_remain_finite_and_smooth_noisy_measurements():
    """A simple particle filter should produce finite smoothed estimates."""
    rng = np.random.default_rng(3)
    true_value = 1.0
    measurements = true_value + rng.normal(0.0, 0.3, 40)
    particle_filter = ParticleFilter(
        particles=np.linspace(-1.0, 3.0, 81).reshape(-1, 1),
        process_model=_identity_process_model,
        measurement_likelihood=_direct_measurement_likelihood,
        rng=1,
    )

    estimates = []
    for measurement in measurements:
        estimate = particle_filter.step(
            measurement,
            dt=0.1,
            resample_threshold=0.0,
        )
        estimates.append(estimate[0])

        assert np.all(np.isfinite(particle_filter.particles))
        assert np.all(np.isfinite(particle_filter.weights))
        assert np.sum(particle_filter.weights) == pytest.approx(1.0)
        assert particle_filter.effective_sample_size() > 0.0

    estimate_error = np.mean(np.abs(true_value - np.array(estimates[-20:])))
    measurement_error = np.mean(np.abs(true_value - measurements[-20:]))

    assert estimate_error < measurement_error
    assert np.all(np.isfinite(estimates))
