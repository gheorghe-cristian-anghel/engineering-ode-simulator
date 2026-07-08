import numpy as np
import pytest

from analysis.unscented_kalman_filter import UnscentedKalmanFilter


def _identity_process_model(x, dt):
    """Return a simple constant-state process model."""
    return np.array(x, dtype=float)


def _identity_measurement_model(x):
    """Measure the first state directly."""
    return np.array([x[0]])


def _simple_ukf():
    """Return a simple one-state UKF for tests."""
    return UnscentedKalmanFilter(
        x0=np.array([0.0]),
        P0=np.array([[1.0]]),
        Q=np.array([[1e-4]]),
        R=np.array([[0.01]]),
        process_model=_identity_process_model,
        measurement_model=_identity_measurement_model,
        dt=0.1,
    )


def test_ukf_initializes_with_correct_state_dimension():
    """UKF should store the expected state and measurement dimensions."""
    ukf = _simple_ukf()

    assert ukf.state_dim == 1
    assert ukf.measurement_dim == 1
    assert ukf.x.shape == (1,)
    assert ukf.P.shape == (1, 1)


def test_generated_sigma_points_have_expected_shape():
    """Sigma point count should be 2n + 1."""
    ukf = UnscentedKalmanFilter(
        x0=np.array([1.0, -2.0]),
        P0=np.eye(2),
        Q=np.eye(2) * 0.01,
        R=np.array([[0.1]]),
        process_model=_identity_process_model,
        measurement_model=_identity_measurement_model,
        dt=0.1,
    )

    sigma_points = ukf.generate_sigma_points()

    assert sigma_points.shape == (5, 2)
    assert np.allclose(sigma_points[0], ukf.x)


def test_weights_have_expected_length_and_mean_sum():
    """Mean and covariance weights should match the sigma point count."""
    ukf = UnscentedKalmanFilter(
        x0=np.array([0.0, 0.0]),
        P0=np.eye(2),
        Q=np.eye(2) * 0.01,
        R=np.array([[0.1]]),
        process_model=_identity_process_model,
        measurement_model=_identity_measurement_model,
        dt=0.1,
    )

    assert len(ukf.weights_mean) == 5
    assert len(ukf.weights_covariance) == 5
    assert np.sum(ukf.weights_mean) == pytest.approx(1.0)


def test_predict_preserves_state_shape():
    """Predict should preserve the state vector shape."""
    ukf = _simple_ukf()

    predicted_state = ukf.predict()

    assert predicted_state.shape == (1,)


def test_update_preserves_state_shape():
    """Update should preserve state and covariance shapes."""
    ukf = _simple_ukf()

    updated_state = ukf.update(1.0)

    assert updated_state.shape == (1,)
    assert ukf.P.shape == (1, 1)


def test_step_preserves_state_and_covariance_shapes():
    """A full UKF step should preserve expected state and covariance shapes."""
    ukf = _simple_ukf()

    updated_state = ukf.step(1.0)

    assert updated_state.shape == (1,)
    assert ukf.P.shape == (1, 1)


def test_simple_linear_system_update_pulls_estimate_toward_measurement():
    """A direct measurement should pull the estimate toward the measurement."""
    ukf = _simple_ukf()
    initial_error = abs(1.0 - ukf.x[0])

    updated_state = ukf.step(1.0)
    updated_error = abs(1.0 - updated_state[0])

    assert updated_state[0] > 0.0
    assert updated_error < initial_error


def test_invalid_x0_raises_value_error():
    """x0 must be one-dimensional."""
    with pytest.raises(ValueError):
        UnscentedKalmanFilter(
            x0=0.0,
            P0=np.array([[1.0]]),
            Q=np.array([[0.01]]),
            R=np.array([[0.1]]),
            process_model=_identity_process_model,
            measurement_model=_identity_measurement_model,
            dt=0.1,
        )


def test_invalid_p0_shape_raises_value_error():
    """P0 shape must match the state dimension."""
    with pytest.raises(ValueError):
        UnscentedKalmanFilter(
            x0=np.array([0.0, 0.0]),
            P0=np.eye(3),
            Q=np.eye(2),
            R=np.array([[0.1]]),
            process_model=_identity_process_model,
            measurement_model=_identity_measurement_model,
            dt=0.1,
        )


def test_invalid_q_shape_raises_value_error():
    """Q shape must match the state dimension."""
    with pytest.raises(ValueError):
        UnscentedKalmanFilter(
            x0=np.array([0.0, 0.0]),
            P0=np.eye(2),
            Q=np.eye(3),
            R=np.array([[0.1]]),
            process_model=_identity_process_model,
            measurement_model=_identity_measurement_model,
            dt=0.1,
        )


def test_invalid_r_shape_raises_value_error():
    """R must be square."""
    with pytest.raises(ValueError):
        UnscentedKalmanFilter(
            x0=np.array([0.0]),
            P0=np.eye(1),
            Q=np.eye(1),
            R=np.array([[0.1, 0.0]]),
            process_model=_identity_process_model,
            measurement_model=_identity_measurement_model,
            dt=0.1,
        )


def test_invalid_covariance_properties_raise_value_error():
    """UKF covariances must be symmetric positive semidefinite."""
    with pytest.raises(ValueError):
        UnscentedKalmanFilter(
            x0=np.array([0.0, 0.0]),
            P0=[[1.0, 2.0], [0.0, 1.0]],
            Q=np.eye(2),
            R=np.array([[0.1]]),
            process_model=_identity_process_model,
            measurement_model=_identity_measurement_model,
            dt=0.1,
        )

    with pytest.raises(ValueError):
        UnscentedKalmanFilter(
            x0=np.array([0.0]),
            P0=np.eye(1),
            Q=[[-0.01]],
            R=np.array([[0.1]]),
            process_model=_identity_process_model,
            measurement_model=_identity_measurement_model,
            dt=0.1,
        )

    with pytest.raises(ValueError):
        UnscentedKalmanFilter(
            x0=np.array([0.0]),
            P0=np.eye(1),
            Q=np.eye(1),
            R=[[-0.1]],
            process_model=_identity_process_model,
            measurement_model=_identity_measurement_model,
            dt=0.1,
        )


def test_invalid_dt_raises_value_error():
    """Sample time must be positive."""
    with pytest.raises(ValueError):
        UnscentedKalmanFilter(
            x0=np.array([0.0]),
            P0=np.eye(1),
            Q=np.eye(1),
            R=np.array([[0.1]]),
            process_model=_identity_process_model,
            measurement_model=_identity_measurement_model,
            dt=0.0,
        )


def test_invalid_alpha_raises_value_error():
    """alpha must be positive."""
    with pytest.raises(ValueError):
        UnscentedKalmanFilter(
            x0=np.array([0.0]),
            P0=np.eye(1),
            Q=np.eye(1),
            R=np.array([[0.1]]),
            process_model=_identity_process_model,
            measurement_model=_identity_measurement_model,
            dt=0.1,
            alpha=0.0,
        )


def test_invalid_beta_raises_value_error():
    """beta must be nonnegative."""
    with pytest.raises(ValueError):
        UnscentedKalmanFilter(
            x0=np.array([0.0]),
            P0=np.eye(1),
            Q=np.eye(1),
            R=np.array([[0.1]]),
            process_model=_identity_process_model,
            measurement_model=_identity_measurement_model,
            dt=0.1,
            beta=-1.0,
        )


def test_measurement_dimension_mismatch_raises_value_error():
    """measurement_model output dimension must match R."""
    with pytest.raises(ValueError):
        UnscentedKalmanFilter(
            x0=np.array([0.0]),
            P0=np.eye(1),
            Q=np.eye(1),
            R=np.array([[0.1]]),
            process_model=_identity_process_model,
            measurement_model=lambda x: np.array([x[0], x[0]]),
            dt=0.1,
        )


def test_ukf_error_is_lower_than_raw_measurement_error_and_covariance_is_valid():
    """A scalar UKF should smooth noisy direct measurements."""
    rng = np.random.default_rng(2)
    true_value = 1.0
    measurements = true_value + rng.normal(0.0, 0.15, 80)
    ukf = UnscentedKalmanFilter(
        x0=np.array([0.0]),
        P0=np.array([[1.0]]),
        Q=np.array([[1e-4]]),
        R=np.array([[0.04]]),
        process_model=_identity_process_model,
        measurement_model=_identity_measurement_model,
        dt=0.1,
    )

    estimates = []
    for measurement in measurements:
        estimate = ukf.step(measurement)
        estimates.append(estimate[0])

        assert np.all(np.isfinite(ukf.P))
        assert np.allclose(ukf.P, ukf.P.T)

    estimate_error = np.mean(np.abs(true_value - np.array(estimates[-40:])))
    measurement_error = np.mean(np.abs(true_value - measurements[-40:]))

    assert estimate_error < measurement_error
    assert np.all(np.isfinite(estimates))
