import numpy as np
import pytest

from analysis.kalman_filter import KalmanFilter, discretize_state_space


def _simple_filter():
    """Return a simple one-state Kalman filter for tests."""
    return KalmanFilter(
        A=np.array([[1.0]]),
        B=np.array([[0.0]]),
        C=np.array([[1.0]]),
        Q=np.array([[0.01]]),
        R=np.array([[0.1]]),
        x_hat=np.array([0.0]),
        P=np.array([[1.0]]),
    )


def test_kalman_filter_stores_arrays_with_expected_shapes():
    """KalmanFilter should store validated arrays with expected dimensions."""
    kalman_filter = _simple_filter()

    assert kalman_filter.A.shape == (1, 1)
    assert kalman_filter.B.shape == (1, 1)
    assert kalman_filter.C.shape == (1, 1)
    assert kalman_filter.Q.shape == (1, 1)
    assert kalman_filter.R.shape == (1, 1)
    assert kalman_filter.x_hat.shape == (1,)
    assert kalman_filter.P.shape == (1, 1)


def test_predict_returns_state_estimate_with_correct_shape():
    """Predict should preserve state vector shape."""
    kalman_filter = _simple_filter()

    x_hat = kalman_filter.predict()

    assert x_hat.shape == (1,)


def test_update_returns_state_estimate_with_correct_shape():
    """Update should preserve state vector shape."""
    kalman_filter = _simple_filter()
    kalman_filter.predict()

    x_hat, gain = kalman_filter.update(1.0)

    assert x_hat.shape == (1,)
    assert gain.shape == (1, 1)


def test_step_works_for_scalar_measurement_in_siso_case():
    """Scalar measurements should be accepted for single-output systems."""
    kalman_filter = _simple_filter()

    x_hat, gain = kalman_filter.step(1.0)

    assert x_hat.shape == (1,)
    assert gain.shape == (1, 1)


def test_invalid_a_dimensions_raise_value_error():
    """A must be square."""
    with pytest.raises(ValueError):
        KalmanFilter(
            A=[[1.0, 0.0]],
            B=[[0.0]],
            C=[[1.0]],
            Q=[[1.0]],
            R=[[1.0]],
            x_hat=[0.0],
            P=[[1.0]],
        )


def test_invalid_b_dimensions_raise_value_error():
    """B row count must match A state count."""
    with pytest.raises(ValueError):
        KalmanFilter(
            A=np.eye(2),
            B=[[1.0]],
            C=[[1.0, 0.0]],
            Q=np.eye(2),
            R=[[1.0]],
            x_hat=[0.0, 0.0],
            P=np.eye(2),
        )


def test_invalid_c_dimensions_raise_value_error():
    """C column count must match A state count."""
    with pytest.raises(ValueError):
        KalmanFilter(
            A=np.eye(2),
            B=[[0.0], [1.0]],
            C=[[1.0]],
            Q=np.eye(2),
            R=[[1.0]],
            x_hat=[0.0, 0.0],
            P=np.eye(2),
        )


def test_invalid_q_dimensions_raise_value_error():
    """Q shape must match A shape."""
    with pytest.raises(ValueError):
        KalmanFilter(
            A=np.eye(2),
            B=[[0.0], [1.0]],
            C=[[1.0, 0.0]],
            Q=np.eye(3),
            R=[[1.0]],
            x_hat=[0.0, 0.0],
            P=np.eye(2),
        )


def test_invalid_r_dimensions_raise_value_error():
    """R shape must match measurement count."""
    with pytest.raises(ValueError):
        KalmanFilter(
            A=np.eye(2),
            B=[[0.0], [1.0]],
            C=[[1.0, 0.0]],
            Q=np.eye(2),
            R=np.eye(2),
            x_hat=[0.0, 0.0],
            P=np.eye(2),
        )


def test_invalid_x_hat_dimensions_raise_value_error():
    """x_hat length must match state count."""
    with pytest.raises(ValueError):
        KalmanFilter(
            A=np.eye(2),
            B=[[0.0], [1.0]],
            C=[[1.0, 0.0]],
            Q=np.eye(2),
            R=[[1.0]],
            x_hat=[0.0],
            P=np.eye(2),
        )


def test_invalid_p_dimensions_raise_value_error():
    """P shape must match A shape."""
    with pytest.raises(ValueError):
        KalmanFilter(
            A=np.eye(2),
            B=[[0.0], [1.0]],
            C=[[1.0, 0.0]],
            Q=np.eye(2),
            R=[[1.0]],
            x_hat=[0.0, 0.0],
            P=np.eye(3),
        )


def test_invalid_covariance_properties_raise_value_error():
    """Covariances must be symmetric positive semidefinite."""
    with pytest.raises(ValueError):
        KalmanFilter(
            A=np.eye(2),
            B=[[0.0], [1.0]],
            C=[[1.0, 0.0]],
            Q=[[1.0, 2.0], [0.0, 1.0]],
            R=[[1.0]],
            x_hat=[0.0, 0.0],
            P=np.eye(2),
        )

    with pytest.raises(ValueError):
        KalmanFilter(
            A=np.eye(2),
            B=[[0.0], [1.0]],
            C=[[1.0, 0.0]],
            Q=np.eye(2),
            R=[[-1.0]],
            x_hat=[0.0, 0.0],
            P=np.eye(2),
        )

    with pytest.raises(ValueError):
        KalmanFilter(
            A=np.eye(2),
            B=[[0.0], [1.0]],
            C=[[1.0, 0.0]],
            Q=np.eye(2),
            R=[[1.0]],
            x_hat=[0.0, 0.0],
            P=[[1.0, 0.0], [0.0, -1.0]],
        )


def test_update_uses_symmetric_covariance_form():
    """Joseph-form updates should leave covariance symmetric."""
    kalman_filter = KalmanFilter(
        A=np.eye(2),
        B=np.zeros((2, 1)),
        C=np.array([[1.0, 1e-6]]),
        Q=np.eye(2) * 1e-9,
        R=np.array([[1e-6]]),
        x_hat=np.array([0.0, 0.0]),
        P=np.array([[1.0, 0.2], [0.2, 2.0]]),
    )

    kalman_filter.predict()
    kalman_filter.update([1.0])

    assert np.allclose(kalman_filter.P, kalman_filter.P.T)


def test_discretize_state_space_returns_expected_shapes():
    """Discretization should preserve state and input dimensions."""
    A = np.array([[0.0, 1.0], [-2.0, -0.5]])
    B = np.array([[0.0], [1.0]])

    A_d, B_d = discretize_state_space(A, B, dt=0.01)

    assert A_d.shape == (2, 2)
    assert B_d.shape == (2, 1)


def test_discretize_state_space_invalid_dt_raises_value_error():
    """Sample time must be positive."""
    with pytest.raises(ValueError):
        discretize_state_space([[1.0]], [[1.0]], dt=0.0)


def test_estimation_error_decreases_for_simple_noisy_measurements():
    """For a constant state, the filter should converge toward the true value."""
    rng = np.random.default_rng(0)
    true_value = 1.0
    measurements = true_value + rng.normal(0.0, 0.2, 80)
    kalman_filter = KalmanFilter(
        A=[[1.0]],
        B=[[0.0]],
        C=[[1.0]],
        Q=[[1e-4]],
        R=[[0.04]],
        x_hat=[0.0],
        P=[[1.0]],
    )

    errors = []
    for measurement in measurements:
        x_hat, _ = kalman_filter.step(measurement)
        errors.append(abs(true_value - x_hat[0]))

    late_error = np.mean(errors[-10:])

    assert late_error < 0.1
    assert errors[-1] < abs(true_value)


def test_filter_error_is_lower_than_raw_measurement_error():
    """A tuned scalar Kalman filter should smooth noisy direct measurements."""
    rng = np.random.default_rng(0)
    true_value = 1.0
    measurements = true_value + rng.normal(0.0, 0.2, 80)
    kalman_filter = KalmanFilter(
        A=[[1.0]],
        B=[[0.0]],
        C=[[1.0]],
        Q=[[1e-4]],
        R=[[0.04]],
        x_hat=[0.0],
        P=[[1.0]],
    )

    estimates = []
    for measurement in measurements:
        x_hat, _ = kalman_filter.step(measurement)
        estimates.append(x_hat[0])

    estimate_error = np.mean(np.abs(true_value - np.array(estimates[-40:])))
    measurement_error = np.mean(np.abs(true_value - measurements[-40:]))

    assert estimate_error < measurement_error
    assert np.all(np.isfinite(kalman_filter.P))
    assert np.allclose(kalman_filter.P, kalman_filter.P.T)
    assert np.min(np.linalg.eigvalsh(kalman_filter.P)) >= -1e-12
