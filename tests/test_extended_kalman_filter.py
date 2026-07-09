import numpy as np
import pytest

from analysis.extended_kalman_filter import (
    ExtendedKalmanFilter,
    pendulum_angle_measurement,
    pendulum_angle_measurement_jacobian,
    pendulum_ekf_state_jacobian,
    pendulum_ekf_state_transition,
)


def _simple_nonlinear_filter():
    """Return a simple one-state EKF for tests."""
    return ExtendedKalmanFilter(
        f=lambda x, u, dt: np.array([x[0] + dt * (-0.5 * x[0])]),
        h=lambda x: np.array([x[0] ** 2]),
        F_jacobian=lambda x, u, dt: np.array([[1.0 - 0.5 * dt]]),
        H_jacobian=lambda x: np.array([[2.0 * x[0]]]),
        Q=np.array([[0.001]]),
        R=np.array([[0.01]]),
        x_hat=np.array([0.8]),
        P=np.array([[1.0]]),
    )


def test_extended_kalman_filter_stores_expected_shapes():
    """EKF should store validated arrays with expected dimensions."""
    ekf = _simple_nonlinear_filter()

    assert ekf.Q.shape == (1, 1)
    assert ekf.R.shape == (1, 1)
    assert ekf.x_hat.shape == (1,)
    assert ekf.P.shape == (1, 1)


def test_predict_returns_state_estimate_with_correct_shape():
    """Predict should preserve state vector shape."""
    ekf = _simple_nonlinear_filter()

    x_hat = ekf.predict(dt=0.01)

    assert x_hat.shape == (1,)


def test_update_returns_state_estimate_with_correct_shape():
    """Update should preserve state vector and Kalman gain shapes."""
    ekf = _simple_nonlinear_filter()
    ekf.predict(dt=0.01)

    x_hat, gain = ekf.update([0.5])

    assert x_hat.shape == (1,)
    assert gain.shape == (1, 1)


def test_step_works_for_scalar_measurement():
    """Scalar measurements should be accepted for single-output systems."""
    ekf = _simple_nonlinear_filter()

    x_hat, gain = ekf.step(0.5, dt=0.01)

    assert x_hat.shape == (1,)
    assert gain.shape == (1, 1)


def test_nonpositive_dt_raises_value_error():
    """Sample time must be positive."""
    ekf = _simple_nonlinear_filter()

    with pytest.raises(ValueError):
        ekf.predict(dt=0.0)


def test_invalid_q_shape_raises_value_error():
    """Q must be square."""
    with pytest.raises(ValueError):
        ExtendedKalmanFilter(
            f=lambda x, u, dt: x,
            h=lambda x: x,
            F_jacobian=lambda x, u, dt: np.eye(2),
            H_jacobian=lambda x: np.eye(2),
            Q=np.array([[1.0, 0.0]]),
            R=np.eye(2),
            x_hat=np.array([0.0, 0.0]),
            P=np.eye(2),
        )


def test_invalid_r_shape_raises_value_error():
    """R must be square."""
    with pytest.raises(ValueError):
        ExtendedKalmanFilter(
            f=lambda x, u, dt: x,
            h=lambda x: x,
            F_jacobian=lambda x, u, dt: np.eye(2),
            H_jacobian=lambda x: np.eye(2),
            Q=np.eye(2),
            R=np.array([[1.0, 0.0]]),
            x_hat=np.array([0.0, 0.0]),
            P=np.eye(2),
        )


def test_invalid_p_shape_raises_value_error():
    """P shape must match Q shape."""
    with pytest.raises(ValueError):
        ExtendedKalmanFilter(
            f=lambda x, u, dt: x,
            h=lambda x: x,
            F_jacobian=lambda x, u, dt: np.eye(2),
            H_jacobian=lambda x: np.eye(2),
            Q=np.eye(2),
            R=np.eye(2),
            x_hat=np.array([0.0, 0.0]),
            P=np.eye(3),
        )


def test_invalid_covariance_properties_raise_value_error():
    """EKF covariances must be symmetric positive semidefinite."""
    with pytest.raises(ValueError):
        ExtendedKalmanFilter(
            f=lambda x, u, dt: x,
            h=lambda x: x,
            F_jacobian=lambda x, u, dt: np.eye(2),
            H_jacobian=lambda x: np.eye(2),
            Q=[[1.0, 2.0], [0.0, 1.0]],
            R=np.eye(2),
            x_hat=np.array([0.0, 0.0]),
            P=np.eye(2),
        )

    with pytest.raises(ValueError):
        ExtendedKalmanFilter(
            f=lambda x, u, dt: x,
            h=lambda x: x,
            F_jacobian=lambda x, u, dt: np.eye(2),
            H_jacobian=lambda x: np.eye(2),
            Q=np.eye(2),
            R=np.eye(2),
            x_hat=np.array([0.0, 0.0]),
            P=[[1.0, 0.0], [0.0, -1.0]],
        )


def test_update_uses_symmetric_covariance_form():
    """Joseph-form updates should leave EKF covariance symmetric."""
    ekf = _simple_nonlinear_filter()
    ekf.predict(dt=0.01)
    ekf.update([0.5])

    assert np.allclose(ekf.P, ekf.P.T)


def test_pendulum_jacobian_helpers_return_expected_shapes():
    """Pendulum EKF helpers should return expected state and output dimensions."""
    state = np.array([0.1, 0.2])

    next_state = pendulum_ekf_state_transition(state, dt=0.01)
    F = pendulum_ekf_state_jacobian(state, dt=0.01)
    measurement = pendulum_angle_measurement(state)
    H = pendulum_angle_measurement_jacobian(state)

    assert next_state.shape == (2,)
    assert F.shape == (2, 2)
    assert measurement.shape == (1,)
    assert H.shape == (1, 2)


def test_simple_nonlinear_estimation_error_remains_finite_and_bounded():
    """For a simple nonlinear measurement, EKF estimates should stay bounded."""
    true_state = 1.0
    dt = 0.05
    ekf = _simple_nonlinear_filter()

    errors = []
    for _ in range(80):
        true_state = true_state + dt * (-0.5 * true_state)
        measurement = true_state**2
        x_hat, _ = ekf.step(measurement, dt=dt)
        errors.append(abs(true_state - x_hat[0]))

    assert np.all(np.isfinite(errors))
    assert errors[-1] < 0.1


def test_ekf_outputs_and_covariance_remain_finite_and_symmetric():
    """Repeated EKF steps should keep outputs finite and covariance symmetric."""
    true_state = 1.0
    dt = 0.05
    ekf = _simple_nonlinear_filter()

    estimates = []
    for _ in range(40):
        true_state = true_state + dt * (-0.5 * true_state)
        measurement = true_state**2
        x_hat, gain = ekf.step(measurement, dt=dt)
        estimates.append(x_hat[0])

        assert np.all(np.isfinite(gain))
        assert np.all(np.isfinite(ekf.P))
        assert np.allclose(ekf.P, ekf.P.T)
        assert np.min(np.linalg.eigvalsh(ekf.P)) >= -1e-12

    assert np.all(np.isfinite(estimates))
