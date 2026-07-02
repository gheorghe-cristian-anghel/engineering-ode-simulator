import numpy as np
import pytest

from analysis.model_predictive_control import (
    LinearMPC,
    discrete_double_integrator,
    simulate_mpc_tracking,
    summarize_mpc_response,
)


def _simple_mpc():
    """Return a small constrained double-integrator MPC for tests."""
    A, B = discrete_double_integrator(dt=0.1)
    return LinearMPC(
        A=A,
        B=B,
        Q=np.diag([10.0, 1.0]),
        R=np.array([[0.1]]),
        horizon=10,
        u_min=[-2.0],
        u_max=[2.0],
    )


def test_discrete_double_integrator_returns_expected_shapes():
    """Double-integrator helper should return compatible A and B matrices."""
    A, B = discrete_double_integrator(dt=0.1)

    assert A.shape == (2, 2)
    assert B.shape == (2, 1)
    assert A[0, 1] == pytest.approx(0.1)
    assert B[0, 0] == pytest.approx(0.005)
    assert B[1, 0] == pytest.approx(0.1)


def test_invalid_dt_raises_value_error():
    """Sample time must be positive."""
    with pytest.raises(ValueError):
        discrete_double_integrator(dt=0.0)


def test_linear_mpc_initializes_with_correct_dimensions():
    """LinearMPC should store validated dimensions."""
    mpc = _simple_mpc()

    assert mpc.n_states == 2
    assert mpc.n_inputs == 1
    assert mpc.horizon == 10


def test_invalid_a_shape_raises_value_error():
    """A must be square."""
    with pytest.raises(ValueError):
        LinearMPC(
            A=[[1.0, 0.1]],
            B=[[0.0], [0.1]],
            Q=np.eye(2),
            R=np.eye(1),
            horizon=10,
        )


def test_invalid_b_shape_raises_value_error():
    """B row count must match the state count."""
    with pytest.raises(ValueError):
        LinearMPC(
            A=np.eye(2),
            B=[[0.1]],
            Q=np.eye(2),
            R=np.eye(1),
            horizon=10,
        )


def test_invalid_q_shape_raises_value_error():
    """Q shape must match the state matrix shape."""
    with pytest.raises(ValueError):
        LinearMPC(
            A=np.eye(2),
            B=[[0.0], [0.1]],
            Q=np.eye(3),
            R=np.eye(1),
            horizon=10,
        )


def test_invalid_r_shape_raises_value_error():
    """R shape must match the number of inputs."""
    with pytest.raises(ValueError):
        LinearMPC(
            A=np.eye(2),
            B=[[0.0], [0.1]],
            Q=np.eye(2),
            R=np.eye(2),
            horizon=10,
        )


def test_invalid_horizon_raises_value_error():
    """Prediction horizon must be a positive integer."""
    A, B = discrete_double_integrator()

    with pytest.raises(ValueError):
        LinearMPC(A=A, B=B, Q=np.eye(2), R=np.eye(1), horizon=0)


def test_predict_trajectory_returns_expected_shape():
    """Predicted trajectory should include the initial state and horizon states."""
    mpc = _simple_mpc()
    u_sequence = np.zeros((mpc.horizon, mpc.n_inputs))

    trajectory = mpc.predict_trajectory([0.0, 0.0], u_sequence)

    assert trajectory.shape == (mpc.horizon + 1, mpc.n_states)


def test_cost_is_nonnegative():
    """Quadratic MPC cost should be nonnegative."""
    mpc = _simple_mpc()
    u_sequence = np.zeros((mpc.horizon, mpc.n_inputs))

    cost = mpc.cost(u_sequence, x0=[0.0, 0.0], x_ref=[1.0, 0.0])

    assert cost >= 0.0


def test_solve_returns_control_within_input_bounds():
    """MPC solve should respect configured acceleration limits."""
    mpc = _simple_mpc()

    solution = mpc.solve(x0=[0.0, 0.0], x_ref=[10.0, 0.0])

    assert solution.success
    assert solution.control.shape == (1,)
    assert solution.control[0] >= -2.0
    assert solution.control[0] <= 2.0


def test_step_returns_first_control_with_correct_shape():
    """step should return only the first optimized control input."""
    mpc = _simple_mpc()

    control = mpc.step(x0=[0.0, 0.0], x_ref=[10.0, 0.0])

    assert control.shape == (1,)


def test_simulate_mpc_tracking_returns_expected_shapes():
    """Receding-horizon simulation should return aligned response arrays."""
    mpc = _simple_mpc()
    result = simulate_mpc_tracking(
        mpc=mpc,
        x0=[0.0, 0.0],
        x_ref=[2.0, 0.0],
        num_steps=20,
    )

    assert result["states"].shape == (21, 2)
    assert result["controls"].shape == (20, 1)
    assert result["references"].shape == (21, 2)
    assert result["costs"].shape == (20,)
    assert result["success"].shape == (20,)
    assert result["predicted_trajectories"].shape == (20, 11, 2)


def test_mpc_moves_double_integrator_position_toward_target():
    """Closed-loop MPC should move the position closer to the reference."""
    mpc = _simple_mpc()
    result = simulate_mpc_tracking(
        mpc=mpc,
        x0=[0.0, 0.0],
        x_ref=[5.0, 0.0],
        num_steps=60,
    )

    initial_error = abs(result["references"][0, 0] - result["states"][0, 0])
    final_error = abs(result["references"][-1, 0] - result["states"][-1, 0])

    assert result["states"][-1, 0] > result["states"][0, 0]
    assert final_error < initial_error


def test_summarize_mpc_response_returns_expected_keys():
    """MPC metrics helper should expose the expected summary fields."""
    mpc = _simple_mpc()
    result = simulate_mpc_tracking(
        mpc=mpc,
        x0=[0.0, 0.0],
        x_ref=[2.0, 0.0],
        num_steps=20,
    )
    metrics = summarize_mpc_response(result)

    assert set(metrics.__dict__) == {
        "final_position",
        "final_velocity",
        "final_position_error",
        "rms_position_error",
        "max_abs_control",
        "number_of_steps",
    }
