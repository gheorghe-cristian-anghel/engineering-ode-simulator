import numpy as np
import pytest
from numpy.testing import assert_allclose

from analysis.state_space import (
    dc_motor_state_space,
    mass_spring_damper_state_space,
    rlc_state_space,
    simulate_state_space,
    step_input,
)


def test_step_input_returns_constant_value():
    """Step input should return the same value for any time."""
    input_func = step_input(5.0)

    assert input_func(0.0) == pytest.approx(5.0)
    assert input_func(10.0) == pytest.approx(5.0)


def test_mass_spring_damper_state_space_matrices():
    """Mass-spring-damper helper should return expected matrices."""
    A, B, C, D = mass_spring_damper_state_space(m=1.0, c=0.4, k=4.0)

    assert A.shape == (2, 2)
    assert B.shape == (2, 1)
    assert C.shape == (1, 2)
    assert D.shape == (1, 1)
    assert_allclose(A, [[0.0, 1.0], [-4.0, -0.4]])
    assert_allclose(B, [[0.0], [1.0]])
    assert_allclose(C, [[1.0, 0.0]])
    assert_allclose(D, [[0.0]])


def test_rlc_state_space_matrices():
    """RLC helper should return expected matrices."""
    A, B, C, D = rlc_state_space(R=2.0, L=1.0, C_value=0.25)

    assert A.shape == (2, 2)
    assert B.shape == (2, 1)
    assert C.shape == (1, 2)
    assert D.shape == (1, 1)
    assert_allclose(A, [[0.0, 4.0], [-1.0, -2.0]])
    assert_allclose(B, [[0.0], [1.0]])
    assert_allclose(C, [[1.0, 0.0]])
    assert_allclose(D, [[0.0]])


def test_dc_motor_state_space_matrices():
    """DC motor helper should return expected matrices."""
    A, B, C, D = dc_motor_state_space(
        R=1.0,
        L=0.5,
        J=0.01,
        b=0.001,
        Kt=0.01,
        Ke=0.01,
    )

    assert A.shape == (2, 2)
    assert B.shape == (2, 1)
    assert C.shape == (1, 2)
    assert D.shape == (1, 1)
    assert_allclose(A, [[-2.0, -0.02], [1.0, -0.1]])
    assert_allclose(B, [[2.0], [0.0]])
    assert_allclose(C, [[0.0, 1.0]])
    assert_allclose(D, [[0.0]])


def test_simulate_state_space_returns_expected_shapes():
    """Simulation should return time, state, and output arrays with expected shapes."""
    A = [[-1.0]]
    B = [[1.0]]
    C = [[1.0]]
    D = [[0.0]]

    t, x, y = simulate_state_space(
        A,
        B,
        C,
        D,
        step_input(1.0),
        [0.0],
        (0.0, 5.0),
        num_points=50,
    )

    assert t.shape == (50,)
    assert x.shape == (50, 1)
    assert y.shape == (50, 1)


def test_simulate_state_space_preserves_initial_state():
    """The first state sample should equal x0."""
    A = [[-1.0]]
    B = [[1.0]]
    C = [[1.0]]
    D = [[0.0]]

    t, x, y = simulate_state_space(
        A,
        B,
        C,
        D,
        step_input(0.0),
        [2.0],
        (0.0, 5.0),
        num_points=50,
    )

    assert t[0] == pytest.approx(0.0)
    assert x[0, 0] == pytest.approx(2.0)
    assert y[0, 0] == pytest.approx(2.0)


def test_invalid_matrix_dimensions_raise_value_error():
    """Incompatible state-space matrix dimensions should raise ValueError."""
    with pytest.raises(ValueError):
        simulate_state_space(
            [[0.0, 1.0]],
            [[1.0]],
            [[1.0, 0.0]],
            [[0.0]],
            step_input(1.0),
            [0.0],
            (0.0, 1.0),
        )


def test_invalid_t_span_raises_value_error():
    """Final time must be greater than initial time."""
    with pytest.raises(ValueError):
        simulate_state_space(
            [[-1.0]],
            [[1.0]],
            [[1.0]],
            [[0.0]],
            step_input(1.0),
            [0.0],
            (1.0, 0.0),
        )


def test_invalid_num_points_raises_value_error():
    """Number of time samples must be positive."""
    with pytest.raises(ValueError):
        simulate_state_space(
            [[-1.0]],
            [[1.0]],
            [[1.0]],
            [[0.0]],
            step_input(1.0),
            [0.0],
            (0.0, 1.0),
            num_points=0,
        )
