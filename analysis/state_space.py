"""Reusable continuous-time state-space simulation utilities."""

import numpy as np
from scipy.integrate import solve_ivp


def _as_2d_matrix(matrix, name):
    """Return a validated two-dimensional float matrix."""
    matrix_array = np.asarray(matrix, dtype=float)

    if matrix_array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix")

    if not np.all(np.isfinite(matrix_array)):
        raise ValueError(f"{name} must contain only finite values")

    return matrix_array


def _as_state_vector(x0, n_states):
    """Return a validated initial state vector."""
    initial_state = np.asarray(x0, dtype=float)

    if initial_state.ndim != 1:
        raise ValueError("x0 must be a one-dimensional state vector")

    if len(initial_state) != n_states:
        raise ValueError("x0 length must match the number of states")

    if not np.all(np.isfinite(initial_state)):
        raise ValueError("x0 must contain only finite values")

    return initial_state


def validate_state_space_matrices(A, B, C, D):
    """Validate continuous-time state-space matrix dimensions.

    The expected model form is ``x_dot = A*x + B*u`` and ``y = C*x + D*u``.
    """
    A = _as_2d_matrix(A, "A")
    B = _as_2d_matrix(B, "B")
    C = _as_2d_matrix(C, "C")
    D = _as_2d_matrix(D, "D")

    n_states = A.shape[0]

    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")

    if B.shape[0] != n_states:
        raise ValueError("B row count must match A state count")

    if C.shape[1] != n_states:
        raise ValueError("C column count must match A state count")

    if D.shape != (C.shape[0], B.shape[1]):
        raise ValueError("D shape must be (number of outputs, number of inputs)")

    return A, B, C, D


def _evaluate_input(u_func, t, n_inputs):
    """Return the input vector at time t."""
    input_value = u_func(t)
    input_array = np.asarray(input_value, dtype=float)

    if input_array.ndim == 0:
        input_array = input_array.reshape(1)

    if input_array.ndim != 1:
        raise ValueError("u_func must return a scalar or one-dimensional input")

    if len(input_array) != n_inputs:
        raise ValueError("u_func output length must match the number of inputs")

    if not np.all(np.isfinite(input_array)):
        raise ValueError("u_func must return only finite values")

    return input_array


def simulate_state_space(A, B, C, D, u_func, x0, t_span, num_points=1000):
    """Simulate a continuous-time linear state-space model.

    Parameters
    ----------
    A, B, C, D : array-like
        State-space matrices for ``x_dot = A*x + B*u`` and ``y = C*x + D*u``.
    u_func : callable
        Function returning the input ``u(t)`` as a scalar or one-dimensional
        array-like value.
    x0 : array-like
        Initial state vector.
    t_span : tuple
        Start and end time in seconds, for example ``(0, 10)``.
    num_points : int, optional
        Number of time samples to evaluate.

    Returns
    -------
    tuple
        ``(t, x, y)`` where ``x`` has shape ``(num_points, n_states)`` and
        ``y`` has shape ``(num_points, n_outputs)``.
    """
    if not callable(u_func):
        raise ValueError("u_func must be callable")

    A, B, C, D = validate_state_space_matrices(A, B, C, D)
    initial_state = _as_state_vector(x0, A.shape[0])

    if num_points <= 0:
        raise ValueError("num_points must be positive")

    if t_span[1] <= t_span[0]:
        raise ValueError("t_span final time must be greater than initial time")

    n_inputs = B.shape[1]

    def state_space_ode(t, state):
        input_vector = _evaluate_input(u_func, t, n_inputs)
        return A @ state + B @ input_vector

    t_eval = np.linspace(t_span[0], t_span[1], num_points)
    solution = solve_ivp(
        state_space_ode,
        t_span,
        initial_state,
        t_eval=t_eval,
    )

    x = solution.y.T
    y = np.zeros((len(solution.t), C.shape[0]))

    for index, time in enumerate(solution.t):
        input_vector = _evaluate_input(u_func, time, n_inputs)
        y[index] = C @ x[index] + D @ input_vector

    return solution.t, x, y


def step_input(amplitude=1.0):
    """Return a callable constant step input ``u(t) = amplitude``."""
    return lambda t: amplitude


def mass_spring_damper_state_space(m=1.0, c=0.5, k=2.0):
    """Return state-space matrices for a mass-spring-damper system.

    States are displacement and velocity, input is external force, and output
    is displacement.
    """
    if m <= 0:
        raise ValueError("m must be positive")

    if c < 0:
        raise ValueError("c must be nonnegative")

    if k <= 0:
        raise ValueError("k must be positive")

    A = np.array([[0.0, 1.0], [-k / m, -c / m]])
    B = np.array([[0.0], [1.0 / m]])
    C = np.array([[1.0, 0.0]])
    D = np.array([[0.0]])

    return A, B, C, D


def rlc_state_space(R=2.0, L=1.0, C_value=0.25):
    """Return state-space matrices for series RLC capacitor voltage output.

    States are capacitor voltage and current, input is source voltage, and
    output is capacitor voltage.
    """
    if R < 0:
        raise ValueError("R must be nonnegative")

    if L <= 0:
        raise ValueError("L must be positive")

    if C_value <= 0:
        raise ValueError("C_value must be positive")

    A = np.array([[0.0, 1.0 / C_value], [-1.0 / L, -R / L]])
    B = np.array([[0.0], [1.0 / L]])
    C = np.array([[1.0, 0.0]])
    D = np.array([[0.0]])

    return A, B, C, D


def dc_motor_state_space(R=1.0, L=0.5, J=0.01, b=0.001, Kt=0.01, Ke=0.01):
    """Return state-space matrices for a permanent-magnet DC motor.

    States are armature current and angular speed, input is applied voltage,
    and output is angular speed.
    """
    if R <= 0:
        raise ValueError("R must be positive")

    if L <= 0:
        raise ValueError("L must be positive")

    if J <= 0:
        raise ValueError("J must be positive")

    if b < 0:
        raise ValueError("b must be nonnegative")

    if Kt <= 0:
        raise ValueError("Kt must be positive")

    if Ke <= 0:
        raise ValueError("Ke must be positive")

    A = np.array([[-R / L, -Ke / L], [Kt / J, -b / J]])
    B = np.array([[1.0 / L], [0.0]])
    C = np.array([[0.0, 1.0]])
    D = np.array([[0.0]])

    return A, B, C, D
