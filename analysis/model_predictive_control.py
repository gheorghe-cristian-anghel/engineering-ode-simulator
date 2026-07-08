"""Educational linear Model Predictive Control utilities."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


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


def _validate_optional_bounds(lower, upper, name, expected_length):
    """Validate optional lower and upper bounds."""
    lower_vector = None
    upper_vector = None

    if lower is not None:
        lower_vector = _as_vector(lower, f"{name}_min", expected_length)

    if upper is not None:
        upper_vector = _as_vector(upper, f"{name}_max", expected_length)

    if lower_vector is not None and upper_vector is not None:
        if np.any(lower_vector > upper_vector):
            raise ValueError(f"{name}_min must be less than or equal to {name}_max")

    return lower_vector, upper_vector


def _validate_horizon(horizon):
    """Validate and return the prediction horizon."""
    if not isinstance(horizon, (int, np.integer)):
        raise ValueError("horizon must be a positive integer")

    if horizon <= 0:
        raise ValueError("horizon must be a positive integer")

    return int(horizon)


def _validate_symmetric(matrix, name):
    """Validate that a matrix is symmetric within numerical tolerance."""
    if not np.allclose(matrix, matrix.T):
        raise ValueError(f"{name} must be symmetric")


def _validate_positive_semidefinite(matrix, name):
    """Validate that a symmetric matrix is positive semidefinite."""
    eigenvalues = np.linalg.eigvalsh(matrix)

    if np.min(eigenvalues) < -1e-10:
        raise ValueError(f"{name} must be positive semidefinite")


def _validate_positive_definite(matrix, name):
    """Validate that a symmetric matrix is positive definite."""
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} must be positive definite") from exc


def discrete_double_integrator(dt=0.1):
    """Return discrete-time double-integrator matrices.

    The state is ``[position, velocity]`` and the input is acceleration.
    """
    dt = float(dt)

    if not np.isfinite(dt):
        raise ValueError("dt must be finite")

    if dt <= 0:
        raise ValueError("dt must be positive")

    A = np.array([[1.0, dt], [0.0, 1.0]])
    B = np.array([[0.5 * dt**2], [dt]])

    return A, B


@dataclass
class LinearMPCSolution:
    """Result from one finite-horizon MPC optimization."""

    control: np.ndarray
    optimal_sequence: np.ndarray
    predicted_trajectory: np.ndarray
    success: bool
    cost: float
    message: str


@dataclass
class MPCResponseMetrics:
    """Summary metrics for an MPC tracking response."""

    final_position: float
    final_velocity: float
    final_position_error: float
    rms_position_error: float
    max_abs_control: float
    number_of_steps: int


class LinearMPC:
    """Small linear Model Predictive Controller for discrete-time systems.

    The model is ``x[k+1] = A*x[k] + B*u[k]``. At each solve, the controller
    finds a sequence of future inputs that minimizes state-tracking error and
    input effort over a finite horizon.
    """

    def __init__(
        self,
        A,
        B,
        Q,
        R,
        horizon,
        u_min=None,
        u_max=None,
        x_min=None,
        x_max=None,
    ):
        """Validate and store linear MPC matrices, weights, and constraints."""
        self.A, self.B, self.Q, self.R = self._validate_matrices(A, B, Q, R)
        self.horizon = _validate_horizon(horizon)
        self.n_states = self.A.shape[0]
        self.n_inputs = self.B.shape[1]
        self.u_min, self.u_max = _validate_optional_bounds(
            u_min,
            u_max,
            "u",
            self.n_inputs,
        )
        self.x_min, self.x_max = _validate_optional_bounds(
            x_min,
            x_max,
            "x",
            self.n_states,
        )
        self.last_solution = None

    @staticmethod
    def _validate_matrices(A, B, Q, R):
        """Validate linear system and cost matrix dimensions."""
        A = _as_2d_matrix(A, "A")
        B = _as_2d_matrix(B, "B")
        Q = _as_2d_matrix(Q, "Q")
        R = _as_2d_matrix(R, "R")

        n_states = A.shape[0]
        n_inputs = B.shape[1]

        if A.shape[0] != A.shape[1]:
            raise ValueError("A must be square")

        if B.shape[0] != n_states:
            raise ValueError("B row count must match A state count")

        if Q.shape != A.shape:
            raise ValueError("Q shape must match A shape")

        if R.shape != (n_inputs, n_inputs):
            raise ValueError("R shape must be (number of inputs, number of inputs)")

        _validate_symmetric(Q, "Q")
        _validate_symmetric(R, "R")
        _validate_positive_semidefinite(Q, "Q")
        _validate_positive_definite(R, "R")

        return A, B, Q, R

    def _input_sequence(self, u_sequence):
        """Return a validated input sequence with shape (horizon, n_inputs)."""
        inputs = np.asarray(u_sequence, dtype=float)

        if inputs.ndim == 1:
            expected_length = self.horizon * self.n_inputs
            if len(inputs) != expected_length:
                raise ValueError(
                    "u_sequence length must be horizon * number of inputs"
                )
            inputs = inputs.reshape(self.horizon, self.n_inputs)

        if inputs.ndim != 2:
            raise ValueError("u_sequence must be one- or two-dimensional")

        if inputs.shape != (self.horizon, self.n_inputs):
            raise ValueError("u_sequence shape must be (horizon, number of inputs)")

        if not np.all(np.isfinite(inputs)):
            raise ValueError("u_sequence must contain only finite values")

        return inputs

    def _reference_trajectory(self, x_ref):
        """Return a reference trajectory with shape (horizon + 1, n_states)."""
        reference = np.asarray(x_ref, dtype=float)

        if reference.ndim == 1:
            if len(reference) != self.n_states:
                raise ValueError("x_ref length must match the number of states")
            reference = np.tile(reference, (self.horizon + 1, 1))
        elif reference.ndim == 2:
            if reference.shape == (self.horizon, self.n_states):
                reference = np.vstack([reference, reference[-1]])
            elif reference.shape != (self.horizon + 1, self.n_states):
                raise ValueError(
                    "x_ref shape must be (states,), (horizon, states), "
                    "or (horizon + 1, states)"
                )
        else:
            raise ValueError("x_ref must be one- or two-dimensional")

        if not np.all(np.isfinite(reference)):
            raise ValueError("x_ref must contain only finite values")

        return reference

    def _optimizer_bounds(self):
        """Return optimizer bounds for the flattened input sequence."""
        bounds = []

        for _ in range(self.horizon):
            for input_index in range(self.n_inputs):
                lower = None if self.u_min is None else self.u_min[input_index]
                upper = None if self.u_max is None else self.u_max[input_index]
                bounds.append((lower, upper))

        return bounds

    def _initial_guess(self):
        """Return a feasible zero-centered input guess when possible."""
        guess = np.zeros((self.horizon, self.n_inputs))

        if self.u_min is not None:
            guess = np.maximum(guess, self.u_min)

        if self.u_max is not None:
            guess = np.minimum(guess, self.u_max)

        return guess.reshape(-1)

    def _state_constraints(self, x0, x_ref):
        """Return simple state bound constraints for SciPy SLSQP."""
        constraints = []

        if self.x_min is not None:
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda flat_inputs: (
                        self.predict_trajectory(x0, flat_inputs)[1:] - self.x_min
                    ).reshape(-1),
                }
            )

        if self.x_max is not None:
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda flat_inputs: (
                        self.x_max - self.predict_trajectory(x0, flat_inputs)[1:]
                    ).reshape(-1),
                }
            )

        return constraints

    def predict_trajectory(self, x0, u_sequence):
        """Predict states for an initial condition and input sequence."""
        state = _as_vector(x0, "x0", self.n_states)
        inputs = self._input_sequence(u_sequence)
        trajectory = np.zeros((self.horizon + 1, self.n_states))
        trajectory[0] = state

        for index in range(self.horizon):
            trajectory[index + 1] = self.A @ trajectory[index] + self.B @ inputs[index]

        return trajectory

    def cost(self, u_sequence, x0, x_ref):
        """Return the finite-horizon quadratic tracking cost.

        The final predicted state is intentionally weighted once in the stage
        loop and once as an additional terminal penalty using the same Q.
        """
        inputs = self._input_sequence(u_sequence)
        trajectory = self.predict_trajectory(x0, inputs)
        reference = self._reference_trajectory(x_ref)
        total_cost = 0.0

        for index in range(self.horizon):
            state_error = trajectory[index + 1] - reference[index + 1]
            input_vector = inputs[index]
            total_cost += float(state_error.T @ self.Q @ state_error)
            total_cost += float(input_vector.T @ self.R @ input_vector)

        terminal_error = trajectory[-1] - reference[-1]
        total_cost += float(terminal_error.T @ self.Q @ terminal_error)

        return total_cost

    def solve(self, x0, x_ref):
        """Solve one MPC optimization problem.

        Returns a :class:`LinearMPCSolution` containing the first input,
        optimized input sequence, predicted trajectory, success flag, final
        cost, and optimizer message.
        """
        initial_state = _as_vector(x0, "x0", self.n_states)
        self._reference_trajectory(x_ref)
        initial_guess = self._initial_guess()
        bounds = self._optimizer_bounds()
        constraints = self._state_constraints(initial_state, x_ref)
        method = "SLSQP" if constraints else "L-BFGS-B"

        result = minimize(
            lambda flat_inputs: self.cost(flat_inputs, initial_state, x_ref),
            initial_guess,
            method=method,
            bounds=bounds,
            constraints=constraints,
        )

        optimal_sequence = self._input_sequence(result.x)
        predicted_trajectory = self.predict_trajectory(initial_state, optimal_sequence)
        solution = LinearMPCSolution(
            control=optimal_sequence[0].copy(),
            optimal_sequence=optimal_sequence,
            predicted_trajectory=predicted_trajectory,
            success=bool(result.success),
            cost=float(result.fun),
            message=str(result.message),
        )
        self.last_solution = solution

        return solution

    def step(self, x0, x_ref):
        """Return the first optimized control input for the current state."""
        return self.solve(x0, x_ref).control


def simulate_mpc_tracking(mpc, x0, x_ref, num_steps=100):
    """Run a receding-horizon MPC tracking simulation."""
    if not isinstance(mpc, LinearMPC):
        raise ValueError("mpc must be a LinearMPC instance")

    if not isinstance(num_steps, int) or num_steps <= 0:
        raise ValueError("num_steps must be a positive integer")

    state = _as_vector(x0, "x0", mpc.n_states)

    if np.asarray(x_ref, dtype=float).ndim == 2:
        reference_array = np.asarray(x_ref, dtype=float)
        if reference_array.shape == (num_steps + 1, mpc.n_states):
            references = reference_array.copy()
        else:
            references = mpc._reference_trajectory(x_ref)
            references = np.tile(references[0], (num_steps + 1, 1))
    else:
        reference = mpc._reference_trajectory(x_ref)[0]
        references = np.tile(reference, (num_steps + 1, 1))

    states = np.zeros((num_steps + 1, mpc.n_states))
    controls = np.zeros((num_steps, mpc.n_inputs))
    costs = np.zeros(num_steps)
    success = np.zeros(num_steps, dtype=bool)
    predicted_trajectories = np.zeros(
        (num_steps, mpc.horizon + 1, mpc.n_states)
    )
    step_indices = np.arange(num_steps + 1)
    states[0] = state

    for index in range(num_steps):
        horizon_reference = references[index : index + mpc.horizon + 1]
        if len(horizon_reference) < mpc.horizon + 1:
            pad_count = mpc.horizon + 1 - len(horizon_reference)
            horizon_reference = np.vstack(
                [
                    horizon_reference,
                    np.tile(horizon_reference[-1], (pad_count, 1)),
                ]
            )

        solution = mpc.solve(states[index], horizon_reference)
        controls[index] = solution.control
        costs[index] = solution.cost
        success[index] = solution.success
        predicted_trajectories[index] = solution.predicted_trajectory
        states[index + 1] = mpc.A @ states[index] + mpc.B @ controls[index]

    return {
        "states": states,
        "controls": controls,
        "references": references,
        "steps": step_indices,
        "costs": costs,
        "success": success,
        "predicted_trajectories": predicted_trajectories,
    }


def summarize_mpc_response(result):
    """Return practical tracking metrics for a position-control MPC result."""
    states = np.asarray(result["states"], dtype=float)
    controls = np.asarray(result["controls"], dtype=float)
    references = np.asarray(result["references"], dtype=float)

    if states.ndim != 2 or states.shape[1] < 2:
        raise ValueError("states must have shape (samples, at least 2 states)")

    if controls.ndim != 2:
        raise ValueError("controls must be a two-dimensional array")

    if references.shape != states.shape:
        raise ValueError("references shape must match states shape")

    if len(states) != len(controls) + 1:
        raise ValueError("states length must be controls length plus one")

    position_error = references[:, 0] - states[:, 0]

    return MPCResponseMetrics(
        final_position=float(states[-1, 0]),
        final_velocity=float(states[-1, 1]),
        final_position_error=float(position_error[-1]),
        rms_position_error=float(np.sqrt(np.mean(position_error**2))),
        max_abs_control=float(np.max(np.abs(controls))) if controls.size else 0.0,
        number_of_steps=int(len(controls)),
    )
