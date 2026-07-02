# Control Systems

The control examples show how feedback changes dynamic response, using simple
plants and standard metrics rather than production controller design.

## Governing Ideas

PID control computes a command from tracking error:

```text
e[k] = r[k] - y[k]
u[k] = Kp e[k] + Ki sum(e[k] dt) + Kd de/dt
```

The repository's discrete PID uses derivative-on-measurement and output
saturation with anti-windup.

LQR uses a linear state-space model:

```text
x_dot = A x + B u
u = -K x
```

The gain minimizes the quadratic cost:

```text
integral(x.T Q x + u.T R u) dt
```

MPC uses a discrete model:

```text
x[k+1] = A x[k] + B u[k]
```

At each step, it solves a finite-horizon constrained optimization and applies
only the first control input.

Step response metrics summarize behavior with rise time, settling time,
overshoot, peak value, peak time, final value, and steady-state estimate.

## Assumptions

- PID examples use sampled control and simplified actuator limits.
- LQR assumes a linearized or linear state-space model and full-state feedback.
- MPC examples are small linear systems with quadratic cost and simple bounds.
- Tuning choices are educational, not certified for safety-critical use.

## Numerical Method

- PID updates at a fixed sample time and holds the command between updates.
- LQR solves the continuous algebraic Riccati equation.
- MPC uses SciPy optimization for a finite input sequence.
- Step metrics are computed directly from sampled response arrays.

## What the Repository Demonstrates

- First- and second-order step response behavior.
- PI/PID motor speed control, saturation, anti-windup, and disturbance
  rejection.
- LQR stabilization for an inverted pendulum near the upright equilibrium.
- Constrained MPC tracking for a double-integrator position model.
- Shared response metrics for comparing transient behavior.

## Relevant Files and Examples

- `models/discrete_pid.py`
- `models/first_order_control.py`
- `models/second_order_control.py`
- `models/dc_motor.py`
- `models/inverted_pendulum_lqr.py`
- `analysis/step_response.py`
- `analysis/lqr.py`
- `analysis/model_predictive_control.py`
- `examples/run_discrete_pid_motor.py`
- `examples/run_inverted_pendulum_lqr_comparison.py`
- `examples/run_mpc_double_integrator.py`

## Limitations

- Controllers are not tuned for hardware deployment.
- LQR validity is local when applied to nonlinear systems.
- MPC is intentionally lightweight and not optimized for large horizons.
- Actuator dynamics, delays, sensor faults, and robustness margins are limited.
