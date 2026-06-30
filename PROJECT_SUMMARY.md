# Engineering ODE Simulator - Project Summary

## Purpose

Engineering ODE Simulator is a Python engineering simulation library for
modeling physical and control systems governed by ordinary differential
equations.

The project goals are to:

- learn scientific computing
- learn differential equation modeling
- learn numerical simulation
- build a portfolio project
- prepare for control systems, robotics, electrical engineering, and automation work

## Technology Stack

- Python
- NumPy
- SciPy
- Matplotlib
- Streamlit
- pytest
- Git/GitHub
- VS Code

Simulations use `scipy.integrate.solve_ivp` for numerical integration.

## Project Structure

- `models/`: engineering models and simulation functions
- `analysis/`: reusable analysis tools independent of one specific model
- `examples/`: runnable scripts that print parameters and plot results
- `streamlit_app.py`: browser UI for selected interactive simulations
- `tests/`: pytest coverage for model behavior and helper formulas
- `docs/`: longer documentation, equations, architecture notes, and future ideas
- `docs/screenshots/`: selected generated plots for documentation and portfolio presentation

## Coding Conventions

- One model per module.
- Each model has validation functions where appropriate.
- Model modules usually include an ODE function and a `simulate_*` function.
- Each model should have an example script and pytest tests.
- Use docstrings and beginner-readable code.
- Prefer clear engineering names over overly abstract names.
- Use SI units unless stated otherwise.
- Examples should print important parameters and plot results.
- Tests should check physics and engineering behavior, not only syntax.

## Implemented Models

| Model | Domain | Type / Order | Main File | Example | Test File | Status |
|---|---|---:|---|---|---|---|
| RC Circuit | Electrical | First-order | `models/rc_circuit.py` | `examples/run_rc_circuit.py` | `tests/test_rc_circuit.py` | Implemented |
| RL Circuit | Electrical | First-order | `models/rl_circuit.py` | `examples/run_rl_circuit.py` | `tests/test_rl_circuit.py` | Implemented |
| RLC Circuit | Electrical | Second-order | `models/rlc_circuit.py` | `examples/run_rlc_circuit.py` | `tests/test_rlc_circuit.py` | Implemented |
| Newton Cooling | Thermal | First-order | `models/cooling.py` | `examples/run_cooling.py` | `tests/test_cooling.py` | Implemented |
| Mass-Spring-Damper | Mechanical | Second-order | `models/mass_spring_damper.py` | `examples/run_mass_spring_damper.py` | `tests/test_mass_spring_damper.py` | Implemented |
| Pendulum | Mechanical | Nonlinear second-order | `models/pendulum.py` | `examples/run_pendulum.py` | `tests/test_pendulum.py` | Implemented |
| Inverted Pendulum / Cart-Pole | Control / Mechanical | Nonlinear fourth-order plus linearized state-space | `models/inverted_pendulum.py` | `examples/run_inverted_pendulum_open_loop.py` | `tests/test_inverted_pendulum.py` | Implemented |
| Inverted Pendulum LQR Control | Control / Mechanical | State-feedback control | `models/inverted_pendulum_lqr.py` | `examples/run_inverted_pendulum_lqr.py` | `tests/test_lqr.py` | Implemented |
| First-Order Control System | Control | First-order | `models/first_order_control.py` | `examples/run_first_order_control.py` | `tests/test_first_order_control.py` | Implemented |
| Second-Order Control System | Control | Second-order | `models/second_order_control.py` | `examples/run_second_order_control.py` | `tests/test_second_order_control.py` | Implemented |
| DC Motor | Electromechanical | Coupled first-order system | `models/dc_motor.py` | `examples/run_dc_motor.py` | `tests/test_dc_motor.py` | Implemented |
| PI Motor Speed Control | Control / Electromechanical | Closed-loop system | `models/pid_motor_control.py` | `examples/run_pid_motor_control.py` | `tests/test_pid_motor_control.py` | Implemented |
| PI Motor Load Disturbance | Control / Electromechanical | Disturbance response | `models/pid_motor_control.py` | `examples/run_motor_load_disturbance.py` | `tests/test_motor_load_disturbance.py` | Implemented |
| Discrete PID Motor Speed Control | Control / Electromechanical | Sampled-data control | `models/discrete_pid.py` | `examples/run_discrete_pid_motor.py` | `tests/test_discrete_pid.py` | Implemented |
| Discrete PID Load Disturbance | Control / Electromechanical | Sampled-data disturbance response | `models/discrete_pid.py` | `examples/run_discrete_pid_disturbance_response.py` | `tests/test_discrete_pid_disturbance.py` | Implemented |

Implemented control features include a reusable `DiscretePID` controller in
`models/discrete_pid.py`. It models embedded-style digital control with fixed
sample time `dt`, output saturation, anti-windup, and
derivative-on-measurement. The DC motor example is
`examples/run_discrete_pid_motor.py`, with tests in `tests/test_discrete_pid.py`.
The discrete PID disturbance response example uses the same sampled controller
and DC motor model with a step load torque disturbance.
Educational PID tuning examples compare P, PI, and PID behavior and show how
changing `Kp`, `Ki`, and `Kd` affects DC motor speed tracking.
DC motor load disturbance rejection examples compare fixed-voltage open-loop
behavior against continuous PI and discrete PID feedback control.

## Analysis Tools

Reusable step response metrics are implemented in `analysis/step_response.py`.
PI motor controller gain sweeps are implemented in
`analysis/parameter_sweep.py`.
DC motor load disturbance metrics are implemented in
`analysis/motor_disturbance.py`.
RLC parameter sweep helpers are implemented in `analysis/rlc_sweep.py` for
educational resistance, capacitance, and inductance transient-response studies.
PID tuning helpers for discrete motor control examples are implemented in
`analysis/pid_tuning.py`.
CSV export helpers are implemented in `analysis/export_utils.py`.
Frequency response and Bode plot helpers are implemented in
`analysis/frequency_response.py`.
Transfer function helpers for continuous-time step and impulse response
simulation are implemented in `analysis/transfer_function.py`.
Continuous-time state-space simulation helpers are implemented in
`analysis/state_space.py`.
Continuous-time LQR design helpers are implemented in `analysis/lqr.py`.

The Streamlit MVP in `streamlit_app.py` provides an interactive browser UI for
RC circuit charging, RLC circuit step response, and discrete PID motor speed
control while reusing the existing model and analysis modules.

The tool estimates:

- initial value
- final value
- steady-state value
- rise time
- settling time
- peak value
- peak time
- overshoot

The parameter sweep tool compares PI controller gain choices using:

- final value
- final tracking error
- peak value
- overshoot
- settling time
- maximum control effort
- maximum current

The RLC sweep helper compares component values using:

- natural frequency
- damping ratio
- final capacitor voltage
- peak voltage
- overshoot
- settling time

The PID tuning helper compares discrete PID controller gain choices using:

- final speed
- final tracking error
- peak speed
- overshoot
- settling time
- maximum voltage
- maximum current

The discrete PID disturbance helper summarizes:

- speed before disturbance
- minimum speed after disturbance
- speed drop
- recovery time when the response returns to the target band
- final speed and final error
- voltage and current before and after recovery or at the end

The motor disturbance helper summarizes open-loop and feedback disturbance
responses using:

- speed before disturbance
- minimum speed after disturbance
- speed drop
- final speed and final error
- recovery time
- optional voltage and current before/final values

The export utility validates column names, dimensions, and column lengths,
creates missing output directories automatically, and writes simulation arrays
to CSV files for later analysis in Excel, MATLAB, Python, or reports.

The frequency response utility computes continuous-time transfer-function
magnitude and phase data and provides a reusable Matplotlib Bode plot helper.
The transfer function utility provides a reusable model representation plus
step response, impulse response, and common low-pass transfer function helpers.
The state-space utility simulates linear systems in the form
`x_dot = A*x + B*u` and `y = C*x + D*u`.

Selected examples also save plot screenshots in `docs/screenshots/` as
documentation and portfolio presentation assets.

## Current Development Workflow

1. Study the engineering concept.
2. Derive the differential equation.
3. Convert to a first-order system for `solve_ivp`.
4. Implement the model module.
5. Add an example script.
6. Add tests.
7. Run pytest.
8. Run the example and inspect plots.
9. Update README/docs.
10. Commit and push to GitHub.

## Current Status

The project can simulate first-order systems, second-order systems, nonlinear
pendulum motion, open-loop inverted pendulum/cart-pole instability, LQR
stabilization of the nonlinear inverted pendulum near upright, open-loop DC
motor dynamics, and closed-loop PI motor speed control with disturbance
rejection. It also includes embedded-style discrete PID motor speed control
and discrete PID disturbance response, a Streamlit GUI MVP for selected
simulations,
reusable step response metrics, frequency response analysis, transfer function
utilities, state-space simulation utilities, PI gain sweep analysis, RLC
parameter sweep examples, DC motor disturbance rejection comparison examples,
PID tuning examples, and pytest coverage for implemented models.

## How Future AI Chats Should Use This File

Future ChatGPT/Codex chats should read `PROJECT_SUMMARY.md` and `ROADMAP.md`
before implementing new features. These files provide the current project
scope, coding style, implemented models, and planned direction.
