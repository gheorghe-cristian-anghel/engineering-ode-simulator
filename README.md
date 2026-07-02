# Engineering ODE Simulator

Engineering ODE Simulator is a portfolio project for modeling simple
engineering systems with ordinary differential equations (ODEs). The goal is
to pair numerical simulations with analytical solutions where possible, so the
results are easy to understand and verify.

The project currently includes:

- RC circuit charging
- Newton's Law of Cooling
- Mass-spring-damper free vibration
- First-order control system step response
- Second-order control system step response
- Reusable step response metrics
- RL circuit step response
- Series RLC circuit step response
- RLC parameter sweep examples for resistance, capacitance, and inductance
- Simple pendulum nonlinear dynamics
- Inverted pendulum / cart-pole nonlinear dynamics
- Linearized inverted pendulum upright state-space model
- LQR stabilization for the inverted pendulum
- Discrete Kalman filter state estimation examples
- Extended Kalman Filter nonlinear pendulum state estimation
- Unscented Kalman Filter nonlinear pendulum state estimation
- Particle Filter nonlinear pendulum state estimation
- Quadcopter altitude dynamics
- Quadcopter altitude PID control
- Quadcopter attitude dynamics
- Quadcopter attitude PID control
- Full 6-DOF quadcopter rigid-body dynamics
- Quadcopter trajectory tracking
- Quadcopter waypoint following
- 3D Matplotlib animation for 6-DOF quadcopter simulations
- DC motor speed response
- DC motor open-loop load disturbance response
- DC motor PI speed control
- PI motor gain sweep analysis
- DC motor PI load disturbance response
- Discrete PID motor speed control
- Discrete PID disturbance response for DC motor speed control
- DC motor disturbance rejection comparison
- Educational PID tuning examples
- Interactive Streamlit GUI for selected simulations
- Matplotlib animation examples for inverted pendulum motion
- Matplotlib 3D animation examples for quadcopter motion
- Frequency response and Bode plot examples
- Transfer Function Utilities
  - Reusable continuous-time transfer function representation
  - Step response simulation
  - Impulse response simulation
  - Common engineering transfer functions
  - Transfer-function comparison utilities
- Continuous-time state-space modeling
  - Generic `x_dot = A*x + B*u` simulation framework
  - Mass-spring-damper state-space model
  - RLC circuit state-space model
  - DC motor state-space model
- CSV export for selected simulation results
- Saved screenshots for selected plots

## Example Results

### Discrete PID Motor Control

![Discrete PID Motor Control](docs/screenshots/discrete_pid_motor.png)

A discrete PID controller regulates DC motor speed with low steady-state error
and controlled actuator voltage.

### Discrete PID Disturbance Response

The discrete PID motor example can also reject a step load torque disturbance.
The controller increases voltage and current after the disturbance to recover
speed toward the target.

### Load Disturbance Response

![Load Disturbance Response](docs/screenshots/load_disturbance_response.png)

The PI motor controller recovers speed after a load torque disturbance.

### PI Gain Sweep

![PI Gain Sweep](docs/screenshots/pi_gain_sweep.png)

The gain sweep compares how different proportional gains affect motor speed
tracking and control effort.

### RLC Circuit Step Response

![RLC Circuit Step Response](docs/screenshots/rlc_circuit.png)

The RLC example shows underdamped second-order electrical dynamics with
overshoot and settling behavior.

## RC Circuit Charging

An RC circuit has a resistor and capacitor connected to an input voltage. When
the input voltage is applied, the capacitor voltage rises over time toward the
input voltage.

The differential equation is:

```text
dVc/dt = (Vin - Vc) / (R*C)
```

Where:

- `R` is the resistance in ohms.
- `C` is the capacitance in farads.
- `Vin` is the input voltage in volts.
- `Vc` is the capacitor voltage in volts.
- `t` is time in seconds.

The time constant is:

```text
tau = R*C
```

The time constant `tau` describes how quickly the capacitor charges. After one
time constant, a capacitor starting from 0 volts reaches about 63.2% of the
input voltage.

## Newton's Law of Cooling

Newton's Law of Cooling models how an object changes temperature as it moves
toward the temperature of its environment. A hot object cools down, and a cold
object warms up.

The differential equation is:

```text
dT/dt = -k(T - T_env)
```

The analytical solution is:

```text
T(t) = T_env + (T0 - T_env) exp(-kt)
```

Where:

- `T` is the object temperature in degrees Celsius.
- `T0` is the initial object temperature in degrees Celsius.
- `T_env` is the environment temperature in degrees Celsius.
- `k` is the cooling constant.
- `t` is time.

The time constant is:

```text
tau = 1/k
```

The time constant `tau` describes how quickly the object approaches the
environment temperature. After one time constant, the remaining temperature
difference is about `exp(-1)`, or 36.8%, of the initial difference.

## Mass-Spring-Damper Free Vibration

A mass-spring-damper system models the motion of a mass attached to a spring
and damper. It is a common mechanical engineering model for vibration,
oscillation, and energy dissipation.

The governing equation is:

```text
m*x'' + c*x' + k*x = F(t)
```

For numerical simulation, the second-order equation is converted into a
first-order system:

```text
x' = v
v' = (F(t) - c*v - k*x) / m
```

Where:

- `m` is the mass in kilograms.
- `c` is the damping coefficient in newton-seconds per meter.
- `k` is the spring stiffness in newtons per meter.
- `x` is displacement in meters.
- `v` is velocity in meters per second.
- `F(t)` is the external force in newtons as a function of time.

For free vibration, there is no external force, so `F(t) = 0`.

The undamped natural frequency is:

```text
omega_n = sqrt(k/m)
```

The damping ratio is:

```text
zeta = c / (2*sqrt(m*k))
```

These values help describe how quickly the system oscillates and how strongly
the motion decays over time.

## First-Order Control System Step Response

This model shows how a first-order control system responds to a step input.
It is useful for visualizing gain, lag, and settling behavior.

The governing equation is:

```text
tau * dy/dt + y = K*u(t)
```

For `solve_ivp`, it is written as:

```text
dy/dt = (K*u(t) - y) / tau
```

For a constant step input with amplitude `A`, the analytical response is:

```text
y(t) = K*A + (y0 - K*A) * exp(-t/tau)
```

The steady-state value is:

```text
y_ss = K*A
```

Where `tau` is the time constant, `K` is the system gain, `u(t)` is the input,
and `y` is the output. The example also prints practical response metrics:
rise time and settling time.

## Second-Order Control System Step Response

This model simulates the standard second-order control system and compares
theoretical response metrics with measured metrics from the numerical output.

The transfer function is:

```text
omega_n^2 / (s^2 + 2*zeta*omega_n*s + omega_n^2)
```

The time-domain equation is:

```text
y'' + 2*zeta*omega_n*y' + omega_n^2*y = omega_n^2*u(t)
```

The state variables are output `y` and output velocity `y'`. The damping ratio
describes the response type: underdamped, critically damped, or overdamped.
The example prints theoretical overshoot, peak time, and approximate settling
time, then uses the reusable step response metrics on the simulated output.

## Reusable Step Response Metrics

The simulator also includes reusable step response analysis utilities in
`analysis/step_response.py`. These can compute:

- rise time
- settling time
- overshoot
- peak value
- peak time

## RL Circuit Step Response

An RL circuit models the current through a resistor and inductor connected to a
DC step input voltage.

The governing equation is:

```text
L di/dt + R i = Vin
```

For `solve_ivp`, it is written as:

```text
di/dt = (Vin - R*i) / L
```

The time constant and steady-state current are:

```text
tau = L/R
i_ss = Vin/R
```

## Series RLC Circuit Step Response

A series RLC circuit models capacitor voltage and current when a resistor,
inductor, and capacitor are driven by a DC step input.

The numerical model uses:

```text
dVc/dt = i/C
di/dt = (Vin - R*i - Vc) / L
```

The natural frequency and damping ratio are:

```text
omega_n = 1/sqrt(L*C)
zeta = (R/2)*sqrt(C/L)
```

For a DC input, the steady-state behavior is:

```text
Vc -> Vin
i -> 0
```

## Simple Pendulum

The simple pendulum model introduces nonlinear dynamics and compares the full
pendulum equation with the small-angle approximation.

The nonlinear equation is:

```text
theta'' + (g/L)sin(theta) = 0
```

The small-angle approximation is:

```text
theta'' + (g/L)theta = 0
```

The state variables are angle `theta` and angular velocity `omega`. The
small-angle natural frequency and period are:

```text
omega_n = sqrt(g/L)
T = 2*pi*sqrt(L/g)
```

## Inverted Pendulum / Cart-Pole

The inverted pendulum model demonstrates nonlinear open-loop instability for a
pendulum mounted on a moving cart. The state variables are cart position,
cart velocity, pendulum angle, and pendulum angular velocity. The angle
`theta` is measured from the upright equilibrium, so `theta = 0` is upright.

The project also includes a linearized upright state-space model with cart
force as the input and cart position plus pendulum angle as outputs. This
linearized model is used for the LQR stabilization example.

## Inverted Pendulum LQR Control

The LQR example uses the linearized upright state-space model to compute an
optimal state-feedback gain, then applies that gain to the nonlinear
cart-pole model. This demonstrates local stabilization near the upright
equilibrium.

The feedback law is:

```text
F = -K*x
```

Where `x` is `[cart_position, cart_velocity, pendulum_angle,
pendulum_angular_velocity]` and `F` is the horizontal cart force.

## Inverted Pendulum Animation

The project includes reusable Matplotlib animation support for cart-pole
trajectories. The animation uses the same state convention as the nonlinear
model, where `theta = 0` is upright. Open-loop and LQR examples show the
unstable departure from upright and the controlled stabilization behavior.

Animations are displayed interactively by default. GIF and MP4 saving is
optional and only happens when a save path is provided.

## Kalman Filter State Estimation

The Kalman filter examples estimate hidden system states from noisy
measurements using discrete-time linear state-space models. The DC motor
example estimates armature current and speed from noisy speed measurements.
The RLC example estimates capacitor voltage and hidden current from noisy
capacitor-voltage measurements.

## Extended Kalman Filter / Nonlinear Observer

The Extended Kalman Filter example estimates nonlinear pendulum angle and
hidden angular velocity from noisy angle measurements. It uses the full
nonlinear pendulum model for prediction and linearizes the model locally with
Jacobians during each EKF step.

## Unscented Kalman Filter / Nonlinear Observer

The Unscented Kalman Filter example estimates nonlinear pendulum angle and
hidden angular velocity from noisy angle measurements without manually
deriving Jacobians. It uses sigma points to propagate uncertainty through the
nonlinear pendulum model.

## Particle Filter / Nonlinear Observer

The Particle Filter example estimates nonlinear pendulum angle and hidden
angular velocity from noisy angle measurements using many weighted particles
and systematic resampling. It demonstrates nonlinear state estimation without
assuming a Gaussian state distribution.

## Quadcopter Altitude Dynamics

The quadcopter altitude model introduces a simplified one-dimensional UAV
vertical dynamics model. The state is `[z, v]`, where `z` is altitude and `v`
is vertical velocity. Positive altitude, velocity, and thrust are upward.

The governing equation is:

```text
dz/dt = v
dv/dt = (T - m*g - c_drag*v) / m
```

The hover thrust is:

```text
T_hover = m*g
```

This plant model is intentionally limited to vertical altitude motion. It does
not include attitude dynamics, rotor dynamics, or full 6-DOF motion.

## Quadcopter Altitude PID Control

The altitude PID control helper tracks a target altitude by adjusting thrust
around hover thrust. The controller runs as a sampled discrete PID loop, holds
the thrust command during each integration step, and uses output saturation
with anti-windup.

The control law is:

```text
error = z_target - z
T = T_hover + PID(error)
```

The saturated thrust command is then applied to the same 1D altitude plant.
This feature focuses only on vertical altitude control and does not include
attitude coupling or full drone trajectory control.

## Quadcopter Attitude Dynamics

The quadcopter attitude model introduces simplified rotational dynamics for
roll, pitch, and yaw motion. The state is `[phi, theta, psi, p, q, r]`, where
`phi`, `theta`, and `psi` are roll, pitch, and yaw angles in radians, and
`p`, `q`, and `r` are body angular rates in radians per second.

The first version uses small-angle kinematics and decoupled rigid-body
rotational dynamics:

```text
phi_dot = p
theta_dot = q
psi_dot = r
p_dot = tau_phi / Ixx
q_dot = tau_theta / Iyy
r_dot = tau_psi / Izz
```

The inputs are body torques `[tau_phi, tau_theta, tau_psi]` in newton-meters.
This model is intentionally limited to rotational attitude motion. It does not
include translational 6-DOF dynamics, rotor dynamics, or position control.

## Quadcopter Attitude PID Control

The attitude PID control helper tracks target roll, pitch, and yaw angles by
commanding independent body torques on each axis. The controller runs as a
sampled discrete PID loop, holds torque commands during each integration step,
and uses torque saturation with anti-windup.

The control law per axis is:

```text
error = target_angle - measured_angle
tau_axis = PID(error)
```

The examples show target attitude step tracking and rejection of a simple
external roll disturbance torque. This feature stays within the simplified
rotational model and does not add altitude-position coupling, trajectory
control, or full 6-DOF dynamics.

## Full 6-DOF Quadcopter Dynamics

The full 6-DOF quadcopter model combines inertial-frame translation with
rigid-body roll, pitch, and yaw rotation. The state is:

```text
[x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
```

The inputs are total thrust and body torques:

```text
[T, tau_phi, tau_theta, tau_psi]
```

Inertial `z` is positive upward, gravity is `[0, 0, -g]`, and hover at zero
attitude requires `T = m*g`. The body-to-inertial rotation uses a standard ZYX
Euler convention:

```text
R = Rz(psi) @ Ry(theta) @ Rx(phi)
```

With zero yaw, positive pitch redirects thrust toward positive inertial `x`,
and positive roll redirects thrust toward negative inertial `y`. The plant
model itself is open-loop and does not include rotor mixing or an autopilot
controller.

## Quadcopter Trajectory Tracking

The trajectory tracking helper builds an educational cascaded PD controller on
top of the full 6-DOF plant. Reference generators provide hover-point and
circular trajectories with desired position, velocity, and acceleration. The
outer position loop computes a desired acceleration, converts horizontal
acceleration into small roll and pitch commands, and the inner attitude loop
commands body torques.

This controller is intentionally simple. It demonstrates how trajectory
tracking can be layered onto a 6-DOF rigid-body model, but it is not a
production drone autopilot and does not include MPC, waypoint planning, rotor
mixing, or individual motor dynamics.

## Quadcopter Waypoint Following

The waypoint-following helper converts a list of 3D waypoints into a time-based
reference trajectory, then tracks that reference with the existing simplified
6-DOF cascaded controller. The reference can use piecewise-linear interpolation
or smoothstep interpolation between waypoints.

This feature demonstrates waypoint following, not full autonomous path
planning. It does not include obstacle avoidance, MPC, rotor mixing, or real
drone autopilot complexity.

## Quadcopter Animation

The project includes reusable 3D Matplotlib animation support for full 6-DOF
quadcopter trajectories. The animation uses the same state convention and ZYX
body-to-inertial rotation matrix as `models/quadcopter_6dof.py`, then draws a
simple drone body with crossing arms, rotor markers, path trail, optional
reference path, and optional waypoint markers.

Animations are displayed interactively by default. GIF and MP4 saving is
optional and only happens when a save path is provided. Saved portfolio
animations can be written under `docs/animations/`.

## DC Motor Speed Response

The DC motor model demonstrates coupled electrical-mechanical dynamics for a
permanent-magnet motor.

The electrical equation is:

```text
V = L di/dt + R i + Ke omega
```

The mechanical equation is:

```text
J domega/dt + b omega = Kt i - TL
```

The state variables are armature current `i` and angular speed `omega`. The
example reports motor speed in both radians per second and rpm.

## DC Motor PI Speed Control

This closed-loop model tracks a target motor speed using a PI controller with a
PID-compatible API. Derivative action is reserved for a future extension.

The controller is:

```text
V = Kp*error + Ki*integral_error
```

The controller voltage is saturated between configured minimum and maximum
limits. The motor plant uses the same coupled electrical-mechanical equations
as the open-loop DC motor model, and the example shows speed tracking to a
target reference.

## Install Dependencies

Create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Interactive Streamlit App

The project includes a simple browser UI for selected simulations:

- RC circuit charging
- RLC circuit step response
- DC motor discrete PID speed control

Install dependencies and run the app with:

```powershell
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Frequency Response and Bode Plots

The project includes reusable frequency response helpers for continuous-time
transfer functions and Bode plot examples for common engineering systems.

Run the examples with:

```powershell
python examples/run_frequency_response_first_order.py
python examples/run_frequency_response_second_order.py
python examples/run_frequency_response_rlc.py
```

## Transfer Function Utilities

The project includes reusable continuous-time transfer function utilities for
first-order, second-order, and RLC low-pass systems. These helpers support step
response simulation, impulse response simulation, and simple response plots.
Transfer functions are one of the fundamental tools of control engineering.
They describe the input-output behavior of linear time-invariant systems and
provide the basis for step-response analysis, impulse-response analysis,
frequency response, Bode plots, and classical controller design.

Run the examples with:

```powershell
python examples/run_transfer_function_step_response.py
python examples/run_transfer_function_impulse_response.py
python examples/run_transfer_function_comparison.py
```

## State-Space Models

The project includes reusable continuous-time state-space simulation utilities
for linear engineering systems.
State-space models provide a modern representation of dynamic systems using
matrices `A`, `B`, `C`, and `D`. They are widely used in control engineering
because they naturally support multi-state systems and form the basis for
advanced techniques such as controllability analysis, observers, Kalman
filters, and LQR control.

Run the examples with:

```powershell
python examples/run_state_space_mass_spring_damper.py
python examples/run_state_space_rlc.py
python examples/run_state_space_dc_motor.py
```

## Run the RC Circuit Example

The example simulates an RC circuit with:

- `R = 1000` ohms
- `C = 0.001` farads
- `Vin = 5` volts
- `V0 = 0` volts
- time from `0` to `5` seconds

Run it with:

```powershell
python examples\run_rc_circuit.py
```

This plots the numerical solution from `scipy.integrate.solve_ivp` and the
analytical solution on the same graph.

## Run the Cooling Example

The example simulates an object cooling with:

- `T0 = 90` degrees Celsius
- `T_env = 22` degrees Celsius
- `k = 0.08` per minute
- time from `0` to `60` minutes

Run it with:

```powershell
python examples\run_cooling.py
```

This plots the numerical and analytical temperature curves on the same graph.

## Run the Mass-Spring-Damper Example

The example simulates free vibration with:

- `m = 1` kilogram
- `c = 0.4` newton-seconds per meter
- `k = 4` newtons per meter
- `x0 = 1` meter
- `v0 = 0` meters per second
- time from `0` to `20` seconds

Run it with:

```powershell
python examples\run_mass_spring_damper.py
```

This prints the natural frequency and damping ratio, then plots displacement
over time.

## Run the First-Order Control Example

The example simulates a step response with:

- `tau = 1.5` seconds
- `K = 2.0`
- input amplitude `A = 1.0`
- `y0 = 0.0`
- time from `0` to `10` seconds

Run it with:

```powershell
python examples\run_first_order_control.py
```

This prints gain, time constant, steady-state value, rise time, and settling
time, then plots the numerical and analytical step responses.

## Run the Second-Order Control Example

The example simulates an underdamped second-order step response and uses the
reusable step response metrics.

Run it with:

```powershell
python examples\run_second_order_control.py
```

## Run the RL Circuit Example

The example simulates an RL circuit with:

- `R = 10` ohms
- `L = 2` henries
- `Vin = 5` volts
- `i0 = 0` amps
- time from `0` to `1.5` seconds

Run it with:

```powershell
python examples\run_rl_circuit.py
```

This prints the time constant and steady-state current, then plots the
numerical and analytical current responses.

## Run the RLC Circuit Example

The example simulates an underdamped series RLC circuit and uses the reusable
step response metrics to estimate rise time, settling time, peak voltage, peak
time, and overshoot.

Run it with:

```powershell
python examples\run_rlc_circuit.py
```

## Run the RLC Parameter Sweep Examples

These examples compare how component values shape the series RLC capacitor
voltage step response. They reuse the existing RLC model and step response
metrics to compare damping ratio, natural frequency, overshoot, peak voltage,
settling time, and final voltage.

Run them with:

```powershell
python examples/run_rlc_resistance_sweep.py
python examples/run_rlc_capacitance_sweep.py
python examples/run_rlc_inductance_sweep.py
```

## Run the Pendulum Example

The example compares nonlinear pendulum motion with the linear small-angle
approximation.

Run it with:

```powershell
python examples\run_pendulum.py
```

## Run the Inverted Pendulum Examples

The open-loop example shows that a small initial angle perturbation grows
without feedback control. The linearized example prints the upright
state-space matrices and eigenvalues, then simulates a small perturbation.

Run them with:

```powershell
python examples/run_inverted_pendulum_open_loop.py
python examples/run_inverted_pendulum_linearized.py
```

The LQR examples show local stabilization of the nonlinear inverted pendulum
and compare open-loop instability against LQR feedback.

Run them with:

```powershell
python examples/run_inverted_pendulum_lqr.py
python examples/run_inverted_pendulum_lqr_comparison.py
```

Animation examples visualize the same cart-pole trajectories with Matplotlib.

Run them with:

```powershell
python examples/run_inverted_pendulum_animation_open_loop.py
python examples/run_inverted_pendulum_animation_lqr.py
```

Optionally save an animation as GIF or MP4:

```powershell
python examples/run_inverted_pendulum_animation_lqr.py --save docs/animations/inverted_pendulum_lqr.gif
```

## Run the Kalman Filter Examples

The Kalman examples demonstrate noisy-measurement state estimation for a DC
motor and a series RLC circuit.

Run them with:

```powershell
python examples/run_kalman_dc_motor.py
python examples/run_kalman_rlc.py
```

The nonlinear pendulum EKF example demonstrates nonlinear observer design by
estimating angle and hidden angular velocity from noisy angle measurements.

Run it with:

```powershell
python examples/run_ekf_pendulum.py
```

The nonlinear pendulum UKF example demonstrates nonlinear state estimation
with sigma points instead of manually derived Jacobians.

Run it with:

```powershell
python examples/run_ukf_pendulum.py
```

The nonlinear pendulum particle-filter example demonstrates probabilistic
state estimation with weighted particles and resampling.

Run it with:

```powershell
python examples/run_particle_filter_pendulum.py
```

## Run the Quadcopter Altitude Examples

The quadcopter altitude examples compare open-loop thrust commands and show a
simple thrust step from hover to climb. The altitude PID examples show
closed-loop target tracking and rejection of a downward force disturbance.

Run them with:

```powershell
python examples/run_quadcopter_altitude_open_loop.py
python examples/run_quadcopter_altitude_thrust_step.py
python examples/run_quadcopter_altitude_pid.py
python examples/run_quadcopter_altitude_pid_disturbance.py
```

## Run the Quadcopter Attitude Examples

The quadcopter attitude examples show open-loop roll response to a constant
body torque, a roll/pitch/yaw response to a torque step, PID attitude tracking,
and rejection of a simple external disturbance torque.

Run them with:

```powershell
python examples/run_quadcopter_attitude_roll_torque.py
python examples/run_quadcopter_attitude_torque_step.py
python examples/run_quadcopter_attitude_pid.py
python examples/run_quadcopter_attitude_pid_disturbance.py
```

## Run the Full 6-DOF Quadcopter Examples

The full 6-DOF examples show open-loop hover, tilted thrust causing horizontal
acceleration, and body torque inputs changing attitude and motion.

Run them with:

```powershell
python examples/run_quadcopter_6dof_hover.py
python examples/run_quadcopter_6dof_tilted_thrust.py
python examples/run_quadcopter_6dof_torque_response.py
```

## Run the Quadcopter Trajectory Tracking Examples

The trajectory tracking examples use the full 6-DOF plant with a simplified
cascaded controller to track a hover point and a slow circular path.

Run them with:

```powershell
python examples/run_quadcopter_trajectory_hover_tracking.py
python examples/run_quadcopter_trajectory_circle_tracking.py
```

## Run the Quadcopter Waypoint Following Example

The waypoint-following example uses the full 6-DOF plant with the simplified
cascaded controller to track a smooth path through several 3D waypoints.

Run it with:

```powershell
python examples/run_quadcopter_waypoint_following.py
```

## Run the Quadcopter Animation Examples

The quadcopter animation examples reuse the existing waypoint-following and
circular trajectory simulations, then visualize the 6-DOF position and
attitude in 3D.

Run them with:

```powershell
python examples/run_quadcopter_waypoint_animation.py
python examples/run_quadcopter_circle_animation.py
```

Optionally save an animation as GIF or MP4:

```powershell
python examples/run_quadcopter_waypoint_animation.py --save docs/animations/quadcopter_waypoint_following.gif --no-show
```

## Run the DC Motor Example

The example simulates motor current and speed after a voltage step.

Run it with:

```powershell
python examples\run_dc_motor.py
```

## Run the DC Motor Open-Loop Disturbance Example

The example shows that a fixed-voltage DC motor loses speed and settles at a
lower operating point after a step load torque disturbance.

Run it with:

```powershell
python examples/run_dc_motor_open_loop_disturbance.py
```

## Run the DC Motor PI Control Example

The example simulates closed-loop speed tracking with voltage saturation.

Run it with:

```powershell
python examples\run_pid_motor_control.py
```

## Run the PI Gain Sweep Example

The example compares PI motor speed-control responses for several `Kp` values.

Run it with:

```powershell
python examples\run_pi_gain_sweep.py
```

## Run the PID Tuning Examples

The PID tuning examples use the discrete PID motor simulation to show how
proportional, integral, and derivative gains affect speed tracking,
steady-state error, overshoot, settling time, and control voltage.

Run them with:

```powershell
python examples/run_pid_p_pi_pid_comparison.py
python examples/run_pid_kp_tuning.py
python examples/run_pid_ki_tuning.py
python examples/run_pid_kd_tuning.py
```

## Run the Motor Load Disturbance Example

The example shows PI speed control recovering after a load torque step.

Run it with:

```powershell
python examples\run_motor_load_disturbance.py
```

## Run the Discrete PID Motor Example

The example simulates embedded-style digital PID speed control for a DC motor
and exports data to `outputs/discrete_pid_motor.csv`. CSV files can be opened
in Excel, MATLAB, LibreOffice Calc, or Python.

Run it with:

```powershell
python examples\run_discrete_pid_motor.py
```

## Run the Discrete PID Disturbance Response Example

The example shows a sampled-data PID controller recovering motor speed after a
step load torque disturbance.

Run it with:

```powershell
python examples/run_discrete_pid_disturbance_response.py
```

## Run the DC Motor Disturbance Rejection Comparison

The comparison example applies the same load torque disturbance to open-loop,
continuous PI, and discrete PID motor responses. It shows how feedback control
increases control effort to recover speed toward the target.

Run it with:

```powershell
python examples/run_dc_motor_disturbance_rejection_comparison.py
```

## Run Tests

Run the test suite with:

```powershell
pytest
```

Selected examples save plot screenshots to `docs/screenshots/` for portfolio
and documentation use.

The current tests cover all implemented models, including initial conditions,
long-term behavior, one-time-constant behavior, and mechanical energy behavior.
