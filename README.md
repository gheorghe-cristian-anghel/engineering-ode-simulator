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
- Reusable step response metrics
- RL circuit step response
- Series RLC circuit step response
- Simple pendulum nonlinear dynamics

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

## Install Dependencies

Create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
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

## Run the Pendulum Example

The example compares nonlinear pendulum motion with the linear small-angle
approximation.

Run it with:

```powershell
python examples\run_pendulum.py
```

## Run Tests

Run the test suite with:

```powershell
pytest
```

The current tests cover all implemented models, including initial conditions,
long-term behavior, one-time-constant behavior, and mechanical energy behavior.
