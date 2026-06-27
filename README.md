# Engineering ODE Simulator

Engineering ODE Simulator is a portfolio project for modeling simple
engineering systems with ordinary differential equations (ODEs). The goal is
to pair numerical simulations with analytical solutions where possible, so the
results are easy to understand and verify.

The project currently includes:

- RC circuit charging
- Newton's Law of Cooling
- Mass-spring-damper free vibration

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

## Run Tests

Run the test suite with:

```powershell
pytest
```

The current tests cover all implemented models, including initial conditions,
long-term behavior, one-time-constant behavior, and mechanical energy behavior.
