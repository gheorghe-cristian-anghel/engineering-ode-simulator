# Equations Reference

## RC Circuit

Governing equation:

```text
dVc/dt = (Vin - Vc)/(R C)
```

State variable:

- `Vc`: capacitor voltage

Important formulas:

```text
tau = R C
Vc(t) = Vin + (V0 - Vin) exp(-t/(R C))
```

## RL Circuit

Governing equation:

```text
L di/dt + R i = Vin
```

First-order form:

```text
di/dt = (Vin - R i)/L
```

State variable:

- `i`: current

Important formulas:

```text
tau = L/R
i_ss = Vin/R
```

## RLC Circuit

First-order form:

```text
dVc/dt = i/C
di/dt = (Vin - R i - Vc)/L
```

State variables:

- `Vc`: capacitor voltage
- `i`: current

Important formulas:

```text
omega_n = 1/sqrt(L C)
zeta = (R/2) sqrt(C/L)
```

DC steady state:

```text
Vc -> Vin
i -> 0
```

## Newton Cooling

Governing equation:

```text
dT/dt = -k(T - T_env)
```

State variable:

- `T`: object temperature

Important formulas:

```text
T(t) = T_env + (T0 - T_env) exp(-k t)
tau = 1/k
```

## Mass-Spring-Damper

Governing equation:

```text
m x'' + c x' + k x = F(t)
```

First-order form:

```text
x' = v
v' = (F(t) - c v - k x)/m
```

State variables:

- `x`: displacement
- `v`: velocity

Important formulas:

```text
omega_n = sqrt(k/m)
zeta = c/(2 sqrt(m k))
E = 0.5 m v^2 + 0.5 k x^2
```

## Pendulum

Nonlinear equation:

```text
theta'' + (g/L) sin(theta) = 0
```

First-order nonlinear form:

```text
theta' = omega
omega' = -(g/L) sin(theta)
```

Small-angle approximation:

```text
theta'' + (g/L) theta = 0
```

State variables:

- `theta`: angle
- `omega`: angular velocity

Important formulas:

```text
omega_n = sqrt(g/L)
T = 2 pi sqrt(L/g)
```

## First-Order Control System

Governing equation:

```text
tau dy/dt + y = K u(t)
```

First-order form:

```text
dy/dt = (K u(t) - y)/tau
```

Step response for input amplitude `A`:

```text
y(t) = K A + (y0 - K A) exp(-t/tau)
y_ss = K A
```

## Second-Order Control System

Transfer function:

```text
G(s) = omega_n^2 / (s^2 + 2 zeta omega_n s + omega_n^2)
```

Time-domain equation:

```text
y'' + 2 zeta omega_n y' + omega_n^2 y = omega_n^2 u(t)
```

First-order form:

```text
y' = v
v' = omega_n^2 (u(t) - y) - 2 zeta omega_n v
```

State variables:

- `y`: output
- `v`: output velocity

Theoretical overshoot for `0 < zeta < 1`:

```text
Mp = exp(-zeta pi / sqrt(1 - zeta^2)) * 100%
```

## DC Motor

Electrical equation:

```text
V = L di/dt + R i + Ke omega
di/dt = (V - R i - Ke omega)/L
```

Mechanical equation:

```text
J domega/dt + b omega = Kt i - TL
domega/dt = (Kt i - b omega - TL)/J
```

`TL` is the external load torque. A positive `TL` opposes motor torque and can
be used to model a step load disturbance.

State variables:

- `i`: armature current
- `omega`: angular speed

Steady-state formulas:

```text
omega_ss = (V - R TL/Kt) / (Ke + R b/Kt)
i_ss = (b omega_ss + TL)/Kt
```

## PI Motor Speed Control

Controller:

```text
error = omega_ref - omega
V = Kp error + Ki integral_error
d(integral_error)/dt = error
```

The controller voltage is saturated between `voltage_min` and `voltage_max`.
The current implementation is PI-only; derivative action is reserved for a
future PID extension.

State variables:

- `i`: armature current
- `omega`: angular speed
- `integral_error`: accumulated speed error

## Discrete PID Control

The discrete PID controller updates once per sample time `dt` and holds its
output constant between updates.

```text
error_k = setpoint_k - measurement_k
I_k = I_{k-1} + error_k * dt
D_k = -(measurement_k - measurement_{k-1}) / dt
u_k = Kp * error_k + Ki * I_k + Kd * D_k
```

The derivative term uses derivative-on-measurement, which avoids a derivative
kick when the setpoint changes suddenly. The controller output is clipped to
voltage limits, and anti-windup prevents integral growth during saturation.
