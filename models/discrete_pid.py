"""Discrete PID controller and DC motor speed-control simulation."""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from models.dc_motor import validate_dc_motor_parameters, zero_load_torque


@dataclass
class DiscretePID:
    """Embedded-style discrete PID controller with output saturation."""

    Kp: float
    Ki: float
    Kd: float
    output_min: float | None = None
    output_max: float | None = None
    anti_windup: bool = True

    def __post_init__(self):
        """Validate gains and initialize controller state."""
        self._validate_parameters()
        self.reset()

    def _validate_parameters(self):
        """Validate PID gains and output limits."""
        if self.Kp < 0:
            raise ValueError("Kp must be nonnegative")

        if self.Ki < 0:
            raise ValueError("Ki must be nonnegative")

        if self.Kd < 0:
            raise ValueError("Kd must be nonnegative")

        if (
            self.output_min is not None
            and self.output_max is not None
            and self.output_max <= self.output_min
        ):
            raise ValueError("output_max must be greater than output_min")

    def reset(self):
        """Reset accumulated controller state."""
        self.integral = 0.0
        self.previous_error = None
        self.previous_measurement = None
        self.last_output = None

    def _clip_output(self, raw_output):
        """Clip controller output to configured limits."""
        output = raw_output

        if self.output_min is not None:
            output = max(self.output_min, output)

        if self.output_max is not None:
            output = min(self.output_max, output)

        return output

    def update(self, setpoint: float, measurement: float, dt: float) -> float:
        """Return a PID output for one discrete control update."""
        if dt <= 0:
            raise ValueError("dt must be positive")

        error = setpoint - measurement
        candidate_integral = self.integral + error * dt

        if self.previous_measurement is None:
            derivative = 0.0
        else:
            derivative = -(measurement - self.previous_measurement) / dt

        raw_output = (
            self.Kp * error
            + self.Ki * candidate_integral
            + self.Kd * derivative
        )
        output = self._clip_output(raw_output)

        if self._should_accept_integral(output, raw_output, error):
            self.integral = candidate_integral

        self.previous_error = error
        self.previous_measurement = measurement
        self.last_output = output

        return output

    def _should_accept_integral(self, output, raw_output, error):
        """Return True when the candidate integral should be stored."""
        if not self.anti_windup:
            return True

        saturated_high = output != raw_output and self.output_max is not None
        saturated_high = saturated_high and np.isclose(output, self.output_max)
        saturated_low = output != raw_output and self.output_min is not None
        saturated_low = saturated_low and np.isclose(output, self.output_min)

        if not saturated_high and not saturated_low:
            return True

        if saturated_high and error < 0:
            return True

        if saturated_low and error > 0:
            return True

        return False


def _dc_motor_constant_voltage_ode(t, state, R, L, J, b, Kt, Ke, voltage, load_torque_func):
    """Return DC motor derivatives for a held voltage command."""
    current = state[0]
    omega = state[1]
    load_torque = load_torque_func(t)

    di_dt = (voltage - R * current - Ke * omega) / L
    domega_dt = (Kt * current - b * omega - load_torque) / J

    return [di_dt, domega_dt]


def simulate_discrete_pid_motor_control(
    R,
    L,
    J,
    b,
    Kt,
    Ke,
    pid: DiscretePID,
    target_speed: float,
    i0: float,
    omega0: float,
    t_final: float,
    dt: float,
    load_torque_func=None,
):
    """Simulate DC motor speed control using a discrete PID controller.

    At each sample, the controller computes a voltage command that is held
    constant while the continuous DC motor plant is integrated for one sample
    interval.
    """
    validate_dc_motor_parameters(R, L, J, b, Kt, Ke)

    if t_final <= 0:
        raise ValueError("t_final must be positive")

    if dt <= 0:
        raise ValueError("dt must be positive")

    if load_torque_func is None:
        load_torque_func = zero_load_torque

    steps = int(t_final / dt)
    t = np.linspace(0.0, steps * dt, steps + 1)
    current = np.zeros(steps + 1)
    speed = np.zeros(steps + 1)
    voltage = np.zeros(steps + 1)
    error = np.zeros(steps + 1)

    current[0] = i0
    speed[0] = omega0

    for index in range(steps):
        error[index] = target_speed - speed[index]
        voltage[index] = pid.update(target_speed, speed[index], dt)

        solution = solve_ivp(
            _dc_motor_constant_voltage_ode,
            (t[index], t[index + 1]),
            [current[index], speed[index]],
            args=(R, L, J, b, Kt, Ke, voltage[index], load_torque_func),
            t_eval=[t[index + 1]],
        )

        current[index + 1] = solution.y[0, -1]
        speed[index + 1] = solution.y[1, -1]

    error[-1] = target_speed - speed[-1]

    if steps > 0:
        voltage[-1] = voltage[-2]

    return t, current, speed, voltage, error
