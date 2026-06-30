"""Discrete PID controller and DC motor speed-control simulation."""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from models.dc_motor import validate_dc_motor_parameters, zero_load_torque


DEFAULT_DISCRETE_PID_MOTOR_PARAMS = {
    "R": 1.0,
    "L": 0.5,
    "J": 0.01,
    "b": 0.001,
    "Kt": 0.01,
    "Ke": 0.01,
}


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


def load_torque_step(
    t: float,
    disturbance_time: float = 12.0,
    initial_torque: float = 0.0,
    final_torque: float = 0.03,
) -> float:
    """Return a step load torque disturbance."""
    if disturbance_time < 0:
        raise ValueError("disturbance_time must be nonnegative")

    if t < disturbance_time:
        return initial_torque

    return final_torque


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


def _validate_t_span(t_span):
    """Validate and return a simulation start and end time."""
    if len(t_span) != 2:
        raise ValueError("t_span must contain start and end times")

    start_time = float(t_span[0])
    end_time = float(t_span[1])

    if end_time <= start_time:
        raise ValueError("t_span final time must be greater than initial time")

    return start_time, end_time


def simulate_discrete_pid_motor_with_disturbance(
    target_speed=80.0,
    Kp=0.5,
    Ki=0.2,
    Kd=0.02,
    dt=0.01,
    t_span=(0.0, 25.0),
    disturbance_time=12.0,
    load_torque_initial=0.0,
    load_torque_final=0.03,
    voltage_limits=(0.0, 24.0),
    motor_params=None,
    i0=0.0,
    omega0=0.0,
    anti_windup=True,
):
    """Simulate discrete PID motor speed control with a load torque step.

    The controller updates once per sample and the voltage command is held
    constant while the DC motor plant is integrated over that sample interval.
    """
    start_time, end_time = _validate_t_span(t_span)

    if dt <= 0:
        raise ValueError("dt must be positive")

    if disturbance_time < start_time or disturbance_time > end_time:
        raise ValueError("disturbance_time must be within t_span")

    if len(voltage_limits) != 2:
        raise ValueError("voltage_limits must contain minimum and maximum voltage")

    voltage_min = float(voltage_limits[0])
    voltage_max = float(voltage_limits[1])

    if voltage_max <= voltage_min:
        raise ValueError("voltage_max must be greater than voltage_min")

    if motor_params is None:
        motor_params = DEFAULT_DISCRETE_PID_MOTOR_PARAMS

    pid = DiscretePID(
        Kp,
        Ki,
        Kd,
        output_min=voltage_min,
        output_max=voltage_max,
        anti_windup=anti_windup,
    )

    load_torque_func = lambda t: load_torque_step(
        t,
        disturbance_time=disturbance_time - start_time,
        initial_torque=load_torque_initial,
        final_torque=load_torque_final,
    )

    time, current, speed, voltage, error = simulate_discrete_pid_motor_control(
        motor_params["R"],
        motor_params["L"],
        motor_params["J"],
        motor_params["b"],
        motor_params["Kt"],
        motor_params["Ke"],
        pid,
        target_speed,
        i0,
        omega0,
        end_time - start_time,
        dt,
        load_torque_func=load_torque_func,
    )
    time = time + start_time
    load_torque = np.array(
        [
            load_torque_step(
                sample_time,
                disturbance_time=disturbance_time,
                initial_torque=load_torque_initial,
                final_torque=load_torque_final,
            )
            for sample_time in time
        ]
    )

    return {
        "time": time,
        "speed": speed,
        "current": current,
        "voltage": voltage,
        "error": error,
        "load_torque": load_torque,
        "target_speed": target_speed,
    }


def summarize_disturbance_response(
    result,
    disturbance_time,
    target_speed,
    recovery_tolerance=0.02,
):
    """Return practical disturbance rejection metrics for a motor response."""
    if recovery_tolerance <= 0:
        raise ValueError("recovery_tolerance must be positive")

    time = np.asarray(result["time"], dtype=float)
    speed = np.asarray(result["speed"], dtype=float)
    voltage = np.asarray(result["voltage"], dtype=float)
    current = np.asarray(result["current"], dtype=float)

    _validate_response_arrays(time, speed, voltage, current)

    disturbance_index = int(np.argmin(np.abs(time - disturbance_time)))
    before_index = max(0, disturbance_index - 1)
    recovery_index = _recovery_index(
        time,
        speed,
        disturbance_index,
        target_speed,
        recovery_tolerance,
    )
    after_recovery_or_final_index = (
        recovery_index if recovery_index is not None else len(time) - 1
    )

    speed_before = float(speed[before_index])
    minimum_speed_after = float(np.min(speed[disturbance_index:]))
    recovery_time = (
        None
        if recovery_index is None
        else float(time[recovery_index] - time[disturbance_index])
    )

    return {
        "speed_before_disturbance": speed_before,
        "minimum_speed_after_disturbance": minimum_speed_after,
        "speed_drop": float(speed_before - minimum_speed_after),
        "final_speed": float(speed[-1]),
        "final_error": float(target_speed - speed[-1]),
        "voltage_before_disturbance": float(voltage[before_index]),
        "voltage_after_recovery_or_final": float(voltage[after_recovery_or_final_index]),
        "current_before_disturbance": float(current[before_index]),
        "current_after_recovery_or_final": float(current[after_recovery_or_final_index]),
        "recovery_time": recovery_time,
    }


def _validate_response_arrays(time, speed, voltage, current):
    """Validate aligned response arrays used for disturbance metrics."""
    lengths = {len(time), len(speed), len(voltage), len(current)}

    if len(time) < 2:
        raise ValueError("response arrays must contain at least 2 samples")

    if len(lengths) != 1:
        raise ValueError("response arrays must have equal lengths")


def _recovery_index(time, speed, disturbance_index, target_speed, tolerance):
    """Return the first post-disturbance index that stays within tolerance."""
    if target_speed == 0:
        band = tolerance
    else:
        band = tolerance * abs(target_speed)

    lower = target_speed - band
    upper = target_speed + band
    inside_band = (speed >= lower) & (speed <= upper)

    for index in range(disturbance_index, len(time)):
        if np.all(inside_band[index:]):
            return index

    return None
