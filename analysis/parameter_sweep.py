"""Reusable parameter sweep tools for simulation studies."""

from dataclasses import dataclass

import numpy as np

from analysis.step_response import calculate_step_info
from models.pid_motor_control import simulate_pi_motor_control, speed_reference_step


@dataclass
class SweepResult:
    """Summary metrics for one parameter sweep run."""

    parameter_name: str
    parameter_value: float
    final_value: float
    final_error: float
    peak_value: float
    overshoot_percent: float
    settling_time: float | None
    max_control_effort: float
    max_current: float


def calculate_tracking_error(reference_value: float, final_value: float) -> float:
    """Return reference_value - final_value."""
    return reference_value - final_value


def _validate_parameter_sweep_inputs(
    parameter_name,
    parameter_values,
    motor_params,
    controller_params,
    simulation_params,
):
    """Validate PI motor gain sweep inputs."""
    if parameter_name not in ("Kp", "Ki"):
        raise ValueError('parameter_name must be "Kp" or "Ki"')

    if len(parameter_values) == 0:
        raise ValueError("parameter_values must not be empty")

    for parameter_value in parameter_values:
        if parameter_value < 0:
            raise ValueError("parameter values must be nonnegative")

    required_motor_keys = ("R", "L", "J", "b", "Kt", "Ke")
    required_controller_keys = (
        "Kp",
        "Ki",
        "Kd",
        "voltage_min",
        "voltage_max",
        "target_speed",
    )
    required_simulation_keys = (
        "i0",
        "omega0",
        "integral_error0",
        "t_span",
        "num_points",
    )

    _validate_required_keys(motor_params, required_motor_keys, "motor_params")
    _validate_required_keys(
        controller_params,
        required_controller_keys,
        "controller_params",
    )
    _validate_required_keys(
        simulation_params,
        required_simulation_keys,
        "simulation_params",
    )


def _validate_required_keys(parameters, required_keys, dictionary_name):
    """Validate that a parameter dictionary has all required keys."""
    missing_keys = [key for key in required_keys if key not in parameters]

    if missing_keys:
        missing_text = ", ".join(missing_keys)
        raise ValueError(f"{dictionary_name} is missing required keys: {missing_text}")


def run_pi_motor_gain_sweep(
    parameter_name: str,
    parameter_values: list[float],
    motor_params: dict,
    controller_params: dict,
    simulation_params: dict,
):
    """Run a PI motor speed-control sweep over Kp or Ki.

    Parameters
    ----------
    parameter_name : str
        Controller gain to sweep. Must be ``"Kp"`` or ``"Ki"``.
    parameter_values : list[float]
        Gain values to simulate.
    motor_params : dict
        DC motor parameters.
    controller_params : dict
        PI controller parameters, voltage limits, and target speed.
    simulation_params : dict
        Initial conditions and time sampling settings.

    Returns
    -------
    tuple
        ``(results, simulations)``. ``results`` is a list of ``SweepResult``
        objects. ``simulations`` maps each parameter value to returned arrays.
    """
    _validate_parameter_sweep_inputs(
        parameter_name,
        parameter_values,
        motor_params,
        controller_params,
        simulation_params,
    )

    results = []
    simulations = {}
    target_speed = controller_params["target_speed"]

    for parameter_value in parameter_values:
        swept_controller_params = controller_params.copy()
        swept_controller_params[parameter_name] = parameter_value

        reference_func = lambda t, target=target_speed: speed_reference_step(t, target)

        t, current, speed, integral_error, voltage = simulate_pi_motor_control(
            motor_params["R"],
            motor_params["L"],
            motor_params["J"],
            motor_params["b"],
            motor_params["Kt"],
            motor_params["Ke"],
            swept_controller_params["Kp"],
            swept_controller_params["Ki"],
            simulation_params["i0"],
            simulation_params["omega0"],
            simulation_params["integral_error0"],
            simulation_params["t_span"],
            simulation_params["num_points"],
            reference_func=reference_func,
            voltage_min=swept_controller_params["voltage_min"],
            voltage_max=swept_controller_params["voltage_max"],
            Kd=swept_controller_params["Kd"],
        )

        step_info = calculate_step_info(t, speed)
        final_value = float(speed[-1])
        final_error = calculate_tracking_error(target_speed, final_value)

        results.append(
            SweepResult(
                parameter_name=parameter_name,
                parameter_value=parameter_value,
                final_value=final_value,
                final_error=final_error,
                peak_value=step_info.peak_value,
                overshoot_percent=step_info.overshoot_percent,
                settling_time=step_info.settling_time,
                max_control_effort=float(np.max(np.abs(voltage))),
                max_current=float(np.max(np.abs(current))),
            )
        )

        simulations[parameter_value] = {
            "t": t,
            "current": current,
            "speed": speed,
            "integral_error": integral_error,
            "voltage": voltage,
        }

    return results, simulations
