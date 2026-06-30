"""Compare DC motor open-loop and feedback load disturbance responses."""

import sys
from pathlib import Path
from _tkinter import TclError

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.motor_disturbance import (
    load_torque_step,
    summarize_motor_disturbance_response,
)
from models.dc_motor import simulate_dc_motor, steady_state_current
from models.discrete_pid import simulate_discrete_pid_motor_with_disturbance
from models.pid_motor_control import simulate_pi_motor_control, speed_reference_step


def _format_optional(value):
    """Format optional numeric table values."""
    if value is None:
        return "None"

    return f"{value:.3f}"


def _print_comparison_table(results):
    """Print disturbance rejection comparison metrics."""
    print("DC Motor Load Disturbance Rejection Comparison:")
    print(
        "Open-loop control cannot reject the load disturbance, while feedback "
        "controllers increase control effort to recover speed."
    )
    print()
    print(
        "Mode          Speed Before   Speed Drop   Final Speed   Final Error   "
        "Recovery"
    )
    print(
        "              (rad/s)        (rad/s)      (rad/s)       (rad/s)       "
        "(s)"
    )

    for label, metrics in results:
        print(
            f"{label:<14}"
            f"{metrics['speed_before_disturbance']:<15.3f}"
            f"{metrics['speed_drop']:<13.3f}"
            f"{metrics['final_speed']:<14.3f}"
            f"{metrics['final_error']:<14.3f}"
            f"{_format_optional(metrics['recovery_time'])}"
        )


def _simulate_open_loop(
    motor_params,
    target_speed,
    fixed_voltage,
    load_before,
    load_after,
    disturbance_time,
    t_span,
    num_points,
):
    """Simulate fixed-voltage open-loop motor response."""
    no_load_current = steady_state_current(
        motor_params["b"],
        motor_params["Kt"],
        target_speed,
        load_before,
    )
    voltage_func = lambda t: fixed_voltage
    load_torque_func = lambda t: load_torque_step(
        t,
        step_time=disturbance_time,
        initial=load_before,
        final=load_after,
    )

    time, current, speed = simulate_dc_motor(
        motor_params["R"],
        motor_params["L"],
        motor_params["J"],
        motor_params["b"],
        motor_params["Kt"],
        motor_params["Ke"],
        i0=no_load_current,
        omega0=target_speed,
        t_span=t_span,
        num_points=num_points,
        voltage_func=voltage_func,
        load_torque_func=load_torque_func,
    )
    voltage = np.full_like(time, fixed_voltage)
    load_torque = np.array([load_torque_func(sample_time) for sample_time in time])

    return {
        "time": time,
        "speed": speed,
        "current": current,
        "voltage": voltage,
        "load_torque": load_torque,
    }


def _simulate_pi_control(
    motor_params,
    target_speed,
    load_before,
    load_after,
    disturbance_time,
    t_span,
    num_points,
    voltage_limits,
):
    """Simulate continuous PI motor speed control with load disturbance."""
    Kp = 0.3
    Ki = 0.04
    Kd = 0.0
    reference_func = lambda t: speed_reference_step(t, target_speed)
    load_torque_func = lambda t: load_torque_step(
        t,
        step_time=disturbance_time,
        initial=load_before,
        final=load_after,
    )

    time, current, speed, _, voltage = simulate_pi_motor_control(
        motor_params["R"],
        motor_params["L"],
        motor_params["J"],
        motor_params["b"],
        motor_params["Kt"],
        motor_params["Ke"],
        Kp,
        Ki,
        i0=0.0,
        omega0=0.0,
        integral_error0=0.0,
        t_span=t_span,
        num_points=num_points,
        reference_func=reference_func,
        load_torque_func=load_torque_func,
        voltage_min=voltage_limits[0],
        voltage_max=voltage_limits[1],
        Kd=Kd,
    )
    load_torque = np.array([load_torque_func(sample_time) for sample_time in time])

    return {
        "time": time,
        "speed": speed,
        "current": current,
        "voltage": voltage,
        "load_torque": load_torque,
    }


def _simulate_discrete_pid(
    motor_params,
    target_speed,
    load_before,
    load_after,
    disturbance_time,
    t_span,
    voltage_limits,
):
    """Simulate discrete PID motor speed control with load disturbance."""
    return simulate_discrete_pid_motor_with_disturbance(
        target_speed=target_speed,
        Kp=0.5,
        Ki=0.2,
        Kd=0.02,
        dt=0.01,
        t_span=t_span,
        disturbance_time=disturbance_time,
        load_torque_initial=load_before,
        load_torque_final=load_after,
        voltage_limits=voltage_limits,
        motor_params=motor_params,
    )


def _draw_plots(simulations, target_speed, disturbance_time):
    """Draw speed, load torque, and voltage comparison plots."""
    figure, axes = plt.subplots(3, 1, sharex=True)

    for label, simulation in simulations.items():
        axes[0].plot(simulation["time"], simulation["speed"], label=label)
        axes[2].plot(simulation["time"], simulation["voltage"], label=label)

    first_simulation = next(iter(simulations.values()))

    axes[0].axhline(target_speed, color="gray", linestyle=":", label="Target speed")
    axes[0].axvline(disturbance_time, color="black", linestyle="--", label="Disturbance")
    axes[0].set_ylabel("Speed (rad/s)")
    axes[0].set_title("DC Motor Speed Under Load Disturbance")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(
        first_simulation["time"],
        first_simulation["load_torque"],
        label="Load torque",
        color="tab:red",
    )
    axes[1].axvline(disturbance_time, color="black", linestyle="--", label="Disturbance")
    axes[1].set_ylabel("Torque (N*m)")
    axes[1].set_title("Step Load Torque")
    axes[1].grid(True)
    axes[1].legend()

    axes[2].axvline(disturbance_time, color="black", linestyle="--", label="Disturbance")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Voltage (V)")
    axes[2].set_title("Control Effort")
    axes[2].grid(True)
    axes[2].legend()

    figure.tight_layout()


def _plot_comparison(simulations, target_speed, disturbance_time):
    """Plot comparison responses, falling back gracefully if Tk is unavailable."""
    try:
        _draw_plots(simulations, target_speed, disturbance_time)
        plt.show()
    except TclError:
        plt.switch_backend("Agg")
        plt.close("all")
        _draw_plots(simulations, target_speed, disturbance_time)
        print("Interactive Matplotlib window is unavailable in this environment.")


def main():
    """Compare open-loop, PI, and discrete PID load disturbance responses."""
    motor_params = {
        "R": 1.0,
        "L": 0.5,
        "J": 0.01,
        "b": 0.001,
        "Kt": 0.01,
        "Ke": 0.01,
    }
    target_speed = 80.0
    disturbance_time = 12.0
    load_before = 0.0
    load_after = 0.03
    t_span = (0.0, 25.0)
    num_points = 2501
    voltage_limits = (0.0, 24.0)
    fixed_voltage = target_speed * (
        motor_params["Ke"] + motor_params["R"] * motor_params["b"] / motor_params["Kt"]
    )

    simulations = {
        "Open-loop": _simulate_open_loop(
            motor_params,
            target_speed,
            fixed_voltage,
            load_before,
            load_after,
            disturbance_time,
            t_span,
            num_points,
        ),
        "Continuous PI": _simulate_pi_control(
            motor_params,
            target_speed,
            load_before,
            load_after,
            disturbance_time,
            t_span,
            num_points,
            voltage_limits,
        ),
        "Discrete PID": _simulate_discrete_pid(
            motor_params,
            target_speed,
            load_before,
            load_after,
            disturbance_time,
            t_span,
            voltage_limits,
        ),
    }

    results = []
    for label, simulation in simulations.items():
        metrics = summarize_motor_disturbance_response(
            simulation["time"],
            simulation["speed"],
            target_speed,
            disturbance_time,
            voltage=simulation["voltage"],
            current=simulation["current"],
        )
        results.append((label, metrics))

    print(f"Target speed: {target_speed:.3f} rad/s")
    print(f"Disturbance time: {disturbance_time:.3f} s")
    print(f"Load torque step: {load_before:.3f} to {load_after:.3f} N*m")
    print()
    _print_comparison_table(results)
    _plot_comparison(simulations, target_speed, disturbance_time)


if __name__ == "__main__":
    main()
