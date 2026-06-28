"""Run and plot the DC motor speed response example."""

import sys
from pathlib import Path
from _tkinter import TclError

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.dc_motor import (
    rad_per_sec_to_rpm,
    simulate_dc_motor,
    steady_state_current,
    steady_state_speed,
    voltage_step,
    zero_load_torque,
)


def _plot_response(t, current, omega, omega_ss):
    """Plot current and speed responses, falling back to saving if needed."""
    output_path = PROJECT_ROOT / "examples" / "dc_motor_response.png"

    try:
        _draw_plots(t, current, omega, omega_ss)
        plt.savefig(output_path, dpi=150)
        plt.show()
    except TclError:
        plt.switch_backend("Agg")
        plt.close("all")
        _draw_plots(t, current, omega, omega_ss)
        plt.savefig(output_path, dpi=150)
        print("Interactive Matplotlib window is unavailable in this environment.")
        print(f"Plot saved to: {output_path}")


def _draw_plots(t, current, omega, omega_ss):
    """Draw DC motor current and speed plots."""
    figure, axes = plt.subplots(2, 1, sharex=True)

    axes[0].plot(t, current, label="Armature current")
    axes[0].set_ylabel("Current (A)")
    axes[0].set_title("DC Motor Armature Current")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(t, omega, label="Angular speed", color="tab:orange")
    axes[1].axhline(omega_ss, color="gray", linestyle=":", label="Steady state")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Angular speed (rad/s)")
    axes[1].set_title("DC Motor Speed Response")
    axes[1].grid(True)
    axes[1].legend()

    figure.tight_layout()


def main():
    """Simulate a permanent-magnet DC motor speed response."""
    R = 1.0
    L = 0.5
    J = 0.01
    b = 0.001
    Kt = 0.01
    Ke = 0.01
    voltage = 12.0
    load_torque = 0.0
    i0 = 0.0
    omega0 = 0.0
    t_span = (0, 5)
    num_points = 2000

    voltage_func = lambda t: voltage_step(t, voltage)
    load_torque_func = zero_load_torque

    omega_ss = steady_state_speed(R, b, Kt, Ke, voltage, load_torque)
    current_ss = steady_state_current(b, Kt, omega_ss, load_torque)
    speed_rpm = rad_per_sec_to_rpm(omega_ss)

    print("DC Motor Parameters:")
    print(f"Resistance R: {R} ohms")
    print(f"Inductance L: {L} H")
    print(f"Rotor inertia J: {J} kg*m^2")
    print(f"Viscous damping b: {b} N*m*s/rad")
    print(f"Torque constant Kt: {Kt} N*m/A")
    print(f"Back-emf constant Ke: {Ke} V*s/rad")
    print(f"Input voltage: {voltage} V")
    print(f"Load torque: {load_torque} N*m")
    print()
    print("Steady-State Estimates:")
    print(f"Speed: {omega_ss:.3f} rad/s")
    print(f"Speed: {speed_rpm:.3f} rpm")
    print(f"Current: {current_ss:.3f} A")

    t, current, omega = simulate_dc_motor(
        R,
        L,
        J,
        b,
        Kt,
        Ke,
        i0,
        omega0,
        t_span,
        num_points,
        voltage_func=voltage_func,
        load_torque_func=load_torque_func,
    )

    _plot_response(t, current, omega, omega_ss)


if __name__ == "__main__":
    main()
