"""Run a DC motor voltage-step response using state-space form."""

import sys
from pathlib import Path
from _tkinter import TclError

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.state_space import dc_motor_state_space, simulate_state_space, step_input
from models.dc_motor import rad_per_sec_to_rpm, steady_state_current, steady_state_speed


def _draw_plots(t, current, speed, steady_state_omega):
    """Draw motor current and speed plots."""
    figure, axes = plt.subplots(2, 1, sharex=True)

    axes[0].plot(t, current, label="Armature current")
    axes[0].set_ylabel("Current (A)")
    axes[0].set_title("DC Motor State-Space Armature Current")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(t, speed, label="Angular speed", color="tab:orange")
    axes[1].axhline(
        steady_state_omega,
        color="gray",
        linestyle=":",
        label="Steady state",
    )
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Angular speed (rad/s)")
    axes[1].set_title("DC Motor State-Space Speed Response")
    axes[1].grid(True)
    axes[1].legend()

    figure.tight_layout()


def _plot_response(t, current, speed, steady_state_omega):
    """Plot response, falling back gracefully if Tk is unavailable."""
    try:
        _draw_plots(t, current, speed, steady_state_omega)
        plt.show()
    except TclError:
        plt.switch_backend("Agg")
        plt.close("all")
        _draw_plots(t, current, speed, steady_state_omega)
        print("Interactive Matplotlib window is unavailable in this environment.")


def main():
    """Simulate a DC motor voltage-step response in state-space form."""
    R = 1.0
    L = 0.5
    J = 0.01
    b = 0.001
    Kt = 0.01
    Ke = 0.01
    voltage = 12.0
    x0 = [0.0, 0.0]
    t_span = (0.0, 80.0)
    num_points = 2000

    A, B, C, D = dc_motor_state_space(R, L, J, b, Kt, Ke)
    t, states, output = simulate_state_space(
        A,
        B,
        C,
        D,
        step_input(voltage),
        x0,
        t_span,
        num_points,
    )

    current = states[:, 0]
    speed = output[:, 0]
    steady_state_omega = steady_state_speed(R, b, Kt, Ke, voltage)
    steady_state_i = steady_state_current(b, Kt, steady_state_omega)

    print("DC Motor State-Space Response:")
    print(f"Resistance R: {R} ohms")
    print(f"Inductance L: {L} H")
    print(f"Rotor inertia J: {J} kg*m^2")
    print(f"Viscous damping b: {b} N*m*s/rad")
    print(f"Torque constant Kt: {Kt} N*m/A")
    print(f"Back-emf constant Ke: {Ke} V*s/rad")
    print(f"Input voltage: {voltage} V")
    print(f"Steady-state speed: {steady_state_omega:.3f} rad/s")
    print(f"Steady-state speed: {rad_per_sec_to_rpm(steady_state_omega):.3f} rpm")
    print(f"Steady-state current: {steady_state_i:.3f} A")
    print(f"Final simulated speed: {speed[-1]:.3f} rad/s")
    print(f"Final simulated current: {current[-1]:.3f} A")

    _plot_response(t, current, speed, steady_state_omega)


if __name__ == "__main__":
    main()
