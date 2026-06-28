"""Run and plot the simple pendulum example."""

import sys
from pathlib import Path
from _tkinter import TclError

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.pendulum import (
    natural_frequency,
    pendulum_energy,
    simulate_pendulum,
    small_angle_period,
)


def _plot_response(t, nonlinear_theta, linear_theta):
    """Plot nonlinear and linear pendulum angles."""
    output_path = PROJECT_ROOT / "examples" / "pendulum_response.png"

    try:
        _draw_plot(t, nonlinear_theta, linear_theta)
        plt.savefig(output_path, dpi=150)
        plt.show()
    except TclError:
        plt.switch_backend("Agg")
        plt.close("all")
        _draw_plot(t, nonlinear_theta, linear_theta)
        plt.savefig(output_path, dpi=150)
        print("Interactive Matplotlib window is unavailable in this environment.")
        print(f"Plot saved to: {output_path}")


def _draw_plot(t, nonlinear_theta, linear_theta):
    """Draw the nonlinear and small-angle pendulum comparison."""
    plt.plot(t, np.degrees(nonlinear_theta), label="Nonlinear pendulum")
    plt.plot(t, np.degrees(linear_theta), "--", label="Small-angle approximation")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle theta (degrees)")
    plt.title("Simple Pendulum: Nonlinear vs Small-Angle Approximation")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()


def main():
    """Simulate and compare nonlinear and linear pendulum motion."""
    L = 1.0
    g = 9.81
    theta0_degrees = 30
    theta0 = np.radians(theta0_degrees)
    omega0 = 0
    t_span = (0, 10)
    num_points = 2000

    omega_n = natural_frequency(L, g)
    period = small_angle_period(L, g)

    t, nonlinear_theta, nonlinear_omega = simulate_pendulum(
        L,
        theta0,
        omega0,
        t_span,
        num_points,
        g=g,
    )
    _, linear_theta, _ = simulate_pendulum(
        L,
        theta0,
        omega0,
        t_span,
        num_points,
        g=g,
        linear=True,
    )

    energy = pendulum_energy(nonlinear_theta, nonlinear_omega, L, g)
    initial_energy = energy[0]
    final_energy = energy[-1]
    relative_energy_drift = (final_energy - initial_energy) / initial_energy

    print("Pendulum Parameters:")
    print(f"Length: {L} m")
    print(f"Gravity: {g} m/s^2")
    print(f"Initial angle: {theta0_degrees} degrees")
    print(f"Initial angular velocity: {omega0} rad/s")
    print(f"Small-angle natural frequency: {omega_n:.3f} rad/s")
    print(f"Small-angle period: {period:.3f} s")
    print()
    print("Nonlinear Energy Check:")
    print(f"Initial energy: {initial_energy:.6f} J")
    print(f"Final energy: {final_energy:.6f} J")
    print(f"Relative energy drift: {relative_energy_drift:.3e}")

    _plot_response(t, nonlinear_theta, linear_theta)


if __name__ == "__main__":
    main()
