"""Run a mass-spring-damper step response using state-space form."""

import sys
from pathlib import Path
from _tkinter import TclError

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.state_space import (
    mass_spring_damper_state_space,
    simulate_state_space,
    step_input,
)
from models.mass_spring_damper import damping_ratio, natural_frequency


def _draw_plots(t, displacement, velocity):
    """Draw displacement and velocity plots."""
    figure, axes = plt.subplots(2, 1, sharex=True)

    axes[0].plot(t, displacement, label="Displacement")
    axes[0].set_ylabel("Displacement (m)")
    axes[0].set_title("Mass-Spring-Damper State-Space Response")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(t, velocity, label="Velocity", color="tab:orange")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Velocity (m/s)")
    axes[1].set_title("Mass Velocity")
    axes[1].grid(True)
    axes[1].legend()

    figure.tight_layout()


def _plot_response(t, displacement, velocity):
    """Plot response, falling back gracefully if Tk is unavailable."""
    try:
        _draw_plots(t, displacement, velocity)
        plt.show()
    except TclError:
        plt.switch_backend("Agg")
        plt.close("all")
        _draw_plots(t, displacement, velocity)
        print("Interactive Matplotlib window is unavailable in this environment.")


def main():
    """Simulate a mass-spring-damper force-step response."""
    m = 1.0
    c = 0.4
    k = 4.0
    force = 1.0
    x0 = [0.0, 0.0]
    t_span = (0.0, 20.0)
    num_points = 1000

    A, B, C, D = mass_spring_damper_state_space(m, c, k)
    t, states, output = simulate_state_space(
        A,
        B,
        C,
        D,
        step_input(force),
        x0,
        t_span,
        num_points,
    )

    displacement = states[:, 0]
    velocity = states[:, 1]
    final_displacement = output[-1, 0]

    print("Mass-Spring-Damper State-Space Response:")
    print(f"Mass: {m} kg")
    print(f"Damping coefficient: {c} N*s/m")
    print(f"Spring stiffness: {k} N/m")
    print(f"Step force: {force} N")
    print(f"Natural frequency: {natural_frequency(m, k):.3f} rad/s")
    print(f"Damping ratio: {damping_ratio(m, c, k):.3f}")
    print(f"Expected static displacement: {force / k:.3f} m")
    print(f"Final simulated displacement: {final_displacement:.3f} m")

    _plot_response(t, displacement, velocity)


if __name__ == "__main__":
    main()
