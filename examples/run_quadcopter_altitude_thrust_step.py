"""Run and plot a quadcopter altitude response to a thrust step."""

import sys
from pathlib import Path
from _tkinter import TclError

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.quadcopter_altitude import (
    hover_thrust,
    simulate_quadcopter_altitude,
    thrust_step,
)


def _draw_plots(t, z, v, thrust, thrust_hover, step_time):
    """Draw altitude, velocity, and thrust command plots."""
    figure, axes = plt.subplots(3, 1, sharex=True)

    axes[0].plot(t, z, label="Altitude")
    axes[0].axvline(step_time, color="black", linestyle="--", label="Thrust step")
    axes[0].set_ylabel("Altitude (m)")
    axes[0].set_title("Quadcopter Altitude Response to Thrust Step")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(t, v, label="Vertical velocity", color="tab:orange")
    axes[1].axvline(step_time, color="black", linestyle="--", label="Thrust step")
    axes[1].set_ylabel("Velocity (m/s)")
    axes[1].set_title("Vertical Velocity")
    axes[1].grid(True)
    axes[1].legend()

    axes[2].plot(t, thrust, label="Thrust command", color="tab:green")
    axes[2].axhline(thrust_hover, color="gray", linestyle=":", label="Hover thrust")
    axes[2].axvline(step_time, color="black", linestyle="--", label="Thrust step")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Thrust (N)")
    axes[2].set_title("Thrust Command")
    axes[2].grid(True)
    axes[2].legend()

    figure.tight_layout()


def _plot_response(t, z, v, thrust, thrust_hover, step_time):
    """Plot response, falling back gracefully if Tk is unavailable."""
    output_path = PROJECT_ROOT / "examples" / "quadcopter_altitude_thrust_step.png"

    try:
        _draw_plots(t, z, v, thrust, thrust_hover, step_time)
        plt.savefig(output_path, dpi=150)
        plt.show()
    except TclError:
        plt.switch_backend("Agg")
        plt.close("all")
        _draw_plots(t, z, v, thrust, thrust_hover, step_time)
        plt.savefig(output_path, dpi=150)
        print("Interactive Matplotlib window is unavailable in this environment.")
        print(f"Plot saved to: {output_path}")


def main():
    """Simulate hover followed by an upward thrust step."""
    m = 1.0
    g = 9.81
    c_drag = 0.2
    z0 = 0.0
    v0 = 0.0
    step_time = 2.0
    t_span = (0.0, 8.0)
    num_points = 1600
    thrust_hover = hover_thrust(m, g)
    thrust_after = 1.1 * thrust_hover
    thrust_func = thrust_step(step_time, thrust_hover, thrust_after)

    t, z, v, thrust = simulate_quadcopter_altitude(
        z0=z0,
        v0=v0,
        t_span=t_span,
        num_points=num_points,
        m=m,
        g=g,
        c_drag=c_drag,
        thrust_func=thrust_func,
    )

    print("Quadcopter Altitude Thrust Step:")
    print("State: [z, v], with positive altitude and velocity upward.")
    print(f"Mass: {m:.3f} kg")
    print(f"Gravity: {g:.3f} m/s^2")
    print(f"Linear drag coefficient: {c_drag:.3f} N*s/m")
    print(f"Hover thrust: {thrust_hover:.3f} N")
    print(f"Thrust step time: {step_time:.3f} s")
    print(f"Thrust after step: {thrust_after:.3f} N")
    print(f"Final altitude: {z[-1]:.3f} m")
    print(f"Final vertical velocity: {v[-1]:.3f} m/s")
    print()
    print("The vehicle starts at hover thrust, then climbs after thrust increases.")

    _plot_response(t, z, v, thrust, thrust_hover, step_time)


if __name__ == "__main__":
    main()
