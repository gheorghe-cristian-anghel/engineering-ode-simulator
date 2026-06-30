"""Compare open-loop quadcopter altitude responses for constant thrust levels."""

import sys
from pathlib import Path
from _tkinter import TclError

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.quadcopter_altitude import (
    constant_thrust,
    hover_thrust,
    simulate_quadcopter_altitude,
)


def _draw_plots(results):
    """Draw altitude and velocity responses for each thrust level."""
    figure, axes = plt.subplots(2, 1, sharex=True)

    for label, result in results.items():
        axes[0].plot(result["t"], result["z"], label=label)
        axes[1].plot(result["t"], result["v"], label=label)

    axes[0].set_ylabel("Altitude (m)")
    axes[0].set_title("Open-Loop Quadcopter Altitude")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Vertical velocity (m/s)")
    axes[1].set_title("Vertical Velocity")
    axes[1].grid(True)
    axes[1].legend()

    figure.tight_layout()


def _plot_response(results):
    """Plot response, falling back gracefully if Tk is unavailable."""
    output_path = PROJECT_ROOT / "examples" / "quadcopter_altitude_open_loop.png"

    try:
        _draw_plots(results)
        plt.savefig(output_path, dpi=150)
        plt.show()
    except TclError:
        plt.switch_backend("Agg")
        plt.close("all")
        _draw_plots(results)
        plt.savefig(output_path, dpi=150)
        print("Interactive Matplotlib window is unavailable in this environment.")
        print(f"Plot saved to: {output_path}")


def main():
    """Simulate descent, hover, and climb with constant thrust commands."""
    m = 1.0
    g = 9.81
    c_drag = 0.0
    z0 = 0.0
    v0 = 0.0
    t_span = (0.0, 4.0)
    num_points = 1000
    thrust_hover = hover_thrust(m, g)
    thrust_cases = {
        "Below hover (0.9 T_hover)": 0.9 * thrust_hover,
        "Hover (1.0 T_hover)": thrust_hover,
        "Above hover (1.1 T_hover)": 1.1 * thrust_hover,
    }

    results = {}
    for label, thrust in thrust_cases.items():
        t, z, v, thrust_history = simulate_quadcopter_altitude(
            z0=z0,
            v0=v0,
            t_span=t_span,
            num_points=num_points,
            m=m,
            g=g,
            c_drag=c_drag,
            thrust_func=constant_thrust(thrust),
        )
        results[label] = {
            "t": t,
            "z": z,
            "v": v,
            "thrust": thrust_history,
        }

    print("Open-Loop Quadcopter Altitude:")
    print("State: [z, v], with positive altitude and velocity upward.")
    print(f"Mass: {m:.3f} kg")
    print(f"Gravity: {g:.3f} m/s^2")
    print(f"Linear drag coefficient: {c_drag:.3f} N*s/m")
    print(f"Hover thrust: {thrust_hover:.3f} N")
    print()

    for label, result in results.items():
        print(label)
        print(f"  thrust: {result['thrust'][-1]:.3f} N")
        print(f"  final altitude: {result['z'][-1]:.3f} m")
        print(f"  final vertical velocity: {result['v'][-1]:.3f} m/s")

    print()
    print("Below-hover thrust descends, hover thrust maintains altitude, and")
    print("above-hover thrust climbs in this simplified vertical model.")

    _plot_response(results)


if __name__ == "__main__":
    main()
