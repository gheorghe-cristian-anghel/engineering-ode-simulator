"""Run constrained MPC position tracking for a double integrator."""

import sys
from pathlib import Path
from _tkinter import TclError

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.model_predictive_control import (
    LinearMPC,
    discrete_double_integrator,
    simulate_mpc_tracking,
    summarize_mpc_response,
)


def _draw_plots(result, dt, u_min, u_max):
    """Draw position, velocity, control, and tracking-error plots."""
    states = result["states"]
    controls = result["controls"]
    references = result["references"]
    time = result["steps"] * dt
    control_time = time[:-1]
    position_error = references[:, 0] - states[:, 0]

    figure, axes = plt.subplots(4, 1, sharex=True)

    axes[0].plot(time, states[:, 0], label="Position")
    axes[0].plot(time, references[:, 0], ":", color="gray", label="Reference")
    axes[0].set_ylabel("Position (m)")
    axes[0].set_title("MPC Double Integrator Position Tracking")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(time, states[:, 1], label="Velocity", color="tab:orange")
    axes[1].plot(time, references[:, 1], ":", color="gray", label="Reference")
    axes[1].set_ylabel("Velocity (m/s)")
    axes[1].set_title("Velocity")
    axes[1].grid(True)
    axes[1].legend()

    axes[2].step(
        control_time,
        controls[:, 0],
        where="post",
        label="Acceleration command",
        color="tab:green",
    )
    axes[2].axhline(u_min[0], color="gray", linestyle=":", label="Input limits")
    axes[2].axhline(u_max[0], color="gray", linestyle=":")
    axes[2].set_ylabel("Acceleration (m/s^2)")
    axes[2].set_title("Constrained MPC Input")
    axes[2].grid(True)
    axes[2].legend()

    axes[3].plot(time, position_error, label="Position error", color="tab:red")
    axes[3].axhline(0.0, color="gray", linestyle=":")
    axes[3].set_xlabel("Time (s)")
    axes[3].set_ylabel("Error (m)")
    axes[3].set_title("Tracking Error")
    axes[3].grid(True)
    axes[3].legend()

    figure.tight_layout()


def _plot_response(result, dt, u_min, u_max):
    """Plot response, falling back gracefully if Tk is unavailable."""
    output_path = PROJECT_ROOT / "examples" / "mpc_double_integrator.png"

    try:
        _draw_plots(result, dt, u_min, u_max)
        plt.savefig(output_path, dpi=150)
        if plt.get_backend().lower() == "agg":
            print(f"Plot saved to: {output_path}")
        else:
            plt.show()
    except TclError:
        plt.switch_backend("Agg")
        plt.close("all")
        _draw_plots(result, dt, u_min, u_max)
        plt.savefig(output_path, dpi=150)
        print("Interactive Matplotlib window is unavailable in this environment.")
        print(f"Plot saved to: {output_path}")


def main():
    """Simulate constrained receding-horizon MPC for position control."""
    dt = 0.1
    target_position = 10.0
    target_velocity = 0.0
    x0 = np.array([0.0, 0.0])
    x_ref = np.array([target_position, target_velocity])
    horizon = 20
    Q = np.diag([10.0, 1.0])
    R = np.array([[0.1]])
    u_min = np.array([-2.0])
    u_max = np.array([2.0])
    num_steps = 120

    A, B = discrete_double_integrator(dt)
    mpc = LinearMPC(
        A=A,
        B=B,
        Q=Q,
        R=R,
        horizon=horizon,
        u_min=u_min,
        u_max=u_max,
    )
    result = simulate_mpc_tracking(
        mpc=mpc,
        x0=x0,
        x_ref=x_ref,
        num_steps=num_steps,
    )
    metrics = summarize_mpc_response(result)

    print("Linear MPC Double Integrator Tracking:")
    print(
        "MPC chooses acceleration commands over a finite prediction horizon "
        "while respecting input limits."
    )
    print()
    print(f"Target position: {target_position:.3f} m")
    print(f"Target velocity: {target_velocity:.3f} m/s")
    print(f"Final position: {metrics.final_position:.3f} m")
    print(f"Final velocity: {metrics.final_velocity:.3f} m/s")
    print(f"Final position error: {metrics.final_position_error:.3f} m")
    print(f"RMS position error: {metrics.rms_position_error:.3f} m")
    print(f"Max acceleration command: {metrics.max_abs_control:.3f} m/s^2")
    print(f"Prediction horizon: {horizon} steps")
    print(f"Input limits: {u_min[0]:.3f} to {u_max[0]:.3f} m/s^2")
    print(f"All optimizations successful: {bool(np.all(result['success']))}")

    _plot_response(result, dt, u_min, u_max)


if __name__ == "__main__":
    main()
