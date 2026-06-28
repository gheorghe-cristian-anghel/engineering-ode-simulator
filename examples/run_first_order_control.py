"""Run and plot the first-order control system step response example."""

import sys
from pathlib import Path
from _tkinter import TclError

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.step_response import calculate_step_info
from models.first_order_control import (
    analytical_step_response,
    rise_time,
    settling_time,
    simulate_first_order_system,
    steady_state_value,
    step_input,
)


def _plot_response(t, numerical_output, analytical_output, final_value, output_path):
    """Plot the response, falling back to saving only if Tk is unavailable."""
    try:
        plt.plot(t, numerical_output, label="Numerical response")
        plt.plot(t, analytical_output, "--", label="Analytical response")
        plt.axhline(final_value, color="gray", linestyle=":", label="Steady state")
        plt.xlabel("Time (s)")
        plt.ylabel("Output y")
        plt.title("First-Order Control System Step Response")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        print(f"Plot saved to: {output_path}")
        plt.show()
    except TclError:
        plt.switch_backend("Agg")
        plt.close("all")
        plt.plot(t, numerical_output, label="Numerical response")
        plt.plot(t, analytical_output, "--", label="Analytical response")
        plt.axhline(final_value, color="gray", linestyle=":", label="Steady state")
        plt.xlabel("Time (s)")
        plt.ylabel("Output y")
        plt.title("First-Order Control System Step Response")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        print("Interactive Matplotlib window is unavailable in this environment.")
        print(f"Plot saved to: {output_path}")


def main():
    """Simulate a first-order step response and plot the result."""
    tau = 1.5
    K = 2.0
    amplitude = 1.0
    y0 = 0.0
    t_span = (0, 10)
    num_points = 300

    input_func = lambda t: step_input(t, amplitude)

    t, numerical_output = simulate_first_order_system(
        tau,
        K,
        y0,
        t_span,
        num_points,
        input_func=input_func,
    )
    analytical_output = analytical_step_response(t, tau, K, amplitude, y0)
    final_value = steady_state_value(K, amplitude)
    response_rise_time = rise_time(tau)
    response_settling_time = settling_time(tau)
    step_info = calculate_step_info(t, numerical_output)

    print("Natural control metrics:")
    print(f"Gain K: {K}")
    print(f"Time constant tau: {tau} s")
    print(f"Input amplitude: {amplitude}")
    print(f"Steady-state value: {final_value}")
    print(f"Rise time: {response_rise_time:.3f} s")
    print(f"Settling time: {response_settling_time:.3f} s")
    print()
    print("Step Response Metrics:")
    print(f"Initial value: {step_info.initial_value:.3f}")
    print(f"Final value: {step_info.final_value:.3f}")
    print(f"Estimated steady-state value: {step_info.steady_state_value:.3f}")
    print(f"Rise time: {step_info.rise_time:.3f} s")
    print(f"Settling time: {step_info.settling_time:.3f} s")
    print(f"Peak value: {step_info.peak_value:.3f}")
    print(f"Peak time: {step_info.peak_time:.3f} s")
    print(f"Overshoot: {step_info.overshoot_percent:.3f}%")

    output_path = PROJECT_ROOT / "examples" / "first_order_control_response.png"

    _plot_response(t, numerical_output, analytical_output, final_value, output_path)


if __name__ == "__main__":
    main()
