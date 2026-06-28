"""Run and plot the second-order control system step response example."""

import sys
from pathlib import Path
from _tkinter import TclError

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.step_response import calculate_step_info
from models.second_order_control import (
    approx_settling_time,
    damped_natural_frequency,
    response_type,
    simulate_second_order_system,
    step_input,
    theoretical_overshoot_percent,
    theoretical_peak_time,
)


def _format_optional(value, unit=""):
    """Format optional metric values for printing."""
    if value is None:
        return "None"

    return f"{value:.3f}{unit}"


def _plot_response(t, output, amplitude):
    """Plot the step response, falling back to saving if Tk is unavailable."""
    output_path = PROJECT_ROOT / "examples" / "second_order_control_response.png"

    try:
        _draw_plot(t, output, amplitude)
        plt.savefig(output_path, dpi=150)
        plt.show()
    except TclError:
        plt.switch_backend("Agg")
        plt.close("all")
        _draw_plot(t, output, amplitude)
        plt.savefig(output_path, dpi=150)
        print("Interactive Matplotlib window is unavailable in this environment.")
        print(f"Plot saved to: {output_path}")


def _draw_plot(t, output, amplitude):
    """Draw the second-order step response plot."""
    plt.plot(t, output, label="Output y(t)")
    plt.axhline(amplitude, color="gray", linestyle=":", label="Steady state")
    plt.xlabel("Time (s)")
    plt.ylabel("Output y")
    plt.title("Second-Order Control System Step Response")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()


def main():
    """Simulate and plot an underdamped second-order step response."""
    omega_n = 4.0
    zeta = 0.3
    amplitude = 1.0
    y0 = 0.0
    v0 = 0.0
    t_span = (0, 8)
    num_points = 2000

    input_func = lambda t: step_input(t, amplitude)

    t, output, _ = simulate_second_order_system(
        omega_n,
        zeta,
        y0,
        v0,
        t_span,
        num_points,
        input_func=input_func,
    )
    step_info = calculate_step_info(t, output)
    omega_d = damped_natural_frequency(omega_n, zeta)
    overshoot = theoretical_overshoot_percent(zeta)
    peak_time = theoretical_peak_time(omega_n, zeta)
    settling = approx_settling_time(omega_n, zeta)

    print("Second-Order Control Parameters:")
    print(f"Natural frequency omega_n: {omega_n:.3f} rad/s")
    print(f"Damping ratio zeta: {zeta:.3f}")
    print(f"Response type: {response_type(zeta)}")
    print(f"Damped natural frequency: {_format_optional(omega_d, ' rad/s')}")
    print(f"Theoretical overshoot: {overshoot:.3f}%")
    print(f"Theoretical peak time: {_format_optional(peak_time, ' s')}")
    print(f"Approximate 2% settling time: {_format_optional(settling, ' s')}")
    print()
    print("Measured Step Response Metrics:")
    print(f"Final value: {step_info.final_value:.3f}")
    print(f"Steady-state estimate: {step_info.steady_state_value:.3f}")
    print(f"Rise time: {_format_optional(step_info.rise_time, ' s')}")
    print(f"Settling time: {_format_optional(step_info.settling_time, ' s')}")
    print(f"Peak value: {step_info.peak_value:.3f}")
    print(f"Peak time: {step_info.peak_time:.3f} s")
    print(f"Overshoot: {step_info.overshoot_percent:.3f}%")

    _plot_response(t, output, amplitude)


if __name__ == "__main__":
    main()
