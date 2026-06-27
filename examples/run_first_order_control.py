"""Run and plot the first-order control system step response example."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.first_order_control import (
    analytical_step_response,
    rise_time,
    settling_time,
    simulate_first_order_system,
    steady_state_value,
    step_input,
)


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

    print("Natural control metrics:")
    print(f"Gain K: {K}")
    print(f"Time constant tau: {tau} s")
    print(f"Input amplitude: {amplitude}")
    print(f"Steady-state value: {final_value}")
    print(f"Rise time: {response_rise_time:.3f} s")
    print(f"Settling time: {response_settling_time:.3f} s")

    output_path = PROJECT_ROOT / "examples" / "first_order_control_response.png"

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


if __name__ == "__main__":
    main()
