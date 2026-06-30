"""Estimate RLC capacitor voltage and current with a Kalman filter."""

import sys
from pathlib import Path
from _tkinter import TclError

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.kalman_filter import KalmanFilter, discretize_state_space
from analysis.state_space import rlc_state_space


def _simulate_discrete_system(A, B, x0, input_value, num_points):
    """Simulate a discrete-time linear system with constant input."""
    states = np.zeros((num_points, len(x0)))
    states[0] = x0
    input_vector = np.asarray([input_value], dtype=float)

    for index in range(num_points - 1):
        states[index + 1] = A @ states[index] + B @ input_vector

    return states


def _run_filter(A, B, C, measurements, input_value):
    """Run the Kalman filter over all noisy measurements."""
    process_noise = np.diag([1e-5, 1e-4])
    measurement_noise = np.array([[0.04]])
    initial_estimate = np.array([0.0, 0.0])
    initial_covariance = np.diag([1.0, 1.0])

    kalman_filter = KalmanFilter(
        A=A,
        B=B,
        C=C,
        Q=process_noise,
        R=measurement_noise,
        x_hat=initial_estimate,
        P=initial_covariance,
        name="RLC voltage Kalman filter",
    )

    estimates = np.zeros((len(measurements), 2))
    estimates[0], _ = kalman_filter.update(measurements[0], input_value)

    for index in range(1, len(measurements)):
        estimates[index], _ = kalman_filter.step(measurements[index], input_value)

    return estimates


def _draw_plots(t, true_voltage, true_current, measured_voltage, estimated_states):
    """Draw RLC Kalman filter plots."""
    estimated_voltage = estimated_states[:, 0]
    estimated_current = estimated_states[:, 1]

    figure, axes = plt.subplots(3, 1, sharex=True)

    axes[0].plot(t, true_voltage, label="True capacitor voltage")
    axes[0].plot(t, measured_voltage, ".", alpha=0.25, label="Noisy voltage measurement")
    axes[0].plot(t, estimated_voltage, "--", label="Estimated voltage")
    axes[0].set_ylabel("Voltage (V)")
    axes[0].set_title("RLC Capacitor Voltage Estimate")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(t, true_current, label="True current")
    axes[1].plot(t, estimated_current, "--", label="Estimated current")
    axes[1].set_ylabel("Current (A)")
    axes[1].set_title("Hidden Inductor Current Estimate")
    axes[1].grid(True)
    axes[1].legend()

    axes[2].plot(t, true_voltage - estimated_voltage, label="Voltage error")
    axes[2].plot(t, true_current - estimated_current, label="Current error")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Error")
    axes[2].set_title("Estimation Errors")
    axes[2].grid(True)
    axes[2].legend()

    figure.tight_layout()


def _plot_response(t, true_voltage, true_current, measured_voltage, estimated_states):
    """Plot response, falling back gracefully if Tk is unavailable."""
    output_path = PROJECT_ROOT / "examples" / "kalman_rlc.png"

    try:
        _draw_plots(t, true_voltage, true_current, measured_voltage, estimated_states)
        plt.savefig(output_path, dpi=150)
        plt.show()
    except TclError:
        plt.switch_backend("Agg")
        plt.close("all")
        _draw_plots(t, true_voltage, true_current, measured_voltage, estimated_states)
        plt.savefig(output_path, dpi=150)
        print("Interactive Matplotlib window is unavailable in this environment.")
        print(f"Plot saved to: {output_path}")


def main():
    """Estimate RLC voltage and current from noisy voltage measurements."""
    R = 2.0
    L = 1.0
    C_value = 0.25
    input_voltage = 5.0
    dt = 0.01
    t_final = 10.0
    num_points = int(t_final / dt) + 1
    rng = np.random.default_rng(1)
    measurement_noise_std = 0.2

    A, B, C, _ = rlc_state_space(R, L, C_value)
    A_d, B_d = discretize_state_space(A, B, dt)

    t = np.linspace(0.0, t_final, num_points)
    true_states = _simulate_discrete_system(
        A_d,
        B_d,
        [0.0, 0.0],
        input_voltage,
        num_points,
    )
    true_voltage = true_states[:, 0]
    true_current = true_states[:, 1]
    measured_voltage = true_voltage + rng.normal(0.0, measurement_noise_std, num_points)
    estimated_states = _run_filter(A_d, B_d, C, measured_voltage, input_voltage)

    estimated_voltage = estimated_states[:, 0]
    estimated_current = estimated_states[:, 1]
    voltage_error = true_voltage - estimated_voltage
    current_error = true_current - estimated_current

    print("RLC Kalman Filter:")
    print(f"Input voltage: {input_voltage:.3f} V")
    print(f"Measurement noise standard deviation: {measurement_noise_std:.3f} V")
    print(f"Final true capacitor voltage: {true_voltage[-1]:.3f} V")
    print(f"Final estimated capacitor voltage: {estimated_voltage[-1]:.3f} V")
    print(f"RMS voltage estimation error: {np.sqrt(np.mean(voltage_error**2)):.3f} V")
    print(f"RMS current estimation error: {np.sqrt(np.mean(current_error**2)):.3f} A")
    print()
    print(
        "The Kalman filter reconstructs unmeasured current from noisy voltage "
        "measurements using the system model."
    )

    _plot_response(t, true_voltage, true_current, measured_voltage, estimated_states)


if __name__ == "__main__":
    main()
