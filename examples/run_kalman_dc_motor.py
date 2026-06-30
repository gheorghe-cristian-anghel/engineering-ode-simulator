"""Estimate DC motor current and speed with a Kalman filter."""

import sys
from pathlib import Path
from _tkinter import TclError

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.kalman_filter import KalmanFilter, discretize_state_space
from analysis.state_space import dc_motor_state_space


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
    process_noise = np.diag([1e-4, 1e-3])
    measurement_noise = np.array([[4.0]])
    initial_estimate = np.array([0.0, 0.0])
    initial_covariance = np.diag([10.0, 100.0])

    kalman_filter = KalmanFilter(
        A=A,
        B=B,
        C=C,
        Q=process_noise,
        R=measurement_noise,
        x_hat=initial_estimate,
        P=initial_covariance,
        name="DC motor speed Kalman filter",
    )

    estimates = np.zeros((len(measurements), 2))
    estimates[0], _ = kalman_filter.update(measurements[0], input_value)

    for index in range(1, len(measurements)):
        estimates[index], _ = kalman_filter.step(measurements[index], input_value)

    return estimates


def _draw_plots(t, true_current, true_speed, measured_speed, estimated_states):
    """Draw DC motor Kalman filter plots."""
    estimated_current = estimated_states[:, 0]
    estimated_speed = estimated_states[:, 1]

    figure, axes = plt.subplots(3, 1, sharex=True)

    axes[0].plot(t, true_speed, label="True speed")
    axes[0].plot(t, measured_speed, ".", alpha=0.25, label="Noisy speed measurement")
    axes[0].plot(t, estimated_speed, "--", label="Estimated speed")
    axes[0].set_ylabel("Speed (rad/s)")
    axes[0].set_title("DC Motor Speed Estimate from Noisy Measurements")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(t, true_current, label="True current")
    axes[1].plot(t, estimated_current, "--", label="Estimated current")
    axes[1].set_ylabel("Current (A)")
    axes[1].set_title("Hidden Armature Current Estimate")
    axes[1].grid(True)
    axes[1].legend()

    axes[2].plot(t, true_speed - estimated_speed, label="Speed error")
    axes[2].plot(t, true_current - estimated_current, label="Current error")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Error")
    axes[2].set_title("Estimation Errors")
    axes[2].grid(True)
    axes[2].legend()

    figure.tight_layout()


def _plot_response(t, true_current, true_speed, measured_speed, estimated_states):
    """Plot response, falling back gracefully if Tk is unavailable."""
    output_path = PROJECT_ROOT / "examples" / "kalman_dc_motor.png"

    try:
        _draw_plots(t, true_current, true_speed, measured_speed, estimated_states)
        plt.savefig(output_path, dpi=150)
        plt.show()
    except TclError:
        plt.switch_backend("Agg")
        plt.close("all")
        _draw_plots(t, true_current, true_speed, measured_speed, estimated_states)
        plt.savefig(output_path, dpi=150)
        print("Interactive Matplotlib window is unavailable in this environment.")
        print(f"Plot saved to: {output_path}")


def main():
    """Estimate DC motor current and speed from noisy speed measurements."""
    R = 1.0
    L = 0.5
    J = 0.01
    b = 0.001
    Kt = 0.01
    Ke = 0.01
    voltage = 12.0
    dt = 0.01
    t_final = 8.0
    num_points = int(t_final / dt) + 1
    rng = np.random.default_rng(0)
    measurement_noise_std = 2.0

    A, B, _, _ = dc_motor_state_space(R, L, J, b, Kt, Ke)
    A_d, B_d = discretize_state_space(A, B, dt)
    measurement_matrix = np.array([[0.0, 1.0]])

    t = np.linspace(0.0, t_final, num_points)
    true_states = _simulate_discrete_system(A_d, B_d, [0.0, 0.0], voltage, num_points)
    true_current = true_states[:, 0]
    true_speed = true_states[:, 1]
    measured_speed = true_speed + rng.normal(0.0, measurement_noise_std, num_points)
    estimated_states = _run_filter(A_d, B_d, measurement_matrix, measured_speed, voltage)

    estimated_current = estimated_states[:, 0]
    estimated_speed = estimated_states[:, 1]
    speed_error = true_speed - estimated_speed
    current_error = true_current - estimated_current

    print("DC Motor Kalman Filter:")
    print(f"Voltage input: {voltage:.3f} V")
    print(f"Measurement noise standard deviation: {measurement_noise_std:.3f} rad/s")
    print(f"Final true speed: {true_speed[-1]:.3f} rad/s")
    print(f"Final estimated speed: {estimated_speed[-1]:.3f} rad/s")
    print(f"Final speed estimation error: {speed_error[-1]:.3f} rad/s")
    print(f"RMS speed estimation error: {np.sqrt(np.mean(speed_error**2)):.3f} rad/s")
    print(f"RMS current estimation error: {np.sqrt(np.mean(current_error**2)):.3f} A")
    print()
    print("The Kalman filter estimates hidden current and smooths noisy speed measurements.")

    _plot_response(t, true_current, true_speed, measured_speed, estimated_states)


if __name__ == "__main__":
    main()
