"""Estimate nonlinear pendulum state with an Extended Kalman Filter."""

import sys
from pathlib import Path
from _tkinter import TclError

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.extended_kalman_filter import (
    ExtendedKalmanFilter,
    pendulum_angle_measurement,
    pendulum_angle_measurement_jacobian,
    pendulum_ekf_state_jacobian,
    pendulum_ekf_state_transition,
)
from models.pendulum import simulate_pendulum


def _run_pendulum_ekf(t, measurements, L, g, initial_estimate):
    """Run the pendulum EKF over noisy angle measurements."""
    measurement_noise_std = np.radians(2.0)
    dt = t[1] - t[0]

    ekf = ExtendedKalmanFilter(
        f=lambda x, u, sample_time: pendulum_ekf_state_transition(
            x,
            u,
            sample_time,
            L=L,
            g=g,
        ),
        h=pendulum_angle_measurement,
        F_jacobian=lambda x, u, sample_time: pendulum_ekf_state_jacobian(
            x,
            u,
            sample_time,
            L=L,
            g=g,
        ),
        H_jacobian=pendulum_angle_measurement_jacobian,
        Q=np.diag([1e-6, 1e-4]),
        R=np.array([[measurement_noise_std**2]]),
        x_hat=initial_estimate,
        P=np.diag([0.1, 1.0]),
        name="Pendulum EKF",
    )

    estimates = np.zeros((len(measurements), 2))
    estimates[0], _ = ekf.update(measurements[0])

    for index in range(1, len(measurements)):
        estimates[index], _ = ekf.step(measurements[index], dt=dt)

    return estimates


def _draw_plots(t, true_theta, true_omega, measured_theta, estimated_states):
    """Draw nonlinear pendulum EKF state-estimation plots."""
    estimated_theta = estimated_states[:, 0]
    estimated_omega = estimated_states[:, 1]

    figure, axes = plt.subplots(3, 1, sharex=True)

    axes[0].plot(t, np.degrees(true_theta), label="True angle")
    axes[0].plot(
        t,
        np.degrees(measured_theta),
        ".",
        alpha=0.25,
        label="Noisy angle measurement",
    )
    axes[0].plot(t, np.degrees(estimated_theta), "--", label="Estimated angle")
    axes[0].set_ylabel("Angle (deg)")
    axes[0].set_title("Nonlinear Pendulum EKF: Angle Estimate")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(t, true_omega, label="True angular velocity")
    axes[1].plot(t, estimated_omega, "--", label="Estimated angular velocity")
    axes[1].set_ylabel("Angular velocity (rad/s)")
    axes[1].set_title("Hidden Angular Velocity Estimate")
    axes[1].grid(True)
    axes[1].legend()

    axes[2].plot(
        t,
        np.degrees(true_theta - estimated_theta),
        label="Angle error",
    )
    axes[2].plot(t, true_omega - estimated_omega, label="Angular velocity error")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Error")
    axes[2].set_title("Estimation Errors")
    axes[2].grid(True)
    axes[2].legend()

    figure.tight_layout()


def _plot_response(t, true_theta, true_omega, measured_theta, estimated_states):
    """Plot response, falling back gracefully if Tk is unavailable."""
    output_path = PROJECT_ROOT / "examples" / "pendulum_ekf.png"

    try:
        _draw_plots(t, true_theta, true_omega, measured_theta, estimated_states)
        plt.savefig(output_path, dpi=150)
        plt.show()
    except TclError:
        plt.switch_backend("Agg")
        plt.close("all")
        _draw_plots(t, true_theta, true_omega, measured_theta, estimated_states)
        plt.savefig(output_path, dpi=150)
        print("Interactive Matplotlib window is unavailable in this environment.")
        print(f"Plot saved to: {output_path}")


def main():
    """Estimate pendulum angle and angular velocity from noisy angle data."""
    L = 1.0
    g = 9.81
    theta0_degrees = 20.0
    theta0 = np.radians(theta0_degrees)
    omega0 = 0.0
    t_final = 10.0
    dt = 0.01
    num_points = int(t_final / dt) + 1
    rng = np.random.default_rng(2)
    measurement_noise_std = np.radians(2.0)
    initial_estimate = np.array([theta0 + np.radians(5.0), 0.5])

    t, true_theta, true_omega = simulate_pendulum(
        L,
        theta0,
        omega0,
        (0.0, t_final),
        num_points,
        g=g,
    )
    measured_theta = true_theta + rng.normal(0.0, measurement_noise_std, len(t))
    estimated_states = _run_pendulum_ekf(
        t,
        measured_theta,
        L,
        g,
        initial_estimate,
    )

    estimated_theta = estimated_states[:, 0]
    estimated_omega = estimated_states[:, 1]
    angle_error = true_theta - estimated_theta
    omega_error = true_omega - estimated_omega

    print("Nonlinear Pendulum Extended Kalman Filter:")
    print(f"Pendulum length: {L:.3f} m")
    print(f"Initial true angle: {theta0_degrees:.3f} degrees")
    print(f"Angle measurement noise std: {np.degrees(measurement_noise_std):.3f} degrees")
    print(f"Final true angle: {np.degrees(true_theta[-1]):.3f} degrees")
    print(f"Final estimated angle: {np.degrees(estimated_theta[-1]):.3f} degrees")
    print(f"Final angle error: {np.degrees(angle_error[-1]):.3f} degrees")
    print(
        "RMS angle estimation error: "
        f"{np.degrees(np.sqrt(np.mean(angle_error**2))):.3f} degrees"
    )
    print(
        "RMS angular velocity estimation error: "
        f"{np.sqrt(np.mean(omega_error**2)):.3f} rad/s"
    )
    print()
    print(
        "The EKF estimates both measured angle and hidden angular velocity "
        "using the nonlinear pendulum model."
    )

    _plot_response(t, true_theta, true_omega, measured_theta, estimated_states)


if __name__ == "__main__":
    main()
