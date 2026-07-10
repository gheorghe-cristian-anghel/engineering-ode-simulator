"""Estimate nonlinear pendulum state with an Unscented Kalman Filter."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.unscented_kalman_filter import UnscentedKalmanFilter  # noqa: E402
from visualization.plot_style import (  # noqa: E402
    apply_plot_style,
    format_axes,
    place_legends_outside,
    save_figure,
)


def _pendulum_derivative(state, L, g, damping):
    """Return nonlinear damped pendulum state derivatives."""
    theta, omega = state

    return np.array(
        [
            omega,
            -(g / L) * np.sin(theta) - damping * omega,
        ]
    )


def _rk4_pendulum_step(state, dt, L, g, damping):
    """Advance the pendulum state by one RK4 integration step."""
    k1 = _pendulum_derivative(state, L, g, damping)
    k2 = _pendulum_derivative(state + 0.5 * dt * k1, L, g, damping)
    k3 = _pendulum_derivative(state + 0.5 * dt * k2, L, g, damping)
    k4 = _pendulum_derivative(state + dt * k3, L, g, damping)

    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _pendulum_measurement(state):
    """Return an angle-only pendulum measurement."""
    return np.array([state[0]])


def _simulate_true_pendulum(initial_state, dt, num_points, L, g, damping):
    """Simulate true nonlinear pendulum motion."""
    states = np.zeros((num_points, 2))
    states[0] = initial_state

    for index in range(num_points - 1):
        states[index + 1] = _rk4_pendulum_step(states[index], dt, L, g, damping)

    return states


def _run_pendulum_ukf(t, measurements, L, g, damping, initial_estimate):
    """Run the pendulum UKF over noisy angle measurements."""
    measurement_noise_std = np.radians(2.0)
    dt = t[1] - t[0]

    ukf = UnscentedKalmanFilter(
        x0=initial_estimate,
        P0=np.diag([0.1, 1.0]),
        Q=np.diag([1e-7, 1e-5]),
        R=np.array([[measurement_noise_std**2]]),
        process_model=lambda x, sample_time: _rk4_pendulum_step(
            x,
            sample_time,
            L,
            g,
            damping,
        ),
        measurement_model=_pendulum_measurement,
        dt=dt,
        alpha=1e-3,
        beta=2.0,
        kappa=0.0,
    )

    estimates = np.zeros((len(measurements), 2))
    estimates[0] = ukf.update(measurements[0])

    for index in range(1, len(measurements)):
        estimates[index] = ukf.step(measurements[index])

    return estimates


def _draw_plots(t, true_theta, true_omega, measured_theta, estimated_states):
    """Draw nonlinear pendulum UKF state-estimation plots."""
    apply_plot_style()

    estimated_theta = estimated_states[:, 0]
    estimated_omega = estimated_states[:, 1]

    figure, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)

    axes[0].plot(t, np.degrees(true_theta), label="True angle")
    axes[0].plot(
        t,
        np.degrees(measured_theta),
        ".",
        alpha=0.25,
        label="Noisy angle measurement",
    )
    axes[0].plot(t, np.degrees(estimated_theta), "--", label="UKF estimated angle")
    format_axes(
        axes[0],
        title="Nonlinear Pendulum UKF: Angle Estimate",
        ylabel="Angle (deg)",
    )

    axes[1].plot(t, true_omega, label="True angular velocity")
    axes[1].plot(t, estimated_omega, "--", label="UKF estimated angular velocity")
    format_axes(
        axes[1],
        title="Hidden Angular Velocity Estimate",
        ylabel="Angular velocity (rad/s)",
    )

    axes[2].plot(
        t,
        np.degrees(true_theta - estimated_theta),
        label="Angle error",
    )
    axes[2].plot(t, true_omega - estimated_omega, label="Angular velocity error")
    format_axes(
        axes[2],
        title="Estimation Errors",
        xlabel="Time (s)",
        ylabel="Error",
    )
    place_legends_outside(axes, location="right")

    figure.tight_layout()
    return figure


def _plot_response(t, true_theta, true_omega, measured_theta, estimated_states):
    """Save and display the pendulum UKF plots."""
    output_path = PROJECT_ROOT / "examples" / "ukf_pendulum.png"

    figure = _draw_plots(t, true_theta, true_omega, measured_theta, estimated_states)
    save_figure(figure, output_path)
    print(f"Plot saved to: {output_path}")
    plt.show()


def main():
    """Estimate pendulum angle and angular velocity from noisy angle data."""
    L = 1.0
    g = 9.81
    damping = 0.08
    theta0_degrees = 25.0
    theta0 = np.radians(theta0_degrees)
    omega0 = 0.0
    t_final = 10.0
    dt = 0.01
    num_points = int(t_final / dt) + 1
    rng = np.random.default_rng(4)
    measurement_noise_std = np.radians(2.0)
    initial_estimate = np.array([np.radians(15.0), 0.0])

    t = np.linspace(0.0, t_final, num_points)
    true_states = _simulate_true_pendulum(
        np.array([theta0, omega0]),
        dt,
        num_points,
        L,
        g,
        damping,
    )
    true_theta = true_states[:, 0]
    true_omega = true_states[:, 1]
    measured_theta = true_theta + rng.normal(0.0, measurement_noise_std, len(t))
    estimated_states = _run_pendulum_ukf(
        t,
        measured_theta,
        L,
        g,
        damping,
        initial_estimate,
    )

    estimated_theta = estimated_states[:, 0]
    estimated_omega = estimated_states[:, 1]
    angle_error = true_theta - estimated_theta
    omega_error = true_omega - estimated_omega

    print("Nonlinear Pendulum Unscented Kalman Filter:")
    print(f"Pendulum length: {L:.3f} m")
    print(f"Damping coefficient: {damping:.3f} 1/s")
    print(f"Initial true angle: {theta0_degrees:.3f} degrees")
    print(f"Angle measurement noise std: {np.degrees(measurement_noise_std):.3f} degrees")
    print(f"Final true angle: {np.degrees(true_theta[-1]):.3f} degrees")
    print(f"Final estimated angle: {np.degrees(estimated_theta[-1]):.3f} degrees")
    print(f"Final angle error: {np.degrees(angle_error[-1]):.3f} degrees")
    print(f"Final angular velocity error: {omega_error[-1]:.3f} rad/s")
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
        "The UKF estimates both measured angle and hidden angular velocity "
        "without requiring Jacobians."
    )

    _plot_response(t, true_theta, true_omega, measured_theta, estimated_states)


if __name__ == "__main__":
    main()
