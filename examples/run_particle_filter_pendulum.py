"""Estimate nonlinear pendulum state with a bootstrap Particle Filter."""

import sys
from pathlib import Path
from _tkinter import TclError

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.particle_filter import ParticleFilter


def _pendulum_derivative(states, L, g, damping):
    """Return nonlinear damped pendulum state derivatives."""
    states = np.asarray(states, dtype=float)
    theta = states[..., 0]
    omega = states[..., 1]

    return np.stack(
        [
            omega,
            -(g / L) * np.sin(theta) - damping * omega,
        ],
        axis=-1,
    )


def _rk4_pendulum_step(states, dt, L, g, damping):
    """Advance pendulum states by one RK4 integration step."""
    k1 = _pendulum_derivative(states, L, g, damping)
    k2 = _pendulum_derivative(states + 0.5 * dt * k1, L, g, damping)
    k3 = _pendulum_derivative(states + 0.5 * dt * k2, L, g, damping)
    k4 = _pendulum_derivative(states + dt * k3, L, g, damping)

    return states + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _simulate_true_pendulum(initial_state, dt, num_points, L, g, damping):
    """Simulate true nonlinear pendulum motion."""
    states = np.zeros((num_points, 2))
    states[0] = initial_state

    for index in range(num_points - 1):
        states[index + 1] = _rk4_pendulum_step(states[index], dt, L, g, damping)

    return states


def _run_pendulum_particle_filter(
    t,
    measurements,
    L,
    g,
    damping,
    rng,
    num_particles,
):
    """Run the pendulum particle filter over noisy angle measurements."""
    measurement_noise_std = np.radians(2.0)
    process_noise_std = np.array([np.radians(0.04), 0.015])
    dt = t[1] - t[0]
    particles = np.column_stack(
        [
            rng.normal(np.radians(15.0), np.radians(10.0), num_particles),
            rng.normal(0.0, 0.5, num_particles),
        ]
    )

    def process_model(particle_states, sample_time):
        return _rk4_pendulum_step(particle_states, sample_time, L, g, damping)

    def process_noise_sampler(n_particles, state_dim, random_generator):
        return random_generator.normal(
            0.0,
            process_noise_std,
            size=(n_particles, state_dim),
        )

    def measurement_likelihood(measurement, particle_states):
        measured_angle = np.asarray(measurement, dtype=float).reshape(-1)[0]
        angle_error = measured_angle - particle_states[:, 0]
        return np.exp(-0.5 * (angle_error / measurement_noise_std) ** 2)

    particle_filter = ParticleFilter(
        particles=particles,
        process_model=process_model,
        measurement_likelihood=measurement_likelihood,
        process_noise_sampler=process_noise_sampler,
        rng=rng,
    )

    estimates = np.zeros((len(measurements), 2))
    effective_sample_sizes = np.zeros(len(measurements))
    resample_threshold = 0.5

    particle_filter.update(measurements[0])
    effective_sample_sizes[0] = particle_filter.effective_sample_size()
    if effective_sample_sizes[0] < resample_threshold * num_particles:
        particle_filter.systematic_resample()
    estimates[0] = particle_filter.estimate()

    for index in range(1, len(measurements)):
        particle_filter.predict(dt)
        particle_filter.update(measurements[index])
        effective_sample_sizes[index] = particle_filter.effective_sample_size()
        if effective_sample_sizes[index] < resample_threshold * num_particles:
            particle_filter.systematic_resample()
        estimates[index] = particle_filter.estimate()

    return estimates, effective_sample_sizes


def _draw_plots(
    t,
    true_theta,
    true_omega,
    measured_theta,
    estimated_states,
    effective_sample_sizes,
):
    """Draw nonlinear pendulum particle-filter state-estimation plots."""
    estimated_theta = estimated_states[:, 0]
    estimated_omega = estimated_states[:, 1]

    figure, axes = plt.subplots(4, 1, sharex=True)

    axes[0].plot(t, np.degrees(true_theta), label="True angle")
    axes[0].plot(
        t,
        np.degrees(measured_theta),
        ".",
        alpha=0.25,
        label="Noisy angle measurement",
    )
    axes[0].plot(t, np.degrees(estimated_theta), "--", label="PF estimated angle")
    axes[0].set_ylabel("Angle (deg)")
    axes[0].set_title("Nonlinear Pendulum Particle Filter: Angle Estimate")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(t, true_omega, label="True angular velocity")
    axes[1].plot(t, estimated_omega, "--", label="PF estimated angular velocity")
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
    axes[2].set_ylabel("Error")
    axes[2].set_title("Estimation Errors")
    axes[2].grid(True)
    axes[2].legend()

    axes[3].plot(t, effective_sample_sizes, label="Effective sample size")
    axes[3].set_xlabel("Time (s)")
    axes[3].set_ylabel("Particles")
    axes[3].set_title("Particle Diversity")
    axes[3].grid(True)
    axes[3].legend()

    figure.tight_layout()


def _plot_response(
    t,
    true_theta,
    true_omega,
    measured_theta,
    estimated_states,
    effective_sample_sizes,
):
    """Plot response, falling back gracefully if Tk is unavailable."""
    output_path = PROJECT_ROOT / "examples" / "particle_filter_pendulum.png"

    try:
        _draw_plots(
            t,
            true_theta,
            true_omega,
            measured_theta,
            estimated_states,
            effective_sample_sizes,
        )
        plt.savefig(output_path, dpi=150)
        plt.show()
    except TclError:
        plt.switch_backend("Agg")
        plt.close("all")
        _draw_plots(
            t,
            true_theta,
            true_omega,
            measured_theta,
            estimated_states,
            effective_sample_sizes,
        )
        plt.savefig(output_path, dpi=150)
        print("Interactive Matplotlib window is unavailable in this environment.")
        print(f"Plot saved to: {output_path}")


def main():
    """Estimate pendulum angle and angular velocity from noisy angle data."""
    L = 1.0
    g = 9.81
    damping = 0.08
    theta0_degrees = 25.0
    theta0 = np.radians(theta0_degrees)
    omega0 = 0.0
    t_final = 10.0
    dt = 0.02
    num_points = int(t_final / dt) + 1
    num_particles = 2000
    rng = np.random.default_rng(5)
    measurement_noise_std = np.radians(2.0)

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
    estimated_states, effective_sample_sizes = _run_pendulum_particle_filter(
        t,
        measured_theta,
        L,
        g,
        damping,
        rng,
        num_particles,
    )

    estimated_theta = estimated_states[:, 0]
    estimated_omega = estimated_states[:, 1]
    angle_error = true_theta - estimated_theta
    omega_error = true_omega - estimated_omega

    print("Nonlinear Pendulum Particle Filter:")
    print(f"Number of particles: {num_particles}")
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
    print(f"Average effective sample size: {np.mean(effective_sample_sizes):.1f}")
    print()
    print(
        "The particle filter estimates nonlinear pendulum state using many "
        "weighted hypotheses and resampling, without assuming a Gaussian "
        "state distribution."
    )

    _plot_response(
        t,
        true_theta,
        true_omega,
        measured_theta,
        estimated_states,
        effective_sample_sizes,
    )


if __name__ == "__main__":
    main()
