"""Tests for the bounded qualitative airfoil particle animation."""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

matplotlib.use("Agg")

from visualization.airfoil_flow import compute_airfoil_flow_field
from visualization.fluid_flow_animations import (
    advect_airfoil_particles,
    create_airfoil_particle_animation,
    initialize_airfoil_particles,
)


def test_particle_advection_produces_finite_positions():
    """A bounded step through the qualitative field stays numerically finite."""
    flow_result = compute_airfoil_flow_field(grid_shape=(40, 80))
    positions = initialize_airfoil_particles(flow_result, num_particles=24, seed=7)

    updated_positions, speeds = advect_airfoil_particles(
        flow_result,
        positions,
        speed_scale=1.0,
        steps=2,
        seed=8,
    )

    assert updated_positions.shape == (24, 2)
    assert speeds.shape == (24,)
    assert np.all(np.isfinite(updated_positions))
    assert np.all(np.isfinite(speeds))


def test_particles_remain_finite_over_multiple_advection_steps():
    """Particle recycling keeps a longer qualitative run stable."""
    flow_result = compute_airfoil_flow_field(grid_shape=(40, 80))
    positions = initialize_airfoil_particles(flow_result, num_particles=24, seed=13)

    for seed in range(20, 30):
        positions, _ = advect_airfoil_particles(
            flow_result,
            positions,
            speed_scale=1.25,
            steps=3,
            seed=seed,
        )

    assert np.all(np.isfinite(positions))


def test_particle_animation_initializes_without_html_rendering():
    """The animation setup is lightweight and does not encode HTML in tests."""
    flow_result = compute_airfoil_flow_field(grid_shape=(40, 80))
    particle_animation = create_airfoil_particle_animation(
        flow_result,
        num_particles=24,
        num_frames=4,
        frame_skip=1,
        speed_scale=1.0,
        trail_length=4,
    )

    assert isinstance(particle_animation, animation.FuncAnimation)
    particle_animation._init_func()
    plt.close(particle_animation._fig)


def test_streamlit_page_keeps_particle_animation_opt_in():
    """The expensive HTML artifact is gated behind an explicit user action."""
    source = Path("streamlit_app.py").read_text(encoding="utf-8")

    assert '"Show animation"' in source
    assert '"Generate animation"' in source
    assert "animate_airfoil_particles" in source
