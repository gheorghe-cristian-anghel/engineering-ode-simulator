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
    advect_vector_field_particles,
    create_airfoil_particle_animation,
    create_vector_field_particle_animation,
    initialize_airfoil_particles,
    make_airfoil_animation_field,
    update_particle_trails,
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


def test_visual_vortex_is_opt_in_and_leaves_static_field_unchanged():
    """The reference-style vortex is animation-only, not a solver change."""
    flow_result = compute_airfoil_flow_field(grid_shape=(40, 80))
    base_field = make_airfoil_animation_field(flow_result, vortex_emphasis=0.0)
    vortex_field = make_airfoil_animation_field(flow_result, vortex_emphasis=1.5)

    assert np.allclose(base_field.velocity_x, flow_result.velocity_x)
    assert np.allclose(base_field.velocity_y, flow_result.velocity_y)
    assert not np.allclose(vortex_field.speed, base_field.speed)
    assert np.allclose(flow_result.velocity_x, base_field.velocity_x)


def test_generic_vector_field_animation_accepts_airfoil_style_field():
    """The reusable renderer accepts any compatible vector-field object."""
    flow_result = compute_airfoil_flow_field(grid_shape=(40, 80))
    animation_field = make_airfoil_animation_field(flow_result, vortex_emphasis=1.0)

    particle_animation = create_vector_field_particle_animation(
        animation_field,
        num_particles=24,
        num_frames=4,
        frame_skip=1,
        speed_scale=1.0,
        trail_length=4,
        visual_style="wind_tunnel_glow",
    )

    assert isinstance(particle_animation, animation.FuncAnimation)
    particle_animation._init_func()
    plt.close(particle_animation._fig)


def test_visual_field_particles_advance_at_a_visible_rate():
    """Masked solid-region values must not slow the visible animation."""
    flow_result = compute_airfoil_flow_field(grid_shape=(40, 80))
    animation_field = make_airfoil_animation_field(flow_result, vortex_emphasis=1.0)
    positions = initialize_airfoil_particles(flow_result, num_particles=24, seed=4)

    advanced_positions, _ = advect_vector_field_particles(
        animation_field,
        positions,
        speed_scale=1.0,
        steps=4,
        seed=5,
    )

    assert np.max(np.abs(advanced_positions - positions)) > 0.1


def test_initial_vector_field_particles_fill_the_visible_field():
    """The first animation frame should not look like a single inlet column."""
    flow_result = compute_airfoil_flow_field(grid_shape=(40, 80))
    animation_field = make_airfoil_animation_field(flow_result, vortex_emphasis=1.0)
    positions = initialize_airfoil_particles(flow_result, num_particles=80, seed=9)

    field_width = animation_field.grid_x.max() - animation_field.grid_x.min()
    assert np.ptp(positions[:, 0]) > 0.5 * field_width


def test_recycled_particles_break_their_trail_instead_of_drawing_a_teleport():
    """A recycled particle should not leave a line across the full field."""
    trails = np.array(
        [
            [[1.8, 0.0]],
            [[1.7, 0.0]],
            [[1.6, 0.0]],
        ]
    )
    previous_positions = np.array([[1.8, 0.0]])
    positions = np.array([[-1.1, 0.2]])

    updated_trails = update_particle_trails(
        trails,
        positions,
        previous_positions,
    )

    assert np.allclose(updated_trails[:, 0], positions[0])


def test_streamlit_page_keeps_particle_animation_opt_in():
    """The expensive HTML artifact is gated behind an explicit user action."""
    source = Path("streamlit_app.py").read_text(encoding="utf-8")

    assert '"Show animation"' in source
    assert '"Generate animation"' in source
    assert "animate_airfoil_particles" in source
    assert '"Vortex emphasis"' in source
    assert '"Trail length"' in source
