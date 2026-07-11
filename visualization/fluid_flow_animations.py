"""Bounded particle animations for qualitative airfoil flow fields."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.path import Path

from visualization import plot_style as ps
from visualization.airfoil_flow import AirfoilFlowResult

MAX_PARTICLES = 800
MAX_FRAMES = 120
MAX_TRAIL_LENGTH = 12


def _validate_positive_integer(value: int, name: str, maximum: int) -> int:
    """Return a bounded positive integer animation setting."""
    if isinstance(value, bool) or int(value) != value:
        raise ValueError(f"{name} must be a whole number.")
    integer_value = int(value)
    if not 1 <= integer_value <= maximum:
        raise ValueError(f"{name} must be between 1 and {maximum}.")
    return integer_value


def _validate_speed_scale(speed_scale: float) -> float:
    """Return a finite positive particle speed multiplier."""
    scale = float(speed_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("speed_scale must be positive and finite.")
    return scale


def _field_limits(flow_result: AirfoilFlowResult) -> tuple[float, float, float, float]:
    """Return fixed animation limits from the static qualitative field."""
    return (
        float(flow_result.grid_x.min()),
        float(flow_result.grid_x.max()),
        float(flow_result.grid_y.min()),
        float(flow_result.grid_y.max()),
    )


def _inlet_positions(
    flow_result: AirfoilFlowResult,
    count: int,
    random_generator: np.random.Generator,
) -> np.ndarray:
    """Seed recycled particles near the fixed wind-tunnel inlet."""
    x_min, x_max, y_min, y_max = _field_limits(flow_result)
    tunnel_width = x_max - x_min
    tunnel_height = y_max - y_min
    return np.column_stack(
        (
            random_generator.uniform(x_min, x_min + 0.12 * tunnel_width, count),
            random_generator.uniform(
                y_min + 0.04 * tunnel_height,
                y_max - 0.04 * tunnel_height,
                count,
            ),
        )
    )


def initialize_airfoil_particles(
    flow_result: AirfoilFlowResult,
    num_particles: int = 300,
    seed: int = 42,
) -> np.ndarray:
    """Return reproducible inlet particle locations for a qualitative animation."""
    num_particles = _validate_positive_integer(
        num_particles,
        "num_particles",
        MAX_PARTICLES,
    )
    return _inlet_positions(flow_result, num_particles, np.random.default_rng(seed))


def _sample_velocity(
    flow_result: AirfoilFlowResult,
    positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Bilinearly sample the static velocity field at particle positions."""
    x_min, x_max, y_min, y_max = _field_limits(flow_result)
    row_count, column_count = flow_result.speed.shape
    clipped_x = np.clip(positions[:, 0], x_min, x_max)
    clipped_y = np.clip(positions[:, 1], y_min, y_max)
    column_position = (clipped_x - x_min) / (x_max - x_min) * (column_count - 1)
    row_position = (clipped_y - y_min) / (y_max - y_min) * (row_count - 1)
    left_column = np.floor(column_position).astype(int)
    lower_row = np.floor(row_position).astype(int)
    right_column = np.minimum(left_column + 1, column_count - 1)
    upper_row = np.minimum(lower_row + 1, row_count - 1)
    column_fraction = column_position - left_column
    row_fraction = row_position - lower_row

    def interpolate(field: np.ndarray) -> np.ndarray:
        lower = (1.0 - column_fraction) * field[lower_row, left_column]
        lower += column_fraction * field[lower_row, right_column]
        upper = (1.0 - column_fraction) * field[upper_row, left_column]
        upper += column_fraction * field[upper_row, right_column]
        return (1.0 - row_fraction) * lower + row_fraction * upper

    return interpolate(flow_result.velocity_x), interpolate(flow_result.velocity_y)


def advect_airfoil_particles(
    flow_result: AirfoilFlowResult,
    positions: np.ndarray,
    speed_scale: float = 1.0,
    steps: int = 1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance particles through the static qualitative field and recycle them.

    This deliberately traces a frozen synthetic field. It is visual intuition,
    not time-accurate particle transport or CFD.
    """
    scale = _validate_speed_scale(speed_scale)
    steps = _validate_positive_integer(steps, "steps", maximum=12)
    particle_positions = np.asarray(positions, dtype=float)
    if particle_positions.ndim != 2 or particle_positions.shape[1] != 2:
        raise ValueError("positions must have shape (num_particles, 2).")
    if not np.all(np.isfinite(particle_positions)):
        raise ValueError("positions must contain only finite values.")

    updated_positions = particle_positions.copy()
    random_generator = np.random.default_rng(seed)
    airfoil_path = Path(np.column_stack((flow_result.airfoil_x, flow_result.airfoil_y)))
    x_min, x_max, y_min, y_max = _field_limits(flow_result)
    time_step = 0.035 / max(flow_result.metrics.max_speed, 1.0)

    for _ in range(steps):
        velocity_x, velocity_y = _sample_velocity(flow_result, updated_positions)
        updated_positions += (
            scale * time_step * np.column_stack((velocity_x, velocity_y))
        )
        outside_tunnel = (
            (updated_positions[:, 0] < x_min)
            | (updated_positions[:, 0] > x_max)
            | (updated_positions[:, 1] < y_min)
            | (updated_positions[:, 1] > y_max)
        )
        inside_airfoil = airfoil_path.contains_points(updated_positions)
        recycle = outside_tunnel | inside_airfoil
        if np.any(recycle):
            updated_positions[recycle] = _inlet_positions(
                flow_result,
                int(np.count_nonzero(recycle)),
                random_generator,
            )

    velocity_x, velocity_y = _sample_velocity(flow_result, updated_positions)
    return updated_positions, np.hypot(velocity_x, velocity_y)


def create_airfoil_particle_animation(
    flow_result: AirfoilFlowResult,
    num_particles: int = 300,
    num_frames: int = 60,
    frame_skip: int = 1,
    speed_scale: float = 1.0,
    trail_length: int = 8,
) -> animation.FuncAnimation:
    """Build a compact Matplotlib animation over a static qualitative field."""
    num_particles = _validate_positive_integer(
        num_particles,
        "num_particles",
        MAX_PARTICLES,
    )
    num_frames = _validate_positive_integer(num_frames, "num_frames", MAX_FRAMES)
    frame_skip = _validate_positive_integer(frame_skip, "frame_skip", maximum=6)
    trail_length = _validate_positive_integer(
        trail_length,
        "trail_length",
        MAX_TRAIL_LENGTH,
    )
    speed_scale = _validate_speed_scale(speed_scale)

    figure, axis = ps.create_streamlit_subplots(width=10, height=5.2)
    figure.set_dpi(80)
    masked_speed = np.ma.masked_where(flow_result.inside_airfoil, flow_result.speed)
    levels = np.linspace(float(masked_speed.min()), float(masked_speed.max()), 28)
    speed_contours = axis.contourf(
        flow_result.grid_x,
        flow_result.grid_y,
        masked_speed,
        levels=levels,
        cmap="viridis",
        extend="both",
    )
    axis.fill(
        flow_result.airfoil_x,
        flow_result.airfoil_y,
        color="#202124",
        zorder=5,
    )
    ps.add_clean_colorbar(
        figure,
        speed_contours,
        axis,
        label="Speed (arbitrary units)",
    )
    ps.format_heatmap_axes(
        axis,
        title="Qualitative airfoil particle visualization",
        xlabel="Tunnel position (chord lengths)",
        ylabel="Tunnel height (chord lengths)",
    )
    x_min, x_max, y_min, y_max = _field_limits(flow_result)
    axis.set_xlim(x_min, x_max)
    axis.set_ylim(y_min, y_max)
    axis.set_aspect("equal", adjustable="box")

    positions = initialize_airfoil_particles(flow_result, num_particles=num_particles)
    trails = np.repeat(positions[np.newaxis, :, :], trail_length, axis=0)
    trail_artists = [
        axis.scatter(
            positions[:, 0],
            positions[:, 1],
            s=6,
            color="#d7f2ff",
            alpha=0.08 + 0.32 * (index + 1) / trail_length,
            linewidths=0.0,
            zorder=3,
        )
        for index in range(trail_length)
    ]
    particle_artist = axis.scatter(
        positions[:, 0],
        positions[:, 1],
        c=np.zeros(num_particles),
        cmap="plasma",
        vmin=0.0,
        vmax=flow_result.metrics.max_speed,
        s=16,
        edgecolors="white",
        linewidths=0.25,
        zorder=4,
    )
    frame_label = axis.text(
        0.015,
        0.965,
        "Qualitative frozen-field particle paths",
        transform=axis.transAxes,
        va="top",
        fontsize=9,
        color="white",
        bbox={"boxstyle": "round", "facecolor": "#202124", "alpha": 0.75},
    )

    def initialize() -> tuple[object, ...]:
        """Initialize artists without encoding animation frames."""
        for index, trail_artist in enumerate(trail_artists):
            trail_artist.set_offsets(trails[index])
        particle_artist.set_offsets(positions)
        return (*trail_artists, particle_artist, frame_label)

    def update(frame_number: int) -> tuple[object, ...]:
        """Advance particle paths over the frozen qualitative velocity field."""
        nonlocal positions
        positions, speeds = advect_airfoil_particles(
            flow_result,
            positions,
            speed_scale=speed_scale,
            steps=frame_skip,
            seed=10_000 + frame_number,
        )
        trails[1:] = trails[:-1]
        trails[0] = positions
        for index, trail_artist in enumerate(trail_artists):
            trail_artist.set_offsets(trails[index])
        particle_artist.set_offsets(positions)
        particle_artist.set_array(speeds)
        return (*trail_artists, particle_artist, frame_label)

    return animation.FuncAnimation(
        figure,
        update,
        init_func=initialize,
        frames=num_frames,
        interval=max(35, 105 // frame_skip),
        blit=True,
        repeat=True,
    )


def animate_airfoil_particles(
    flow_result: AirfoilFlowResult,
    num_particles: int = 300,
    num_frames: int = 60,
    frame_skip: int = 1,
    speed_scale: float = 1.0,
    trail_length: int = 8,
) -> str:
    """Return self-contained HTML for a bounded qualitative particle animation."""
    particle_animation = create_airfoil_particle_animation(
        flow_result,
        num_particles=num_particles,
        num_frames=num_frames,
        frame_skip=frame_skip,
        speed_scale=speed_scale,
        trail_length=trail_length,
    )
    try:
        return particle_animation.to_jshtml(
            fps=max(4, 18 // frame_skip),
            embed_frames=True,
            default_mode="loop",
        )
    finally:
        plt.close(particle_animation._fig)
