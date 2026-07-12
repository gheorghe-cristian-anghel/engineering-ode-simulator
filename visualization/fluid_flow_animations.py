"""Reusable 2D vector-field particle animations for qualitative simulations."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.collections import LineCollection
from matplotlib.path import Path

from visualization import plot_style as ps
from visualization.airfoil_flow import AirfoilFlowResult

MAX_PARTICLES = 800
MAX_FRAMES = 120
MAX_TRAIL_LENGTH = 12


@dataclass(frozen=True)
class VectorField2D:
    """A reusable 2D velocity field with optional solid geometry and mask."""

    grid_x: np.ndarray
    grid_y: np.ndarray
    velocity_x: np.ndarray
    velocity_y: np.ndarray
    speed: np.ndarray
    mask: np.ndarray | None = None
    geometry_x: np.ndarray | None = None
    geometry_y: np.ndarray | None = None
    geometry_label: str | None = None
    title: str = "Qualitative vector-field visualization"


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


def _field_limits(field: VectorField2D) -> tuple[float, float, float, float]:
    """Return fixed animation limits from a vector field."""
    return (
        float(field.grid_x.min()),
        float(field.grid_x.max()),
        float(field.grid_y.min()),
        float(field.grid_y.max()),
    )


def _airfoil_base_field(flow_result: AirfoilFlowResult) -> VectorField2D:
    """Adapt the unchanged static airfoil field to the generic renderer."""
    return VectorField2D(
        grid_x=flow_result.grid_x,
        grid_y=flow_result.grid_y,
        velocity_x=flow_result.velocity_x,
        velocity_y=flow_result.velocity_y,
        speed=flow_result.speed,
        mask=flow_result.inside_airfoil,
        geometry_x=flow_result.airfoil_x,
        geometry_y=flow_result.airfoil_y,
        geometry_label="Airfoil",
        title="Airfoil — 2D wind tunnel",
    )


def make_airfoil_animation_field(
    flow_result: AirfoilFlowResult,
    vortex_emphasis: float = 1.2,
) -> VectorField2D:
    """Return an animation-only airfoil field with an optional visual vortex.

    The added vortex makes the educational wake more legible and is deliberately
    separate from the static flow field and all numerical solvers.
    """
    emphasis = float(vortex_emphasis)
    if not np.isfinite(emphasis) or not 0.0 <= emphasis <= 3.0:
        raise ValueError("vortex_emphasis must be between 0.0 and 3.0.")

    field = _airfoil_base_field(flow_result)
    if emphasis == 0.0:
        return field

    trailing_index = int(np.argmax(flow_result.airfoil_x))
    center_x = float(flow_result.airfoil_x[trailing_index] + 0.42)
    center_y = float(flow_result.airfoil_y[trailing_index] - 0.30)
    dx = field.grid_x - center_x
    dy = field.grid_y - center_y
    radius = np.hypot(dx, dy)
    core_radius = 0.18
    safe_radius = np.maximum(radius, 1e-9)
    reference_speed = max(flow_result.metrics.max_speed, 1.0)
    tangential_speed = (
        2.0
        * reference_speed
        * emphasis
        * core_radius
        * radius
        / (radius**2 + core_radius**2)
    )
    vortex_u = tangential_speed * dy / safe_radius
    vortex_v = -tangential_speed * dx / safe_radius
    velocity_x = field.velocity_x + vortex_u
    velocity_y = field.velocity_y + vortex_v
    return VectorField2D(
        grid_x=field.grid_x,
        grid_y=field.grid_y,
        velocity_x=velocity_x,
        velocity_y=velocity_y,
        speed=np.hypot(velocity_x, velocity_y),
        mask=field.mask,
        geometry_x=field.geometry_x,
        geometry_y=field.geometry_y,
        geometry_label=field.geometry_label,
        title=field.title,
    )


def _inlet_positions(
    field: VectorField2D,
    count: int,
    random_generator: np.random.Generator,
) -> np.ndarray:
    """Seed recycled particles near the fixed left-hand field boundary."""
    x_min, x_max, y_min, y_max = _field_limits(field)
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


def initialize_vector_field_particles(
    field: VectorField2D,
    num_particles: int = 300,
    seed: int = 42,
) -> np.ndarray:
    """Return reproducible full-field particle locations for any 2D field."""
    num_particles = _validate_positive_integer(
        num_particles,
        "num_particles",
        MAX_PARTICLES,
    )
    random_generator = np.random.default_rng(seed)
    x_min, x_max, y_min, y_max = _field_limits(field)
    positions = np.column_stack(
        (
            random_generator.uniform(x_min, x_max, num_particles),
            random_generator.uniform(y_min, y_max, num_particles),
        )
    )
    obstacle_path = _geometry_path(field)
    if obstacle_path is not None:
        inside_geometry = obstacle_path.contains_points(positions)
        if np.any(inside_geometry):
            positions[inside_geometry] = _inlet_positions(
                field,
                int(np.count_nonzero(inside_geometry)),
                random_generator,
            )
    return positions


def initialize_airfoil_particles(
    flow_result: AirfoilFlowResult,
    num_particles: int = 300,
    seed: int = 42,
) -> np.ndarray:
    """Return reproducible inlet particle locations for the static airfoil field."""
    return initialize_vector_field_particles(
        _airfoil_base_field(flow_result),
        num_particles=num_particles,
        seed=seed,
    )


def _sample_velocity(
    field: VectorField2D,
    positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Bilinearly sample a regular vector field at particle positions."""
    x_min, x_max, y_min, y_max = _field_limits(field)
    row_count, column_count = field.speed.shape
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

    def interpolate(values: np.ndarray) -> np.ndarray:
        lower = (1.0 - column_fraction) * values[lower_row, left_column]
        lower += column_fraction * values[lower_row, right_column]
        upper = (1.0 - column_fraction) * values[upper_row, left_column]
        upper += column_fraction * values[upper_row, right_column]
        return (1.0 - row_fraction) * lower + row_fraction * upper

    return interpolate(field.velocity_x), interpolate(field.velocity_y)


def _geometry_path(field: VectorField2D) -> Path | None:
    """Return a particle-recycling obstacle path when geometry is available."""
    if field.geometry_x is None or field.geometry_y is None:
        return None
    return Path(np.column_stack((field.geometry_x, field.geometry_y)))


def advect_vector_field_particles(
    field: VectorField2D,
    positions: np.ndarray,
    speed_scale: float = 1.0,
    steps: int = 1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance and recycle particles through a frozen qualitative vector field."""
    scale = _validate_speed_scale(speed_scale)
    steps = _validate_positive_integer(steps, "steps", maximum=12)
    particle_positions = np.asarray(positions, dtype=float)
    if particle_positions.ndim != 2 or particle_positions.shape[1] != 2:
        raise ValueError("positions must have shape (num_particles, 2).")
    if not np.all(np.isfinite(particle_positions)):
        raise ValueError("positions must contain only finite values.")

    updated_positions = particle_positions.copy()
    random_generator = np.random.default_rng(seed)
    obstacle_path = _geometry_path(field)
    x_min, x_max, y_min, y_max = _field_limits(field)
    time_step = 0.16 / max(float(_masked_speed(field).max()), 1.0)

    for _ in range(steps):
        velocity_x, velocity_y = _sample_velocity(field, updated_positions)
        updated_positions += (
            scale * time_step * np.column_stack((velocity_x, velocity_y))
        )
        outside_field = (
            (updated_positions[:, 0] < x_min)
            | (updated_positions[:, 0] > x_max)
            | (updated_positions[:, 1] < y_min)
            | (updated_positions[:, 1] > y_max)
        )
        inside_geometry = (
            obstacle_path.contains_points(updated_positions)
            if obstacle_path is not None
            else np.zeros(len(updated_positions), dtype=bool)
        )
        recycle = outside_field | inside_geometry
        if np.any(recycle):
            updated_positions[recycle] = _inlet_positions(
                field,
                int(np.count_nonzero(recycle)),
                random_generator,
            )

    velocity_x, velocity_y = _sample_velocity(field, updated_positions)
    return updated_positions, np.hypot(velocity_x, velocity_y)


def advect_airfoil_particles(
    flow_result: AirfoilFlowResult,
    positions: np.ndarray,
    speed_scale: float = 1.0,
    steps: int = 1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance particles through the unchanged static airfoil field."""
    return advect_vector_field_particles(
        _airfoil_base_field(flow_result),
        positions,
        speed_scale=speed_scale,
        steps=steps,
        seed=seed,
    )


def _masked_speed(field: VectorField2D) -> np.ma.MaskedArray:
    """Return the speed field with optional solid geometry masked out."""
    if field.mask is None:
        return np.ma.asarray(field.speed)
    return np.ma.masked_where(field.mask, field.speed)


def _wind_tunnel_style(axis, field: VectorField2D) -> None:
    """Apply the compact dark visual treatment used by the reference style."""
    axis.set_facecolor("#06162b")
    axis.set_title(field.title, color="#e8f6ff", fontsize=12, pad=8)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color("#2a5369")


def _trail_segments(trails: np.ndarray) -> np.ndarray:
    """Return short particle streak segments in age order."""
    return np.stack((trails[:-1], trails[1:]), axis=2).reshape(-1, 2, 2)


def update_particle_trails(
    trails: np.ndarray,
    positions: np.ndarray,
    previous_positions: np.ndarray,
    recycle_distance: float = 0.5,
) -> np.ndarray:
    """Update trails and clear history for particles recycled across the field."""
    updated_trails = trails.copy()
    updated_trails[1:] = updated_trails[:-1]
    updated_trails[0] = positions
    recycled = (
        np.hypot(
            positions[:, 0] - previous_positions[:, 0],
            positions[:, 1] - previous_positions[:, 1],
        )
        > recycle_distance
    )
    if np.any(recycled):
        updated_trails[:, recycled] = positions[recycled]
    return updated_trails


def create_vector_field_particle_animation(
    field: VectorField2D,
    num_particles: int = 300,
    num_frames: int = 60,
    frame_skip: int = 1,
    speed_scale: float = 1.0,
    trail_length: int = 8,
    visual_style: str = "wind_tunnel_glow",
) -> animation.FuncAnimation:
    """Build a bounded, reusable 2D particle animation for a vector field."""
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
    if visual_style not in {"wind_tunnel_glow", "scientific"}:
        raise ValueError("visual_style must be 'wind_tunnel_glow' or 'scientific'.")

    figure, axis = ps.create_streamlit_subplots(width=10, height=5.2)
    figure.set_dpi(80)
    figure.set_facecolor("#06162b")
    x_min, x_max, y_min, y_max = _field_limits(field)
    speed_field = _masked_speed(field)
    display_max = max(float(np.percentile(speed_field.compressed(), 99)), 1.0)
    cmap = "turbo" if visual_style == "wind_tunnel_glow" else "viridis"
    speed_image = axis.imshow(
        speed_field,
        extent=(x_min, x_max, y_min, y_max),
        origin="lower",
        interpolation="bilinear",
        cmap=cmap,
        vmin=0.0,
        vmax=display_max,
        aspect="auto",
        zorder=0,
    )
    if field.geometry_x is not None and field.geometry_y is not None:
        axis.fill(
            field.geometry_x,
            field.geometry_y,
            color="#101720",
            edgecolor="#e8f6ff",
            linewidth=1.25,
            zorder=5,
        )
    if visual_style == "scientific":
        ps.add_clean_colorbar(
            figure, speed_image, axis, label="Speed (arbitrary units)"
        )
        ps.format_heatmap_axes(
            axis,
            title=field.title,
            xlabel="Field x",
            ylabel="Field y",
        )
    else:
        _wind_tunnel_style(axis, field)
        if field.geometry_label is not None:
            axis.text(
                0.015,
                0.965,
                field.geometry_label,
                transform=axis.transAxes,
                va="top",
                color="#8ff5ff",
                fontsize=9,
                bbox={
                    "boxstyle": "square,pad=0.35",
                    "facecolor": "#06101c",
                    "alpha": 0.9,
                },
            )
    axis.set_xlim(x_min, x_max)
    axis.set_ylim(y_min, y_max)
    axis.set_aspect("equal", adjustable="box")

    positions = initialize_vector_field_particles(field, num_particles=num_particles)
    trails = np.repeat(positions[np.newaxis, :, :], trail_length, axis=0)
    trail_alphas = np.linspace(0.62, 0.06, trail_length - 1)
    trail_colors = np.repeat(
        np.column_stack(
            (
                np.full(trail_length - 1, 0.42),
                np.full(trail_length - 1, 0.96),
                np.full(trail_length - 1, 1.0),
                trail_alphas,
            )
        ),
        num_particles,
        axis=0,
    )
    trail_artist = LineCollection(
        _trail_segments(trails),
        colors=trail_colors,
        linewidths=0.7,
        capstyle="round",
        zorder=3,
    )
    axis.add_collection(trail_artist)
    particle_artist = axis.scatter(
        positions[:, 0],
        positions[:, 1],
        s=8,
        color="#ddffff",
        alpha=0.9,
        linewidths=0.0,
        zorder=4,
    )

    def initialize() -> tuple[object, ...]:
        """Initialize artists without encoding animation frames."""
        trail_artist.set_segments(_trail_segments(trails))
        particle_artist.set_offsets(positions)
        return trail_artist, particle_artist

    def update(frame_number: int) -> tuple[object, ...]:
        """Advance luminous particle streaks through the frozen vector field."""
        nonlocal positions
        previous_positions = positions
        positions, _ = advect_vector_field_particles(
            field,
            positions,
            speed_scale=speed_scale,
            steps=frame_skip,
            seed=10_000 + frame_number,
        )
        trails[:] = update_particle_trails(trails, positions, previous_positions)
        trail_artist.set_segments(_trail_segments(trails))
        particle_artist.set_offsets(positions)
        return trail_artist, particle_artist

    return animation.FuncAnimation(
        figure,
        update,
        init_func=initialize,
        frames=num_frames,
        interval=max(35, 105 // frame_skip),
        blit=True,
        repeat=True,
    )


def create_airfoil_particle_animation(
    flow_result: AirfoilFlowResult,
    num_particles: int = 300,
    num_frames: int = 60,
    frame_skip: int = 1,
    speed_scale: float = 1.0,
    trail_length: int = 8,
    vortex_emphasis: float = 1.2,
    visual_style: str = "wind_tunnel_glow",
) -> animation.FuncAnimation:
    """Build a reference-style qualitative airfoil particle animation."""
    return create_vector_field_particle_animation(
        make_airfoil_animation_field(flow_result, vortex_emphasis=vortex_emphasis),
        num_particles=num_particles,
        num_frames=num_frames,
        frame_skip=frame_skip,
        speed_scale=speed_scale,
        trail_length=trail_length,
        visual_style=visual_style,
    )


def animate_airfoil_particles(
    flow_result: AirfoilFlowResult,
    num_particles: int = 300,
    num_frames: int = 60,
    frame_skip: int = 1,
    speed_scale: float = 1.0,
    trail_length: int = 8,
    vortex_emphasis: float = 1.2,
    visual_style: str = "wind_tunnel_glow",
) -> str:
    """Return self-contained HTML for a bounded qualitative airfoil animation."""
    particle_animation = create_airfoil_particle_animation(
        flow_result,
        num_particles=num_particles,
        num_frames=num_frames,
        frame_skip=frame_skip,
        speed_scale=speed_scale,
        trail_length=trail_length,
        vortex_emphasis=vortex_emphasis,
        visual_style=visual_style,
    )
    try:
        return particle_animation.to_jshtml(
            fps=max(4, 18 // frame_skip),
            embed_frames=True,
            default_mode="loop",
        )
    finally:
        plt.close(particle_animation._fig)
