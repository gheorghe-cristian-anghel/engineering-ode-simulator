"""Qualitative, potential-flow-inspired airfoil visualization helpers."""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path

from visualization import plot_style as ps

_NACA_PARAMETERS = {
    "NACA 0012": (0.0, 0.0, 0.12),
    "NACA 2412": (0.02, 0.4, 0.12),
}


@dataclass(frozen=True)
class AirfoilFlowMetrics:
    """Summary values displayed with the qualitative flow visualization."""

    max_speed: float
    min_speed: float
    wake_strength: float
    grid_shape: tuple[int, int]


def _normalize_airfoil_code(airfoil_code: str) -> str:
    normalized = airfoil_code.upper().strip()
    if not normalized.startswith("NACA"):
        normalized = f"NACA {normalized}"
    if normalized not in _NACA_PARAMETERS:
        supported_codes = ", ".join(_NACA_PARAMETERS)
        raise ValueError(
            f"Unsupported airfoil '{airfoil_code}'. Use {supported_codes}."
        )
    return normalized


def naca_airfoil_outline(
    airfoil_code: str = "NACA 2412",
    point_count: int = 180,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a closed, unit-chord NACA four-digit airfoil outline."""
    if point_count < 20:
        raise ValueError("point_count must be at least 20.")

    camber, camber_location, thickness = _NACA_PARAMETERS[
        _normalize_airfoil_code(airfoil_code)
    ]
    x = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, point_count)))
    thickness_distribution = (
        5.0
        * thickness
        * (
            0.2969 * np.sqrt(x)
            - 0.1260 * x
            - 0.3516 * x**2
            + 0.2843 * x**3
            - 0.1015 * x**4
        )
    )

    if camber == 0.0:
        camber_line = np.zeros_like(x)
        camber_slope = np.zeros_like(x)
    else:
        leading_segment = x < camber_location
        camber_line = np.where(
            leading_segment,
            camber / camber_location**2 * (2.0 * camber_location * x - x**2),
            camber
            / (1.0 - camber_location) ** 2
            * ((1.0 - 2.0 * camber_location) + 2.0 * camber_location * x - x**2),
        )
        camber_slope = np.where(
            leading_segment,
            2.0 * camber / camber_location**2 * (camber_location - x),
            2.0 * camber / (1.0 - camber_location) ** 2 * (camber_location - x),
        )

    angle = np.arctan(camber_slope)
    upper_x = x - thickness_distribution * np.sin(angle)
    upper_y = camber_line + thickness_distribution * np.cos(angle)
    lower_x = x + thickness_distribution * np.sin(angle)
    lower_y = camber_line - thickness_distribution * np.cos(angle)

    outline_x = np.concatenate((upper_x[::-1], lower_x[1:], upper_x[-1:]))
    outline_y = np.concatenate((upper_y[::-1], lower_y[1:], upper_y[-1:]))
    return outline_x, outline_y


def _rotate_points(
    x: np.ndarray,
    y: np.ndarray,
    angle_rad: float,
    inverse: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate coordinates around the quarter-chord reference point."""
    cosine = np.cos(angle_rad)
    sine = np.sin(angle_rad)
    shifted_x = x - 0.25
    if inverse:
        return cosine * shifted_x + sine * y + 0.25, -sine * shifted_x + cosine * y
    return cosine * shifted_x - sine * y + 0.25, sine * shifted_x + cosine * y


def _validate_inputs(
    free_stream_velocity: float,
    grid_shape: tuple[int, int],
    streamline_density: float,
) -> None:
    if free_stream_velocity <= 0.0:
        raise ValueError("free_stream_velocity must be positive.")
    rows, columns = grid_shape
    if rows < 40 or columns < 80:
        raise ValueError("grid_shape must be at least 40 rows by 80 columns.")
    if rows > 160 or columns > 320:
        raise ValueError(
            "grid_shape exceeds the supported interactive limit of 160 by 320."
        )
    if not 0.5 <= streamline_density <= 2.0:
        raise ValueError("streamline_density must be between 0.5 and 2.0.")


def plot_airfoil_flow_field(
    airfoil_code: str = "NACA 2412",
    angle_of_attack_deg: float = 5.0,
    free_stream_velocity: float = 30.0,
    circulation_strength: float = 0.0,
    wake_strength: float = 0.6,
    grid_shape: tuple[int, int] = (80, 160),
    streamline_density: float = 1.1,
) -> tuple[plt.Figure, AirfoilFlowMetrics]:
    """Plot a qualitative 2D airfoil field with potential-flow-inspired effects.

    This educational visualization uses a synthetic circulation and wake term. It
    is not a Navier-Stokes solution and must not be used for force prediction.
    """
    airfoil_code = _normalize_airfoil_code(airfoil_code)
    _validate_inputs(free_stream_velocity, grid_shape, streamline_density)

    row_count, column_count = grid_shape
    x_values = np.linspace(-1.2, 3.0, column_count)
    y_values = np.linspace(-1.25, 1.25, row_count)
    grid_x, grid_y = np.meshgrid(x_values, y_values)
    attack_angle = np.deg2rad(angle_of_attack_deg)
    local_x, local_y = _rotate_points(grid_x, grid_y, attack_angle, inverse=True)

    center_x = 0.28
    center_y = 0.0
    relative_x = local_x - center_x
    relative_y = local_y - center_y
    radius_squared = np.maximum(relative_x**2 + relative_y**2, 0.035**2)
    doublet_scale = 0.055 * free_stream_velocity
    vortex_scale = 0.07 * free_stream_velocity * circulation_strength

    local_u = free_stream_velocity * np.cos(attack_angle)
    local_v = -free_stream_velocity * np.sin(attack_angle)
    local_u += doublet_scale * (relative_x**2 - relative_y**2) / radius_squared**2
    local_v += 2.0 * doublet_scale * relative_x * relative_y / radius_squared**2
    local_u -= vortex_scale * relative_y / (2.0 * np.pi * radius_squared)
    local_v += vortex_scale * relative_x / (2.0 * np.pi * radius_squared)

    downstream_distance = np.maximum(local_x - 0.85, 0.0)
    wake_width = 0.12 + 0.18 * downstream_distance
    wake_profile = np.exp(-0.5 * (local_y / wake_width) ** 2) * np.exp(
        -0.55 * downstream_distance
    )
    wake_deficit = 0.32 * wake_strength * free_stream_velocity * wake_profile
    local_u -= wake_deficit
    local_v += (
        0.08 * wake_strength * free_stream_velocity * np.sign(local_y) * wake_profile
    )

    velocity_x, velocity_y = _rotate_points(
        local_u + 0.25,
        local_v,
        attack_angle,
    )
    velocity_x -= 0.25
    speed = np.hypot(velocity_x, velocity_y)

    outline_x, outline_y = naca_airfoil_outline(airfoil_code)
    airfoil_x, airfoil_y = _rotate_points(outline_x, outline_y, attack_angle)
    airfoil_path = Path(np.column_stack((airfoil_x, airfoil_y)))
    inside_airfoil = airfoil_path.contains_points(
        np.column_stack((grid_x.ravel(), grid_y.ravel()))
    ).reshape(grid_x.shape)
    masked_speed = np.ma.masked_where(inside_airfoil, speed)
    masked_u = np.ma.masked_where(inside_airfoil, velocity_x)
    masked_v = np.ma.masked_where(inside_airfoil, velocity_y)

    figure, axis = ps.create_streamlit_subplots(width=12, height=6.5)
    speed_levels = np.linspace(float(masked_speed.min()), float(masked_speed.max()), 32)
    speed_contours = axis.contourf(
        grid_x,
        grid_y,
        masked_speed,
        levels=speed_levels,
        cmap="viridis",
        extend="both",
    )
    axis.streamplot(
        x_values,
        y_values,
        masked_u,
        masked_v,
        density=streamline_density,
        color="white",
        linewidth=0.75,
        arrowsize=0.7,
    )
    axis.fill(airfoil_x, airfoil_y, color="#202124", zorder=4, label=airfoil_code)
    ps.add_clean_colorbar(figure, speed_contours, axis, label="Speed (arbitrary units)")
    ps.format_heatmap_axes(
        axis,
        title="Qualitative airfoil flow field",
        xlabel="Tunnel position (chord lengths)",
        ylabel="Tunnel height (chord lengths)",
    )
    axis.set_aspect("equal", adjustable="box")
    axis.legend(loc="upper right", frameon=True)

    metrics = AirfoilFlowMetrics(
        max_speed=float(masked_speed.max()),
        min_speed=float(masked_speed.min()),
        wake_strength=float(np.max(wake_deficit)),
        grid_shape=grid_shape,
    )
    return figure, metrics
