"""Matplotlib animation helper for inverted pendulum trajectories."""

from pathlib import Path

try:
    from _tkinter import TclError
except ImportError:
    TclError = RuntimeError

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle


def _validate_trajectory(t, states):
    """Return validated time and state arrays for cart-pole animation."""
    time = np.asarray(t, dtype=float)
    state_array = np.asarray(states, dtype=float)

    if time.ndim != 1:
        raise ValueError("t must be a one-dimensional array")

    if len(time) == 0:
        raise ValueError("t must not be empty")

    if state_array.ndim != 2 or state_array.shape[1] != 4:
        raise ValueError("states must have shape (n_samples, 4)")

    if len(time) != state_array.shape[0]:
        raise ValueError("t and states must have the same number of samples")

    if not np.all(np.isfinite(time)):
        raise ValueError("t must contain only finite values")

    if not np.all(np.isfinite(state_array)):
        raise ValueError("states must contain only finite values")

    return time, state_array


def _validate_positive(value, name):
    """Return a validated positive float."""
    value = float(value)

    if value <= 0:
        raise ValueError(f"{name} must be positive")

    return value


def _pendulum_coordinates(states, l, cart_height):
    """Return cart pivot and pendulum mass coordinates for upright-angle states."""
    cart_x = states[:, 0]
    theta = states[:, 2]
    pivot_y = cart_height / 2.0
    mass_x = cart_x + l * np.sin(theta)
    mass_y = pivot_y + l * np.cos(theta)

    return cart_x, np.full_like(cart_x, pivot_y), mass_x, mass_y


def _axis_limits(cart_x, mass_x, mass_y, l, cart_width, cart_height):
    """Return adaptive axis limits for the full cart-pole trajectory."""
    horizontal_margin = cart_width + 0.2 * l
    vertical_margin = 0.2 * l

    x_min = min(np.min(cart_x - cart_width / 2.0), np.min(mass_x))
    x_max = max(np.max(cart_x + cart_width / 2.0), np.max(mass_x))
    y_min = min(-cart_height / 2.0, np.min(mass_y))
    y_max = max(cart_height / 2.0 + l, np.max(mass_y))

    return (
        (x_min - horizontal_margin, x_max + horizontal_margin),
        (y_min - vertical_margin, y_max + vertical_margin),
    )


def _save_animation(anim, save_path, interval_ms):
    """Save an animation to GIF or MP4 using available Matplotlib writers."""
    output_path = Path(save_path)
    suffix = output_path.suffix.lower()
    fps = max(1, int(round(1000.0 / interval_ms)))

    if suffix == ".gif":
        if not animation.writers.is_available("pillow"):
            raise RuntimeError(
                "GIF saving requires the Matplotlib Pillow writer. "
                "Install Pillow or choose an interactive animation instead."
            )
        writer = animation.PillowWriter(fps=fps)
    elif suffix == ".mp4":
        if not animation.writers.is_available("ffmpeg"):
            raise RuntimeError(
                "MP4 saving requires FFmpeg. Install FFmpeg and make sure it "
                "is available on PATH, or save as GIF instead."
            )
        writer = animation.FFMpegWriter(fps=fps)
    else:
        raise ValueError("save_path must end with .gif or .mp4")

    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    anim.save(output_path, writer=writer)


def _validate_save_path_extension(save_path):
    """Validate optional animation output extension before creating a figure."""
    if save_path is None:
        return

    suffix = Path(save_path).suffix.lower()

    if suffix not in {".gif", ".mp4"}:
        raise ValueError("save_path must end with .gif or .mp4")


def _create_figure_and_axis(show, save_path):
    """Create a Matplotlib figure, falling back when Tk is unavailable."""
    try:
        return plt.subplots(), show
    except TclError:
        plt.switch_backend("Agg")
        figure_and_axis = plt.subplots()

        if show and save_path is None:
            print("Interactive Matplotlib window is unavailable in this environment.")
            print("Run with --save to create a GIF or MP4 animation file.")
            show = False

        return figure_and_axis, show


def animate_inverted_pendulum(
    t,
    states,
    l=0.5,
    title="Inverted Pendulum Animation",
    interval_ms=30,
    cart_width=0.35,
    cart_height=0.18,
    save_path=None,
    show=True,
):
    """Animate an inverted pendulum / cart-pole trajectory.

    The state convention is ``[cart_position, cart_velocity, pendulum_angle,
    pendulum_angular_velocity]``. The angle is measured from upright, matching
    ``models.inverted_pendulum``: ``theta = 0`` points straight up.

    Parameters
    ----------
    t : array-like
        Time samples in seconds.
    states : array-like
        State array with shape ``(n_samples, 4)``.
    l : float, optional
        Pendulum length in meters.
    title : str, optional
        Animation title.
    interval_ms : float, optional
        Time between animation frames in milliseconds.
    cart_width, cart_height : float, optional
        Cart drawing dimensions in meters.
    save_path : path-like, optional
        Optional ``.gif`` or ``.mp4`` output path.
    show : bool, optional
        If True, display the animation interactively with ``plt.show()``.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        The created animation object.
    """
    time, state_array = _validate_trajectory(t, states)
    l = _validate_positive(l, "l")
    interval_ms = _validate_positive(interval_ms, "interval_ms")
    cart_width = _validate_positive(cart_width, "cart_width")
    cart_height = _validate_positive(cart_height, "cart_height")
    _validate_save_path_extension(save_path)

    cart_x, pivot_y, mass_x, mass_y = _pendulum_coordinates(
        state_array,
        l,
        cart_height,
    )
    x_limits, y_limits = _axis_limits(
        cart_x,
        mass_x,
        mass_y,
        l,
        cart_width,
        cart_height,
    )

    (figure, axis), show = _create_figure_and_axis(show, save_path)
    axis.set_title(title)
    axis.set_xlabel("Cart position (m)")
    axis.set_ylabel("Height (m)")
    axis.set_xlim(*x_limits)
    axis.set_ylim(*y_limits)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True)

    track_y = -cart_height / 2.0
    axis.axhline(track_y, color="0.4", linewidth=1.0)

    cart = Rectangle(
        (cart_x[0] - cart_width / 2.0, -cart_height / 2.0),
        cart_width,
        cart_height,
        facecolor="tab:blue",
        edgecolor="black",
    )
    axis.add_patch(cart)

    rod_line, = axis.plot([], [], color="black", linewidth=2.0)
    mass_marker, = axis.plot(
        [],
        [],
        marker="o",
        markersize=10,
        color="tab:orange",
    )
    pivot_marker, = axis.plot(
        [],
        [],
        marker="o",
        markersize=4,
        color="black",
    )
    time_text = axis.text(
        0.02,
        0.95,
        "",
        transform=axis.transAxes,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )

    def update(frame_index):
        """Update cart, rod, mass, and time label for one frame."""
        x = cart_x[frame_index]
        y = pivot_y[frame_index]
        tip_x = mass_x[frame_index]
        tip_y = mass_y[frame_index]

        cart.set_xy((x - cart_width / 2.0, -cart_height / 2.0))
        rod_line.set_data([x, tip_x], [y, tip_y])
        mass_marker.set_data([tip_x], [tip_y])
        pivot_marker.set_data([x], [y])
        time_text.set_text(f"t = {time[frame_index]:.2f} s")

        return cart, rod_line, mass_marker, pivot_marker, time_text

    anim = animation.FuncAnimation(
        figure,
        update,
        frames=len(time),
        interval=interval_ms,
        blit=True,
        repeat=True,
    )

    if save_path is not None:
        _save_animation(anim, save_path, interval_ms)

    if show:
        plt.show()

    return anim
