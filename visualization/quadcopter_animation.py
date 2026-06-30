"""3D Matplotlib animation helpers for full 6-DOF quadcopter trajectories."""

from pathlib import Path

try:
    from _tkinter import TclError
except ImportError:
    TclError = RuntimeError

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from models.quadcopter_6dof import rotation_matrix_body_to_inertial


def _validate_three_vector(values, name):
    """Return a validated three-element finite vector."""
    value_array = np.asarray(values, dtype=float)

    if value_array.ndim != 1 or len(value_array) != 3:
        raise ValueError(f"{name} must be a one-dimensional vector of length 3")

    if not np.all(np.isfinite(value_array)):
        raise ValueError(f"{name} must contain only finite values")

    return value_array


def _validate_positive(value, name):
    """Return a validated positive float."""
    value = float(value)

    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite")

    return value


def _validate_positive_integer(value, name):
    """Return a validated positive integer."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")

    integer_value = int(value)

    if integer_value != value or integer_value <= 0:
        raise ValueError(f"{name} must be a positive integer")

    return integer_value


def _validate_time_and_states(time, states):
    """Return validated time and 6-DOF state arrays."""
    time_array = np.asarray(time, dtype=float)
    state_array = np.asarray(states, dtype=float)

    if time_array.ndim != 1:
        raise ValueError("time must be a one-dimensional array")

    if len(time_array) == 0:
        raise ValueError("time must not be empty")

    if state_array.ndim != 2 or state_array.shape[1] != 12:
        raise ValueError("states must have shape (n_samples, 12)")

    if len(time_array) != state_array.shape[0]:
        raise ValueError("time and states must have the same number of samples")

    if not np.all(np.isfinite(time_array)):
        raise ValueError("time must contain only finite values")

    if not np.all(np.isfinite(state_array)):
        raise ValueError("states must contain only finite values")

    return time_array, state_array


def _validate_optional_positions(values, name, expected_length=None):
    """Return validated optional three-column position data."""
    if values is None:
        return None

    position_array = np.asarray(values, dtype=float)

    if position_array.ndim != 2 or position_array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (n_samples, 3)")

    if expected_length is not None and len(position_array) != expected_length:
        raise ValueError(f"{name} must have the same number of samples as time")

    if not np.all(np.isfinite(position_array)):
        raise ValueError(f"{name} must contain only finite values")

    return position_array


def _validate_save_path_extension(save_path):
    """Validate optional animation output extension before creating a figure."""
    if save_path is None:
        return

    suffix = Path(save_path).suffix.lower()

    if suffix not in {".gif", ".mp4"}:
        raise ValueError("save_path must end with .gif or .mp4")


def _save_animation(anim, save_path, interval):
    """Save an animation to GIF or MP4 using available Matplotlib writers."""
    output_path = Path(save_path)
    suffix = output_path.suffix.lower()
    fps = max(1, int(round(1000.0 / interval)))

    if suffix == ".gif":
        if not animation.writers.is_available("pillow"):
            raise RuntimeError(
                "GIF saving requires the Matplotlib Pillow writer. "
                "Install Pillow or run the example without saving."
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


def _create_figure_and_axis(show, save_path):
    """Create a 3D Matplotlib axis, falling back when Tk is unavailable."""
    try:
        figure = plt.figure(figsize=(8, 7))
        axis = figure.add_subplot(111, projection="3d")
        return (figure, axis), show
    except TclError:
        plt.switch_backend("Agg")
        figure = plt.figure(figsize=(8, 7))
        axis = figure.add_subplot(111, projection="3d")

        if show and save_path is None:
            print("Interactive Matplotlib window is unavailable in this environment.")
            print("Run with save_path to create a GIF or MP4 animation file.")
            show = False

        return (figure, axis), show


def _frame_indices(sample_count, frame_stride):
    """Return animation frame indices, always including the final sample."""
    indices = np.arange(0, sample_count, frame_stride)

    if indices[-1] != sample_count - 1:
        indices = np.append(indices, sample_count - 1)

    return indices


def _axis_limits(states, arm_length, reference_positions=None, waypoints=None):
    """Return fixed 3D axis limits that include trajectory and optional targets."""
    positions = states[:, 0:3]
    data_sets = [positions]

    if reference_positions is not None:
        data_sets.append(reference_positions)

    if waypoints is not None:
        data_sets.append(waypoints)

    all_points = np.vstack(data_sets)
    min_values = np.min(all_points, axis=0)
    max_values = np.max(all_points, axis=0)
    center = 0.5 * (min_values + max_values)
    span = max(np.max(max_values - min_values), 2.5 * arm_length)
    half_span = 0.5 * span + arm_length

    return tuple((center[index] - half_span, center[index] + half_span) for index in range(3))


def compute_quadcopter_body_points(
    position,
    phi,
    theta,
    psi,
    arm_length=0.25,
):
    """Return inertial-frame hub, rotor, and arm endpoints for one state.

    The body-to-inertial rotation convention matches
    ``models.quadcopter_6dof.rotation_matrix_body_to_inertial``.
    """
    center = _validate_three_vector(position, "position")
    arm_length = _validate_positive(arm_length, "arm_length")

    angles = np.asarray([phi, theta, psi], dtype=float)
    if not np.all(np.isfinite(angles)):
        raise ValueError("Euler angles must be finite")

    front_body = np.array([arm_length, 0.0, 0.0])
    back_body = np.array([-arm_length, 0.0, 0.0])
    right_body = np.array([0.0, arm_length, 0.0])
    left_body = np.array([0.0, -arm_length, 0.0])

    rotation = rotation_matrix_body_to_inertial(phi, theta, psi)

    front = center + rotation @ front_body
    back = center + rotation @ back_body
    right = center + rotation @ right_body
    left = center + rotation @ left_body

    return {
        "center": center,
        "front": front,
        "back": back,
        "right": right,
        "left": left,
        "arm_1": (front, back),
        "arm_2": (right, left),
    }


def animate_quadcopter_6dof(
    time,
    states,
    reference_positions=None,
    waypoints=None,
    arm_length=0.25,
    trail_length=None,
    frame_stride=5,
    interval=30,
    save_path=None,
    show=True,
    title="6-DOF Quadcopter Animation",
):
    """Animate a full 6-DOF quadcopter trajectory with a simple body model.

    The state convention is ``[x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]``.
    Optional reference positions and waypoints are plotted as static context.
    """
    time_array, state_array = _validate_time_and_states(time, states)
    reference_positions = _validate_optional_positions(
        reference_positions,
        "reference_positions",
        expected_length=len(time_array),
    )
    waypoints = _validate_optional_positions(waypoints, "waypoints")
    arm_length = _validate_positive(arm_length, "arm_length")
    frame_stride = _validate_positive_integer(frame_stride, "frame_stride")
    interval = _validate_positive(interval, "interval")

    if trail_length is not None:
        trail_length = _validate_positive_integer(trail_length, "trail_length")

    _validate_save_path_extension(save_path)

    frames = _frame_indices(len(time_array), frame_stride)
    x_limits, y_limits, z_limits = _axis_limits(
        state_array,
        arm_length,
        reference_positions,
        waypoints,
    )

    (figure, axis), show = _create_figure_and_axis(show, save_path)
    axis.set_title(title)
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_zlabel("z (m)")
    axis.set_xlim(*x_limits)
    axis.set_ylim(*y_limits)
    axis.set_zlim(*z_limits)
    axis.set_box_aspect(
        (
            x_limits[1] - x_limits[0],
            y_limits[1] - y_limits[0],
            z_limits[1] - z_limits[0],
        )
    )
    axis.grid(True)

    positions = state_array[:, 0:3]

    if reference_positions is not None:
        axis.plot(
            reference_positions[:, 0],
            reference_positions[:, 1],
            reference_positions[:, 2],
            linestyle="--",
            color="tab:orange",
            linewidth=1.5,
            label="Reference path",
        )

    if waypoints is not None:
        axis.plot(
            waypoints[:, 0],
            waypoints[:, 1],
            waypoints[:, 2],
            marker="o",
            linestyle=":",
            color="tab:red",
            label="Waypoints",
        )

    path_line, = axis.plot([], [], [], color="tab:blue", linewidth=1.5, label="Trail")
    arm_1_line, = axis.plot([], [], [], color="black", linewidth=2.5)
    arm_2_line, = axis.plot([], [], [], color="black", linewidth=2.5)
    rotor_markers, = axis.plot(
        [],
        [],
        [],
        linestyle="",
        marker="o",
        markersize=7,
        color="tab:green",
        label="Rotors",
    )
    center_marker, = axis.plot(
        [],
        [],
        [],
        linestyle="",
        marker="o",
        markersize=5,
        color="tab:blue",
        label="Body center",
    )
    time_text = axis.text2D(
        0.02,
        0.95,
        "",
        transform=axis.transAxes,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )
    axis.legend(loc="upper right")

    def _set_line_3d(line, points):
        line.set_data(points[:, 0], points[:, 1])
        line.set_3d_properties(points[:, 2])

    def update(frame_number):
        """Update drone body, trail, and time label for one animation frame."""
        sample_index = frames[frame_number]
        state = state_array[sample_index]
        geometry = compute_quadcopter_body_points(
            state[0:3],
            state[6],
            state[7],
            state[8],
            arm_length=arm_length,
        )

        if trail_length is None:
            trail_start = 0
        else:
            trail_start = max(0, sample_index - trail_length + 1)
        trail = positions[trail_start : sample_index + 1]

        arm_1_points = np.vstack(geometry["arm_1"])
        arm_2_points = np.vstack(geometry["arm_2"])
        rotor_points = np.vstack(
            [
                geometry["front"],
                geometry["back"],
                geometry["right"],
                geometry["left"],
            ]
        )
        center = geometry["center"].reshape(1, 3)

        _set_line_3d(path_line, trail)
        _set_line_3d(arm_1_line, arm_1_points)
        _set_line_3d(arm_2_line, arm_2_points)
        _set_line_3d(rotor_markers, rotor_points)
        _set_line_3d(center_marker, center)
        time_text.set_text(f"t = {time_array[sample_index]:.2f} s")

        return (
            path_line,
            arm_1_line,
            arm_2_line,
            rotor_markers,
            center_marker,
            time_text,
        )

    anim = animation.FuncAnimation(
        figure,
        update,
        frames=len(frames),
        interval=interval,
        blit=False,
        repeat=True,
    )

    if save_path is not None:
        _save_animation(anim, save_path, interval)

    if show:
        plt.show()

    return anim
