"""Animate 6-DOF quadcopter circular trajectory tracking."""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.quadcopter_trajectory_tracking import (  # noqa: E402
    circular_trajectory,
    simulate_quadcopter_trajectory_tracking,
)
from visualization.quadcopter_animation import animate_quadcopter_6dof  # noqa: E402


def parse_args():
    """Parse optional animation output arguments."""
    parser = argparse.ArgumentParser(
        description="Animate 6-DOF quadcopter circular trajectory tracking."
    )
    parser.add_argument(
        "--save",
        help=(
            "Optional .gif or .mp4 path for saving the animation. "
            "GIF saving uses the Matplotlib Pillow writer."
        ),
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Create or save the animation without opening an interactive window.",
    )

    return parser.parse_args()


def main():
    """Simulate circular trajectory tracking and animate the 6-DOF body motion."""
    args = parse_args()
    save_animation = args.save is not None

    if args.no_show and not save_animation:
        warnings.filterwarnings(
            "ignore",
            message="Animation was deleted without rendering anything.*",
            category=UserWarning,
            module="matplotlib.animation",
        )

    radius = 1.0
    altitude = 2.0
    angular_speed = 0.3
    dt = 0.03
    frame_stride = 5

    initial_state = np.zeros(12)
    initial_state[0:3] = [radius, 0.0, altitude]

    result = simulate_quadcopter_trajectory_tracking(
        circular_trajectory(
            radius=radius,
            altitude=altitude,
            angular_speed=angular_speed,
        ),
        initial_state=initial_state,
        t_span=(0.0, 12.0),
        dt=dt,
    )
    metrics = result["tracking_metrics"]
    frame_count = (len(result["time"]) - 1) // frame_stride + 1

    print("6-DOF Quadcopter Circular Trajectory Animation")
    print("The animation shows attitude changes while tracking a circular path.")
    print()
    print(f"Simulation duration: {result['time'][-1] - result['time'][0]:.3f} s")
    print(f"Animation frames: {frame_count}")
    print(f"Final position error norm: {metrics['final_position_error_norm']:.4f} m")
    print(f"Saving enabled: {save_animation}")

    if save_animation:
        print(f"Animation save path: {args.save}")

    try:
        anim = animate_quadcopter_6dof(
            result["time"],
            result["states"],
            reference_positions=result["reference_positions"],
            frame_stride=frame_stride,
            interval=30,
            save_path=args.save,
            show=not args.no_show,
            title="6-DOF Quadcopter Circular Trajectory",
        )
    except RuntimeError as error:
        print(f"Animation saving failed: {error}")
        raise

    if save_animation:
        print("Animation saved.")

    return anim


if __name__ == "__main__":
    main()
