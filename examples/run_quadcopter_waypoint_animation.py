"""Animate 6-DOF quadcopter waypoint following."""

import argparse
import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.quadcopter_waypoint_following import (  # noqa: E402
    simulate_quadcopter_waypoint_following,
)
from visualization.quadcopter_animation import animate_quadcopter_6dof  # noqa: E402


def parse_args():
    """Parse optional animation output arguments."""
    parser = argparse.ArgumentParser(
        description="Animate 6-DOF quadcopter waypoint following."
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
    """Simulate waypoint following and animate the 6-DOF body motion."""
    args = parse_args()
    save_animation = args.save is not None

    if args.no_show and not save_animation:
        warnings.filterwarnings(
            "ignore",
            message="Animation was deleted without rendering anything.*",
            category=UserWarning,
            module="matplotlib.animation",
        )

    waypoints = [
        (0.0, 0.0, 1.0),
        (0.8, 0.0, 1.4),
        (0.8, 0.8, 1.8),
        (0.0, 0.8, 1.4),
        (0.0, 0.0, 1.8),
    ]
    segment_time = 3.0
    dt = 0.03
    hold_time = 1.5
    frame_stride = 5

    result = simulate_quadcopter_waypoint_following(
        waypoints,
        segment_time=segment_time,
        smoothing="smoothstep",
        dt=dt,
        hold_time=hold_time,
    )
    metrics = result["waypoint_metrics"]
    frame_count = (len(result["time"]) - 1) // frame_stride + 1

    print("6-DOF Quadcopter Waypoint Animation")
    print("The animation shows position and attitude during waypoint tracking.")
    print()
    print(f"Simulation duration: {result['time'][-1] - result['time'][0]:.3f} s")
    print(f"Animation frames: {frame_count}")
    print(f"Final waypoint error: {metrics['final_waypoint_error']:.4f} m")
    print(f"Saving enabled: {save_animation}")

    if save_animation:
        print(f"Animation save path: {args.save}")

    try:
        anim = animate_quadcopter_6dof(
            result["time"],
            result["states"],
            reference_positions=result["reference_positions"],
            waypoints=result["waypoints"],
            frame_stride=frame_stride,
            interval=30,
            save_path=args.save,
            show=not args.no_show,
            title="6-DOF Quadcopter Waypoint Following",
        )
    except RuntimeError as error:
        print(f"Animation saving failed: {error}")
        raise

    if save_animation:
        print("Animation saved.")

    return anim


if __name__ == "__main__":
    main()
