"""Animate the nonlinear open-loop inverted pendulum example."""

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.inverted_pendulum import simulate_inverted_pendulum
from visualization.inverted_pendulum_animation import animate_inverted_pendulum


def parse_args():
    """Parse optional animation output arguments."""
    parser = argparse.ArgumentParser(
        description="Animate the open-loop inverted pendulum / cart-pole model."
    )
    parser.add_argument(
        "--save",
        help="Optional .gif or .mp4 path for saving the animation.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Create or save the animation without opening an interactive window.",
    )

    return parser.parse_args()


def main():
    """Simulate and animate open-loop cart-pole instability."""
    args = parse_args()

    M = 1.0
    m = 0.1
    l = 0.5
    g = 9.81
    b = 0.0
    initial_angle_degrees = 5.0
    x0 = [0.0, 0.0, np.radians(initial_angle_degrees), 0.0]
    t_span = (0.0, 0.75)
    num_points = 250

    t, states = simulate_inverted_pendulum(
        x0,
        t_span=t_span,
        num_points=num_points,
        M=M,
        m=m,
        l=l,
        g=g,
        b=b,
    )

    print("Open-Loop Inverted Pendulum Animation")
    print("State: [cart_position, cart_velocity, theta, theta_dot]")
    print("theta = 0 is the upright equilibrium.")
    print(f"Initial angle: {initial_angle_degrees:.3f} degrees")
    print("The open-loop pendulum departs from upright without feedback control.")

    anim = animate_inverted_pendulum(
        t,
        states,
        l=l,
        title="Open-Loop Inverted Pendulum Animation",
        interval_ms=30,
        save_path=args.save,
        show=not args.no_show,
    )

    return anim


if __name__ == "__main__":
    main()
