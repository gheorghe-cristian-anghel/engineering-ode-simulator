"""Animate nonlinear inverted pendulum stabilization with LQR."""

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.inverted_pendulum_lqr import (
    design_inverted_pendulum_lqr,
    simulate_inverted_pendulum_lqr,
)
from visualization.inverted_pendulum_animation import animate_inverted_pendulum


def parse_args():
    """Parse optional animation output arguments."""
    parser = argparse.ArgumentParser(
        description="Animate LQR stabilization of the inverted pendulum."
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
    """Design LQR feedback, simulate the nonlinear model, and animate it."""
    args = parse_args()

    M = 1.0
    m = 0.1
    l = 0.5
    g = 9.81
    b = 0.0
    force_limit = 20.0
    initial_angle_degrees = 5.0
    x0 = [0.0, 0.0, np.radians(initial_angle_degrees), 0.0]
    t_span = (0.0, 5.0)
    num_points = 500

    K, closed_loop_eigenvalues, _, _, _, _ = design_inverted_pendulum_lqr(
        M=M,
        m=m,
        l=l,
        g=g,
        b=b,
    )
    t, states, control_force = simulate_inverted_pendulum_lqr(
        x0,
        K,
        t_span=t_span,
        num_points=num_points,
        M=M,
        m=m,
        l=l,
        g=g,
        b=b,
        force_limit=force_limit,
    )

    print("LQR Inverted Pendulum Animation")
    print("State: [cart_position, cart_velocity, theta, theta_dot]")
    print("theta = 0 is the upright equilibrium.")
    print(f"Initial angle: {initial_angle_degrees:.3f} degrees")
    print(f"Final angle: {np.degrees(states[-1, 2]):.3f} degrees")
    print(f"Maximum control force: {np.max(np.abs(control_force)):.3f} N")
    print()
    print("LQR gain K:")
    print(np.array2string(K, precision=4, suppress_small=True))
    print()
    print("Closed-loop eigenvalues:")
    print(np.array2string(closed_loop_eigenvalues, precision=4, suppress_small=True))
    print()
    print("The LQR controller stabilizes the nonlinear model near upright locally.")

    anim = animate_inverted_pendulum(
        t,
        states,
        l=l,
        title="LQR Inverted Pendulum Animation",
        interval_ms=30,
        save_path=args.save,
        show=not args.no_show,
    )

    return anim


if __name__ == "__main__":
    main()
