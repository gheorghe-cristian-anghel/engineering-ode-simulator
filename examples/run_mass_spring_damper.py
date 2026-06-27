"""Run and plot the mass-spring-damper free vibration example."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.mass_spring_damper import (
    damping_ratio,
    natural_frequency,
    simulate_mass_spring_damper,
)


def main():
    """Simulate free vibration and plot displacement over time."""
    m = 1
    c = 0.4
    k = 4
    x0 = 1
    v0 = 0
    t_span = (0, 20)
    num_points = 500

    omega_n = natural_frequency(m, k)
    zeta = damping_ratio(m, c, k)

    print(f"Natural frequency: {omega_n:.3f} rad/s")
    print(f"Damping ratio: {zeta:.3f}")

    t, displacement, _ = simulate_mass_spring_damper(
        m,
        c,
        k,
        x0,
        v0,
        t_span,
        num_points,
    )

    plt.plot(t, displacement, label="Displacement")
    plt.xlabel("Time (s)")
    plt.ylabel("Displacement (m)")
    plt.title("Mass-Spring-Damper Free Vibration")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
