"""Run and plot the Newton's Law of Cooling example."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.cooling import analytical_cooling, simulate_cooling


def main():
    """Simulate cooling and compare numerical and analytical results."""
    T0 = 90
    T_env = 22
    k = 0.08
    t_span = (0, 60)
    num_points = 200

    t, numerical_temperature = simulate_cooling(k, T_env, T0, t_span, num_points)
    analytical_temperature = analytical_cooling(t, k, T_env, T0)

    plt.plot(t, numerical_temperature, label="Numerical solution")
    plt.plot(t, analytical_temperature, "--", label="Analytical solution")
    plt.xlabel("Time (minutes)")
    plt.ylabel("Temperature (degrees Celsius)")
    plt.title("Newton's Law of Cooling")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
