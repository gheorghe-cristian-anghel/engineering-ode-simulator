"""Run and plot the RC circuit charging example."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.rc_circuit import analytical_rc, simulate_rc


def main():
    """Simulate an RC circuit and compare numerical and analytical results."""
    R = 1000
    C = 0.001
    Vin = 5
    V0 = 0
    t_span = (0, 5)
    num_points = 200

    t, numerical_voltage = simulate_rc(R, C, Vin, V0, t_span, num_points)
    analytical_voltage = analytical_rc(t, R, C, Vin, V0)

    plt.plot(t, numerical_voltage, label="Numerical solution")
    plt.plot(t, analytical_voltage, "--", label="Analytical solution")
    plt.xlabel("Time (s)")
    plt.ylabel("Capacitor voltage (V)")
    plt.title("RC Circuit Charging")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
