"""Run and plot the Bode response for a second-order low-pass system."""

import sys
from pathlib import Path
from _tkinter import TclError

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.frequency_response import (
    compute_frequency_response,
    plot_bode_response,
    second_order_transfer_function,
)


def _plot_response(w, magnitude_db, phase_deg):
    """Plot the Bode response, falling back to saving if Tk is unavailable."""
    output_path = PROJECT_ROOT / "examples" / "frequency_response_second_order.png"

    try:
        figure = plot_bode_response(
            w,
            magnitude_db,
            phase_deg,
            title="Second-Order Low-Pass Bode Plot",
        )
        figure.savefig(output_path, dpi=150)
        print(f"Plot saved to: {output_path}")
        plt.show()
    except TclError:
        plt.switch_backend("Agg")
        plt.close("all")
        figure = plot_bode_response(
            w,
            magnitude_db,
            phase_deg,
            title="Second-Order Low-Pass Bode Plot",
        )
        figure.savefig(output_path, dpi=150)
        print("Interactive Matplotlib window is unavailable in this environment.")
        print(f"Plot saved to: {output_path}")


def main():
    """Compute and plot the frequency response of a second-order system."""
    omega_n = 5.0
    zeta = 0.3
    K = 1.0

    num, den = second_order_transfer_function(omega_n, zeta, K)
    w, magnitude_db, phase_deg = compute_frequency_response(
        num,
        den,
        w_min=1e-1,
        w_max=1e2,
        num_points=800,
    )

    peak_index = magnitude_db.argmax()

    print("Second-Order Low-Pass Frequency Response:")
    print(f"Natural frequency omega_n: {omega_n:.3f} rad/s")
    print(f"Damping ratio zeta: {zeta:.3f}")
    print(f"Gain K: {K}")
    print(
        "Peak magnitude: "
        f"{magnitude_db[peak_index]:.3f} dB at {w[peak_index]:.3f} rad/s"
    )
    print(f"High-frequency phase: {phase_deg[-1]:.3f} deg")

    _plot_response(w, magnitude_db, phase_deg)


if __name__ == "__main__":
    main()
