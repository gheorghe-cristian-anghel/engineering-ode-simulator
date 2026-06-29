"""Run and plot the Bode response for a first-order low-pass system."""

import sys
from pathlib import Path
from _tkinter import TclError

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.frequency_response import (
    compute_frequency_response,
    first_order_transfer_function,
    plot_bode_response,
)


def _plot_response(w, magnitude_db, phase_deg):
    """Plot the Bode response, falling back to saving if Tk is unavailable."""
    output_path = PROJECT_ROOT / "examples" / "frequency_response_first_order.png"

    try:
        figure = plot_bode_response(
            w,
            magnitude_db,
            phase_deg,
            title="First-Order Low-Pass Bode Plot",
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
            title="First-Order Low-Pass Bode Plot",
        )
        figure.savefig(output_path, dpi=150)
        print("Interactive Matplotlib window is unavailable in this environment.")
        print(f"Plot saved to: {output_path}")


def main():
    """Compute and plot the frequency response of a first-order system."""
    K = 1.0
    tau = 1.0
    cutoff_frequency = 1 / tau

    num, den = first_order_transfer_function(K, tau)
    w, magnitude_db, phase_deg = compute_frequency_response(
        num,
        den,
        w_min=1e-2,
        w_max=1e2,
        num_points=500,
    )

    cutoff_index = abs(w - cutoff_frequency).argmin()

    print("First-Order Low-Pass Frequency Response:")
    print(f"Gain K: {K}")
    print(f"Time constant tau: {tau} s")
    print(f"Cutoff frequency: {cutoff_frequency:.3f} rad/s")
    print(f"Low-frequency magnitude: {magnitude_db[0]:.3f} dB")
    print(
        "Magnitude near cutoff: "
        f"{magnitude_db[cutoff_index]:.3f} dB at {w[cutoff_index]:.3f} rad/s"
    )
    print(f"High-frequency phase: {phase_deg[-1]:.3f} deg")

    _plot_response(w, magnitude_db, phase_deg)


if __name__ == "__main__":
    main()
