"""Run and plot the Bode response for a series RLC low-pass circuit."""

import sys
from pathlib import Path
from _tkinter import TclError

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.frequency_response import (
    compute_frequency_response,
    plot_bode_response,
    rlc_lowpass_transfer_function,
)
from models.rlc_circuit import damping_ratio, natural_frequency


def _plot_response(w, magnitude_db, phase_deg):
    """Plot the Bode response, falling back to saving if Tk is unavailable."""
    output_path = PROJECT_ROOT / "examples" / "frequency_response_rlc.png"

    try:
        figure = plot_bode_response(
            w,
            magnitude_db,
            phase_deg,
            title="Series RLC Capacitor Voltage Bode Plot",
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
            title="Series RLC Capacitor Voltage Bode Plot",
        )
        figure.savefig(output_path, dpi=150)
        print("Interactive Matplotlib window is unavailable in this environment.")
        print(f"Plot saved to: {output_path}")


def main():
    """Compute and plot the frequency response of a series RLC circuit."""
    R = 2.0
    L = 1.0
    C = 0.25

    num, den = rlc_lowpass_transfer_function(R, L, C)
    w, magnitude_db, phase_deg = compute_frequency_response(
        num,
        den,
        w_min=1e-2,
        w_max=1e2,
        num_points=800,
    )

    omega_n = natural_frequency(L, C)
    zeta = damping_ratio(R, L, C)
    peak_index = magnitude_db.argmax()

    print("Series RLC Low-Pass Frequency Response:")
    print(f"Resistance R: {R} ohms")
    print(f"Inductance L: {L} H")
    print(f"Capacitance C: {C} F")
    print(f"Natural frequency: {omega_n:.3f} rad/s")
    print(f"Damping ratio: {zeta:.3f}")
    print(
        "Peak magnitude: "
        f"{magnitude_db[peak_index]:.3f} dB at {w[peak_index]:.3f} rad/s"
    )
    print(f"High-frequency phase: {phase_deg[-1]:.3f} deg")

    _plot_response(w, magnitude_db, phase_deg)


if __name__ == "__main__":
    main()
