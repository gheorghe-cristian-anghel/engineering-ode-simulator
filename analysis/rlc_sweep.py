"""RLC parameter sweep helpers for educational transient studies."""

from analysis.step_response import calculate_step_info
from models.rlc_circuit import damping_ratio, natural_frequency, simulate_rlc


VALID_RLC_SWEEP_PARAMETERS = ("R", "L", "C")


def summarize_rlc_response(t, vc, vin, R, L, C):
    """Return summary metrics for a series RLC capacitor-voltage response."""
    step_info = calculate_step_info(t, vc)

    return {
        "natural_frequency": natural_frequency(L, C),
        "damping_ratio": damping_ratio(R, L, C),
        "final_voltage": float(vc[-1]),
        "peak_voltage": step_info.peak_value,
        "overshoot_percent": step_info.overshoot_percent,
        "settling_time": step_info.settling_time,
    }


def _validate_sweep_inputs(parameter_name, parameter_values):
    """Validate common RLC sweep inputs."""
    if parameter_name not in VALID_RLC_SWEEP_PARAMETERS:
        valid_names = ", ".join(VALID_RLC_SWEEP_PARAMETERS)
        raise ValueError(f"parameter_name must be one of: {valid_names}")

    if len(parameter_values) == 0:
        raise ValueError("parameter_values must not be empty")


def run_rlc_sweep(
    parameter_name,
    parameter_values,
    R=2.0,
    L=1.0,
    C=0.25,
    Vin=5.0,
    t_span=(0.0, 10.0),
    num_points=1000,
):
    """Run a series RLC sweep over resistance, inductance, or capacitance.

    Parameters
    ----------
    parameter_name : str
        Parameter to sweep. Must be ``"R"``, ``"L"``, or ``"C"``.
    parameter_values : sequence
        Values to test for the selected parameter.
    R : float, optional
        Baseline resistance in ohms.
    L : float, optional
        Baseline inductance in henries.
    C : float, optional
        Baseline capacitance in farads.
    Vin : float, optional
        Step input voltage in volts.
    t_span : tuple, optional
        Start and end time in seconds.
    num_points : int, optional
        Number of time samples.

    Returns
    -------
    tuple
        ``(results, simulations)`` where each result is a summary dictionary
        and each simulation dictionary contains ``t``, ``capacitor_voltage``,
        and ``current`` arrays.
    """
    _validate_sweep_inputs(parameter_name, parameter_values)

    results = []
    simulations = []

    for parameter_value in parameter_values:
        case_R = R
        case_L = L
        case_C = C

        if parameter_name == "R":
            case_R = parameter_value
        elif parameter_name == "L":
            case_L = parameter_value
        else:
            case_C = parameter_value

        t, capacitor_voltage, current = simulate_rlc(
            case_R,
            case_L,
            case_C,
            Vin,
            Vc0=0.0,
            i0=0.0,
            t_span=t_span,
            num_points=num_points,
        )
        summary = summarize_rlc_response(
            t,
            capacitor_voltage,
            Vin,
            case_R,
            case_L,
            case_C,
        )
        summary["parameter_name"] = parameter_name
        summary["parameter_value"] = float(parameter_value)

        results.append(summary)
        simulations.append(
            {
                "parameter_name": parameter_name,
                "parameter_value": float(parameter_value),
                "R": case_R,
                "L": case_L,
                "C": case_C,
                "t": t,
                "capacitor_voltage": capacitor_voltage,
                "current": current,
            }
        )

    return results, simulations
