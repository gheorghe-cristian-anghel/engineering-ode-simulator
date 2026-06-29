"""Interactive Streamlit app for selected engineering ODE simulations."""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from models.dc_motor import rad_per_sec_to_rpm
from models.discrete_pid import DiscretePID, simulate_discrete_pid_motor_control
from models.rc_circuit import analytical_rc, simulate_rc
from models.rlc_circuit import damping_ratio, natural_frequency, simulate_rlc


def response_type(zeta):
    """Return a readable damping classification from damping ratio."""
    if np.isclose(zeta, 1.0):
        return "critically damped"

    if zeta < 1.0:
        return "underdamped"

    return "overdamped"


def format_value(value, unit="", precision=3):
    """Format a numeric value for Streamlit metrics."""
    return f"{value:.{precision}f} {unit}".strip()


def show_figure(figure):
    """Display a Matplotlib figure in Streamlit, then close it."""
    st.pyplot(figure)
    plt.close(figure)


def render_rc_circuit():
    """Render the RC circuit charging simulation."""
    st.header("RC Circuit Charging")
    st.write(
        "A resistor-capacitor circuit charges toward the input voltage with "
        "time constant `tau = R*C`."
    )

    with st.sidebar:
        st.subheader("RC Circuit Inputs")
        resistance = st.number_input(
            "Resistance R (ohms)",
            min_value=1.0,
            value=1000.0,
            step=100.0,
        )
        capacitance = st.number_input(
            "Capacitance C (F)",
            min_value=0.000001,
            value=0.001,
            step=0.0001,
            format="%.6f",
        )
        input_voltage = st.number_input("Input voltage Vin (V)", value=5.0, step=1.0)
        initial_voltage = st.number_input(
            "Initial capacitor voltage Vc0 (V)",
            value=0.0,
            step=0.5,
        )
        simulation_time = st.number_input(
            "Simulation time (s)",
            min_value=0.1,
            value=5.0,
            step=0.5,
        )

    t, capacitor_voltage = simulate_rc(
        resistance,
        capacitance,
        input_voltage,
        initial_voltage,
        (0.0, simulation_time),
        500,
    )
    analytical_voltage = analytical_rc(
        t,
        resistance,
        capacitance,
        input_voltage,
        initial_voltage,
    )

    tau = resistance * capacitance

    col1, col2 = st.columns(2)
    col1.metric("Time constant", format_value(tau, "s"))
    col2.metric("Final capacitor voltage", format_value(capacitor_voltage[-1], "V"))

    figure, axis = plt.subplots()
    axis.plot(t, capacitor_voltage, label="Numerical solution")
    axis.plot(t, analytical_voltage, "--", label="Analytical solution")
    axis.set_title("RC Circuit Charging")
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Capacitor voltage (V)")
    axis.grid(True)
    axis.legend()
    figure.tight_layout()
    show_figure(figure)

    st.write("Equation: `dVc/dt = (Vin - Vc) / (R*C)`")


def render_rlc_circuit():
    """Render the series RLC circuit step response simulation."""
    st.header("RLC Circuit Step Response")
    st.write(
        "A series resistor-inductor-capacitor circuit shows second-order "
        "electrical dynamics, including damping, overshoot, and settling."
    )

    with st.sidebar:
        st.subheader("RLC Circuit Inputs")
        resistance = st.number_input(
            "Resistance R (ohms)",
            min_value=0.0,
            value=2.0,
            step=0.5,
        )
        inductance = st.number_input(
            "Inductance L (H)",
            min_value=0.001,
            value=1.0,
            step=0.1,
        )
        capacitance = st.number_input(
            "Capacitance C (F)",
            min_value=0.000001,
            value=0.25,
            step=0.05,
            format="%.6f",
        )
        input_voltage = st.number_input("Input voltage Vin (V)", value=5.0, step=1.0)
        initial_voltage = st.number_input(
            "Initial capacitor voltage (V)",
            value=0.0,
            step=0.5,
        )
        initial_current = st.number_input(
            "Initial current (A)",
            value=0.0,
            step=0.1,
        )
        simulation_time = st.number_input(
            "Simulation time (s)",
            min_value=0.1,
            value=10.0,
            step=0.5,
        )

    t, capacitor_voltage, current = simulate_rlc(
        resistance,
        inductance,
        capacitance,
        input_voltage,
        initial_voltage,
        initial_current,
        (0.0, simulation_time),
        1000,
    )

    omega_n = natural_frequency(inductance, capacitance)
    zeta = damping_ratio(resistance, inductance, capacitance)

    col1, col2, col3 = st.columns(3)
    col1.metric("Natural frequency", format_value(omega_n, "rad/s"))
    col2.metric("Damping ratio", format_value(zeta))
    col3.metric("Response type", response_type(zeta))

    figure, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(t, capacitor_voltage, label="Capacitor voltage")
    axes[0].axhline(input_voltage, color="gray", linestyle=":", label="DC steady state")
    axes[0].set_title("RLC Capacitor Voltage Step Response")
    axes[0].set_ylabel("Voltage (V)")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(t, current, color="tab:orange", label="Current")
    axes[1].set_title("RLC Circuit Current")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Current (A)")
    axes[1].grid(True)
    axes[1].legend()
    figure.tight_layout()
    show_figure(figure)

    st.write(
        "The voltage plot shows how the capacitor approaches the DC input "
        "voltage. The current plot shows the transient current that decays "
        "toward zero at steady state."
    )


def render_dc_motor_pid_control():
    """Render the discrete PID DC motor speed-control simulation."""
    st.header("DC Motor PID Control")
    st.write(
        "A discrete PID controller updates a voltage command at a fixed sample "
        "time and holds that command while the continuous motor plant evolves."
    )

    with st.sidebar:
        st.subheader("PID Control Inputs")
        target_speed = st.number_input(
            "Target speed (rad/s)",
            min_value=1.0,
            value=80.0,
            step=5.0,
        )
        kp = st.number_input("Kp", min_value=0.0, value=0.16, step=0.01)
        ki = st.number_input(
            "Ki",
            min_value=0.0,
            value=0.018,
            step=0.001,
            format="%.3f",
        )
        kd = st.number_input("Kd", min_value=0.0, value=0.12, step=0.01)
        simulation_time = st.number_input(
            "Simulation time (s)",
            min_value=0.1,
            value=25.0,
            step=1.0,
        )
        sample_time = st.number_input(
            "Sample time dt (s)",
            min_value=0.001,
            value=0.01,
            step=0.001,
            format="%.3f",
        )
        voltage_limit = st.number_input(
            "Voltage limit (V)",
            min_value=1.0,
            value=24.0,
            step=1.0,
        )

    motor_params = {
        "R": 1.0,
        "L": 0.5,
        "J": 0.01,
        "b": 0.001,
        "Kt": 0.01,
        "Ke": 0.01,
    }

    pid = DiscretePID(
        kp,
        ki,
        kd,
        output_min=0.0,
        output_max=voltage_limit,
        anti_windup=True,
    )
    t, current, speed, voltage, error = simulate_discrete_pid_motor_control(
        motor_params["R"],
        motor_params["L"],
        motor_params["J"],
        motor_params["b"],
        motor_params["Kt"],
        motor_params["Ke"],
        pid,
        target_speed,
        0.0,
        0.0,
        simulation_time,
        sample_time,
    )

    final_error = target_speed - speed[-1]
    peak_speed = float(np.max(speed))
    overshoot = max(0.0, (peak_speed - target_speed) / target_speed * 100.0)

    col1, col2, col3 = st.columns(3)
    col1.metric("Final speed", format_value(speed[-1], "rad/s"))
    col2.metric("Final error", format_value(final_error, "rad/s"))
    col3.metric("Overshoot", format_value(overshoot, "%"))

    col4, col5, col6 = st.columns(3)
    col4.metric("Peak speed", format_value(peak_speed, "rad/s"))
    col5.metric(
        "Final speed in rpm",
        format_value(rad_per_sec_to_rpm(speed[-1]), "rpm"),
    )
    col6.metric("Max voltage", format_value(np.max(voltage), "V"))

    figure, axes = plt.subplots(3, 1, sharex=True)
    axes[0].plot(t, speed, label="Motor speed")
    axes[0].axhline(target_speed, color="gray", linestyle=":", label="Target speed")
    axes[0].set_title("Discrete PID DC Motor Speed Control")
    axes[0].set_ylabel("Speed (rad/s)")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(t, voltage, color="tab:orange", label="Control voltage")
    axes[1].axhline(voltage_limit, color="gray", linestyle=":", label="Voltage limit")
    axes[1].set_title("Held Voltage Command")
    axes[1].set_ylabel("Voltage (V)")
    axes[1].grid(True)
    axes[1].legend()

    axes[2].plot(t, current, color="tab:green", label="Armature current")
    axes[2].set_title("Armature Current")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Current (A)")
    axes[2].grid(True)
    axes[2].legend()
    figure.tight_layout()
    show_figure(figure)

    st.write(
        "The controller compares target speed with measured speed, computes a "
        "voltage command from proportional, integral, and derivative terms, "
        "then clips that command to the configured voltage limit."
    )


def main():
    """Run the Streamlit application."""
    st.set_page_config(page_title="Engineering ODE Simulator", layout="wide")
    st.title("Engineering ODE Simulator")
    st.write(
        "Interactive browser-based simulations for selected electrical and "
        "control-system ODE models. The app reuses the existing model modules "
        "and keeps plotting/UI code separate from the simulation equations."
    )

    simulation = st.sidebar.selectbox(
        "Select simulation",
        ("RC Circuit", "RLC Circuit", "DC Motor PID Control"),
    )

    if simulation == "RC Circuit":
        render_rc_circuit()
    elif simulation == "RLC Circuit":
        render_rlc_circuit()
    else:
        render_dc_motor_pid_control()


if __name__ == "__main__":
    main()
