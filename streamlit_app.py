"""Interactive Streamlit app for selected engineering simulations."""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from analysis.finite_difference import (
    backward_difference,
    central_difference,
    estimate_convergence_order,
    forward_difference,
    rms_error,
    uniform_grid_1d,
)
from analysis.finite_element_1d import simulate_axial_bar_fem
from models.dc_motor import rad_per_sec_to_rpm
from models.discrete_pid import DiscretePID, simulate_discrete_pid_motor_control
from models.heat_equation_1d import (
    gaussian_initial_condition,
    heat_stability_number,
    simulate_heat_equation_1d,
)
from models.heat_equation_2d import (
    gaussian_hotspot_2d,
    heat_stability_numbers_2d,
    simulate_heat_equation_2d,
)
from models.rc_circuit import analytical_rc, simulate_rc
from models.rlc_circuit import damping_ratio, natural_frequency, simulate_rlc
from models.wave_equation_1d import (
    gaussian_displacement,
    simulate_wave_equation_1d,
    wave_cfl_number,
    zero_initial_velocity,
)
from models.wave_equation_2d import (
    gaussian_displacement_2d,
    simulate_wave_equation_2d,
    wave_stability_numbers_2d,
    zero_initial_velocity_2d,
)


DOMAIN_DEMOS = {
    "Home / overview": ("Overview",),
    "Circuits": ("RC Circuit", "RLC Circuit"),
    "Mechanical systems": ("Portfolio examples",),
    "Control systems": ("DC Motor PID Control",),
    "State estimation": ("Portfolio examples",),
    "UAV / quadcopter": ("Portfolio examples",),
    "PDEs": (
        "1D Heat Equation",
        "1D Wave Equation",
        "2D Heat Equation",
        "2D Wave Equation",
    ),
    "Numerical methods": ("Finite Difference Convergence",),
    "FEM basics": ("1D Axial Bar FEM",),
}


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


def profile_indices(values, fractions):
    """Return stable sample indices for a sequence."""
    max_index = len(values) - 1
    return [int(round(fraction * max_index)) for fraction in fractions]


def store_every_for_frame_cap(num_intervals, max_frames=30):
    """Return a store interval that keeps 2D Streamlit arrays modest."""
    return max(1, int(np.ceil(num_intervals / max_frames)))


def show_figure(figure):
    """Display a Matplotlib figure in Streamlit, then close it."""
    st.pyplot(figure)
    plt.close(figure)


def show_stability_status(label, value, limit):
    """Show a stability metric with a simple pass/fail message."""
    if value <= limit:
        st.success(f"{label}: {value:.4f} <= {limit:.4f} stable")
        return True

    st.warning(
        f"{label}: {value:.4f} > {limit:.4f}. "
        "This explicit simulation is unstable, so it was not run."
    )
    return False


def render_home():
    """Render the app overview page."""
    st.header("Engineering Simulation Toolkit")
    st.write(
        "Interactive demonstrations for engineering dynamics, controls, "
        "partial differential equations, numerical methods, and FEM basics. "
        "The app reuses the tested project modules and keeps the simulations "
        "small enough for responsive exploration."
    )

    st.subheader("Best interactive demos in this app")
    st.write(
        "- PDEs: 1D/2D heat diffusion and 1D/2D wave propagation with explicit "
        "finite-difference stability checks.\n"
        "- Numerical methods: finite-difference convergence for forward, "
        "backward, and central first derivatives.\n"
        "- FEM basics: fixed-free 1D axial bar displacement, stress, support "
        "reaction, and analytical comparison.\n"
        "- Circuits/control: RC/RLC transients and discrete PID motor control."
    )

    st.subheader("Performance guardrails")
    st.write(
        "2D grids default to modest sizes, stored frames are capped, and "
        "explicit PDE solvers are blocked when the selected time step violates "
        "the heat-equation stability limit or wave-equation CFL limit."
    )


def render_portfolio_examples(domain):
    """Render a lightweight page for domains available through scripts."""
    st.header(domain)
    st.write(
        "This domain is represented in the repository through command-line "
        "examples, tests, and generated figures. The Streamlit polish pass "
        "keeps this page lightweight and focuses interactivity on the new PDE, "
        "numerical-methods, and FEM demos."
    )

    examples_by_domain = {
        "Mechanical systems": (
            "Mass-spring-damper vibration",
            "Simple pendulum",
            "Inverted pendulum open-loop and LQR examples",
        ),
        "State estimation": (
            "Kalman filter for DC motor and RLC systems",
            "Extended Kalman Filter pendulum observer",
            "Unscented Kalman Filter and Particle Filter pendulum observers",
        ),
        "UAV / quadcopter": (
            "Altitude and attitude dynamics",
            "Full 6-DOF quadcopter model",
            "Trajectory tracking, waypoint following, and obstacle avoidance",
        ),
    }

    for item in examples_by_domain.get(domain, ()):
        st.write(f"- {item}")


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

    figure, axis = plt.subplots(figsize=(8, 4.5))
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

    figure, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
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
        "voltage. The current plot shows transient current decaying toward "
        "zero at steady state."
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

    figure, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 7))
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


def render_heat_equation_1d():
    """Render the 1D heat equation simulation."""
    st.header("1D Heat Equation")
    st.write(
        "Heat diffusion smooths temperature gradients. A localized hot pulse "
        "spreads along the rod while its peak temperature decays."
    )

    with st.sidebar:
        st.subheader("1D Heat Inputs")
        length = st.slider("Rod length L (m)", 0.5, 2.0, 1.0, 0.1)
        alpha = st.slider("Thermal diffusivity alpha", 0.002, 0.05, 0.01, 0.002)
        num_points = st.slider("Grid points", 41, 151, 81, 10)
        t_final = st.slider("Final time (s)", 0.2, 5.0, 2.0, 0.2)
        dt = st.slider("Time step dt (s)", 0.0005, 0.02, 0.003, 0.0005)
        pulse_width = st.slider("Initial pulse width", 0.03, 0.20, 0.08, 0.01)
        amplitude = st.slider("Initial peak temperature", 0.2, 2.0, 1.0, 0.1)

    dx = length / (num_points - 1)
    r = heat_stability_number(alpha, dt, dx)
    col1, col2, col3 = st.columns(3)
    col1.metric("dx", format_value(dx, "m", 4))
    col2.metric("Requested dt", format_value(dt, "s", 4))
    col3.metric("Stability r", format_value(r, precision=4))

    if not show_stability_status("Explicit heat stability r", r, 0.5):
        return

    def initial_condition(x):
        return gaussian_initial_condition(
            x,
            center=0.5 * length,
            width=pulse_width,
            amplitude=amplitude,
        )

    result = simulate_heat_equation_1d(
        length=length,
        alpha=alpha,
        t_final=t_final,
        num_points=num_points,
        dt=dt,
        initial_condition=initial_condition,
    )
    temperature = result["temperature"]

    cols = st.columns(4)
    cols[0].metric("Actual dt", format_value(result["dt"], "s", 4))
    cols[1].metric("Initial peak", format_value(np.max(temperature[0])))
    cols[2].metric("Final peak", format_value(np.max(temperature[-1])))
    cols[3].metric(
        "Peak decay",
        format_value(np.max(temperature[0]) - np.max(temperature[-1])),
    )

    x = result["x"]
    t = result["t"]
    figure, axes = plt.subplots(2, 1, figsize=(8, 7))
    for index in profile_indices(t, [0.0, 0.25, 0.5, 1.0]):
        axes[0].plot(x, temperature[index], label=f"t = {t[index]:.2f} s")
    axes[0].set_title("Temperature Profiles")
    axes[0].set_xlabel("Position x (m)")
    axes[0].set_ylabel("Temperature")
    axes[0].grid(True)
    axes[0].legend()

    heatmap = axes[1].imshow(
        temperature,
        aspect="auto",
        origin="lower",
        extent=[x[0], x[-1], t[0], t[-1]],
        cmap="inferno",
    )
    axes[1].set_title("Temperature History")
    axes[1].set_xlabel("Position x (m)")
    axes[1].set_ylabel("Time (s)")
    figure.colorbar(heatmap, ax=axes[1], label="Temperature")
    figure.tight_layout()
    show_figure(figure)


def render_wave_equation_1d():
    """Render the 1D wave equation simulation."""
    st.header("1D Wave Equation")
    st.write(
        "A displacement pulse propagates through the domain, oscillates, and "
        "reflects from fixed boundaries. The explicit scheme is controlled by "
        "the CFL number."
    )

    with st.sidebar:
        st.subheader("1D Wave Inputs")
        length = st.slider("String length L (m)", 0.5, 2.0, 1.0, 0.1)
        c = st.slider("Wave speed c (m/s)", 0.2, 3.0, 1.0, 0.1)
        num_points = st.slider("Grid points", 51, 201, 121, 10)
        t_final = st.slider("Final time (s)", 0.2, 4.0, 2.0, 0.2)
        dt = st.slider("Time step dt (s)", 0.001, 0.03, 0.006, 0.001)
        pulse_width = st.slider("Initial pulse width", 0.03, 0.20, 0.08, 0.01)
        amplitude = st.slider("Initial displacement amplitude", 0.2, 2.0, 1.0, 0.1)

    dx = length / (num_points - 1)
    cfl = wave_cfl_number(c, dt, dx)
    col1, col2, col3 = st.columns(3)
    col1.metric("dx", format_value(dx, "m", 4))
    col2.metric("Requested dt", format_value(dt, "s", 4))
    col3.metric("CFL lambda", format_value(cfl, precision=4))

    if not show_stability_status("Wave CFL lambda", cfl, 1.0):
        return

    def initial_displacement(x):
        return gaussian_displacement(
            x,
            center=0.5 * length,
            width=pulse_width,
            amplitude=amplitude,
        )

    result = simulate_wave_equation_1d(
        length=length,
        c=c,
        t_final=t_final,
        num_points=num_points,
        dt=dt,
        initial_displacement=initial_displacement,
        initial_velocity=zero_initial_velocity,
    )
    displacement = result["displacement"]

    cols = st.columns(4)
    cols[0].metric("Actual dt", format_value(result["dt"], "s", 4))
    cols[1].metric("Max |u|", format_value(np.max(np.abs(displacement))))
    cols[2].metric("Final max", format_value(np.max(displacement[-1])))
    cols[3].metric("Final min", format_value(np.min(displacement[-1])))

    x = result["x"]
    t = result["t"]
    figure, axes = plt.subplots(2, 1, figsize=(8, 7))
    for index in profile_indices(t, [0.0, 0.25, 0.5, 0.75, 1.0]):
        axes[0].plot(x, displacement[index], label=f"t = {t[index]:.2f} s")
    axes[0].set_title("Displacement Profiles")
    axes[0].set_xlabel("Position x (m)")
    axes[0].set_ylabel("Displacement")
    axes[0].grid(True)
    axes[0].legend()

    color_limit = max(float(np.max(np.abs(displacement))), 1e-12)
    heatmap = axes[1].imshow(
        displacement,
        aspect="auto",
        origin="lower",
        extent=[x[0], x[-1], t[0], t[-1]],
        cmap="coolwarm",
        vmin=-color_limit,
        vmax=color_limit,
    )
    axes[1].set_title("Displacement History")
    axes[1].set_xlabel("Position x (m)")
    axes[1].set_ylabel("Time (s)")
    figure.colorbar(heatmap, ax=axes[1], label="Displacement")
    figure.tight_layout()
    show_figure(figure)


def render_heat_equation_2d():
    """Render the 2D heat equation simulation."""
    st.header("2D Heat Equation")
    st.write(
        "A hot spot diffuses across a rectangular plate. The explicit 2D heat "
        "scheme requires `rx + ry <= 0.5`, so unstable choices are blocked."
    )

    with st.sidebar:
        st.subheader("2D Heat Inputs")
        width = st.slider("Plate width (m)", 0.5, 2.0, 1.0, 0.1)
        height = st.slider("Plate height (m)", 0.5, 2.0, 1.0, 0.1)
        alpha = st.slider("Thermal diffusivity alpha", 0.002, 0.05, 0.01, 0.002)
        grid_size = st.slider("Grid size nx = ny", 31, 61, 41, 10)
        t_final = st.slider("Final time (s)", 0.2, 2.0, 0.8, 0.1)
        dt = st.slider("Time step dt (s)", 0.001, 0.03, 0.008, 0.001)
        hotspot_width = st.slider("Hot spot width", 0.04, 0.20, 0.08, 0.01)

    dx = width / (grid_size - 1)
    dy = height / (grid_size - 1)
    rx, ry, stability_sum = heat_stability_numbers_2d(alpha, dt, dx, dy)
    cols = st.columns(4)
    cols[0].metric("dx", format_value(dx, "m", 4))
    cols[1].metric("dy", format_value(dy, "m", 4))
    cols[2].metric("rx", format_value(rx, precision=4))
    cols[3].metric("ry", format_value(ry, precision=4))

    if not show_stability_status("2D heat stability rx + ry", stability_sum, 0.5):
        return

    num_intervals = max(1, int(np.ceil(t_final / dt)))
    store_every = store_every_for_frame_cap(num_intervals)
    def initial_condition(X, Y):
        return gaussian_hotspot_2d(
            X,
            Y,
            center=(0.5 * width, 0.5 * height),
            width=hotspot_width,
            amplitude=1.0,
        )

    result = simulate_heat_equation_2d(
        width=width,
        height=height,
        alpha=alpha,
        t_final=t_final,
        nx=grid_size,
        ny=grid_size,
        dt=dt,
        initial_condition=initial_condition,
        store_every=store_every,
    )
    temperature = result["temperature"]

    metric_cols = st.columns(4)
    metric_cols[0].metric("Actual dt", format_value(result["dt"], "s", 4))
    metric_cols[1].metric("Stored frames", str(len(result["t"])))
    metric_cols[2].metric("Initial peak", format_value(np.max(temperature[0])))
    metric_cols[3].metric("Final peak", format_value(np.max(temperature[-1])))

    draw_2d_heat_plots(result)


def render_wave_equation_2d():
    """Render the 2D wave equation simulation."""
    st.header("2D Wave Equation")
    st.write(
        "A membrane displacement propagates outward, oscillates, and reflects "
        "from fixed edges. The explicit 2D wave scheme requires "
        "`rx + ry <= 1`."
    )

    with st.sidebar:
        st.subheader("2D Wave Inputs")
        width = st.slider("Membrane width (m)", 0.5, 2.0, 1.0, 0.1)
        height = st.slider("Membrane height (m)", 0.5, 2.0, 1.0, 0.1)
        c = st.slider("Wave speed c (m/s)", 0.2, 2.0, 1.0, 0.1)
        grid_size = st.slider("Grid size nx = ny", 31, 61, 41, 10)
        t_final = st.slider("Final time (s)", 0.2, 1.5, 0.8, 0.1)
        dt = st.slider("Time step dt (s)", 0.001, 0.03, 0.012, 0.001)
        pulse_width = st.slider("Initial pulse width", 0.04, 0.16, 0.07, 0.01)

    dx = width / (grid_size - 1)
    dy = height / (grid_size - 1)
    lambda_x, lambda_y, rx, ry, stability_sum = wave_stability_numbers_2d(
        c,
        dt,
        dx,
        dy,
    )
    cols = st.columns(4)
    cols[0].metric("lambda x", format_value(lambda_x, precision=4))
    cols[1].metric("lambda y", format_value(lambda_y, precision=4))
    cols[2].metric("rx", format_value(rx, precision=4))
    cols[3].metric("ry", format_value(ry, precision=4))

    if not show_stability_status("2D wave stability rx + ry", stability_sum, 1.0):
        return

    num_intervals = max(1, int(np.ceil(t_final / dt)))
    store_every = store_every_for_frame_cap(num_intervals)
    def initial_displacement(X, Y):
        return gaussian_displacement_2d(
            X,
            Y,
            center=(0.45 * width, 0.5 * height),
            width=pulse_width,
            amplitude=1.0,
        )

    result = simulate_wave_equation_2d(
        width=width,
        height=height,
        c=c,
        t_final=t_final,
        nx=grid_size,
        ny=grid_size,
        dt=dt,
        initial_displacement=initial_displacement,
        initial_velocity=zero_initial_velocity_2d,
        store_every=store_every,
    )
    displacement = result["displacement"]

    metric_cols = st.columns(4)
    metric_cols[0].metric("Actual dt", format_value(result["dt"], "s", 4))
    metric_cols[1].metric("Stored frames", str(len(result["t"])))
    metric_cols[2].metric("Max |u|", format_value(np.max(np.abs(displacement))))
    metric_cols[3].metric("Final peak", format_value(np.max(displacement[-1])))

    draw_2d_wave_plots(result)


def draw_2d_heat_plots(result):
    """Draw 2D heat snapshots and a centerline profile."""
    x = result["x"]
    y = result["y"]
    t = result["t"]
    temperature = result["temperature"]
    snapshot_indices = [0, int(round(0.5 * (len(t) - 1))), len(t) - 1]

    figure, axes = plt.subplots(2, 2, figsize=(10, 8))
    flat_axes = axes.ravel()
    color_min = float(np.min(temperature))
    color_max = float(np.max(temperature))

    for axis, index in zip(flat_axes[:3], snapshot_indices):
        heatmap = axis.imshow(
            temperature[index],
            origin="lower",
            extent=[x[0], x[-1], y[0], y[-1]],
            aspect="auto",
            cmap="inferno",
            vmin=color_min,
            vmax=color_max,
        )
        axis.set_title(f"Temperature at t = {t[index]:.2f} s")
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
        figure.colorbar(heatmap, ax=axis, label="Temperature")

    centerline_axis = flat_axes[3]
    center_y_index = len(y) // 2
    for index in profile_indices(t, [0.0, 0.33, 0.67, 1.0]):
        centerline_axis.plot(
            x,
            temperature[index, center_y_index, :],
            label=f"t = {t[index]:.2f} s",
        )
    centerline_axis.set_title("Centerline Temperature")
    centerline_axis.set_xlabel("x (m)")
    centerline_axis.set_ylabel("Temperature")
    centerline_axis.grid(True)
    centerline_axis.legend()
    figure.tight_layout()
    show_figure(figure)


def draw_2d_wave_plots(result):
    """Draw 2D wave snapshots and a centerline profile."""
    x = result["x"]
    y = result["y"]
    t = result["t"]
    displacement = result["displacement"]
    snapshot_indices = [
        0,
        int(round(0.33 * (len(t) - 1))),
        int(round(0.67 * (len(t) - 1))),
        len(t) - 1,
    ]
    color_limit = max(float(np.max(np.abs(displacement))), 1e-12)

    figure, axes = plt.subplots(2, 3, figsize=(13, 8))
    flat_axes = axes.ravel()
    for axis, index in zip(flat_axes[:4], snapshot_indices):
        heatmap = axis.imshow(
            displacement[index],
            origin="lower",
            extent=[x[0], x[-1], y[0], y[-1]],
            aspect="auto",
            cmap="coolwarm",
            vmin=-color_limit,
            vmax=color_limit,
        )
        axis.set_title(f"Displacement at t = {t[index]:.2f} s")
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
        figure.colorbar(heatmap, ax=axis, label="Displacement")

    centerline_axis = flat_axes[4]
    center_y_index = len(y) // 2
    for index in profile_indices(t, [0.0, 0.25, 0.5, 0.75, 1.0]):
        centerline_axis.plot(
            x,
            displacement[index, center_y_index, :],
            label=f"t = {t[index]:.2f} s",
        )
    centerline_axis.set_title("Centerline Displacement")
    centerline_axis.set_xlabel("x (m)")
    centerline_axis.set_ylabel("Displacement")
    centerline_axis.grid(True)
    centerline_axis.legend()
    figure.delaxes(flat_axes[5])
    figure.tight_layout()
    show_figure(figure)


def render_finite_difference_convergence():
    """Render finite-difference convergence analysis."""
    st.header("Finite Difference Convergence")
    st.write(
        "This demo differentiates `f(x) = sin(2*pi*x)` on successively finer "
        "uniform grids. Forward and backward differences are first-order; "
        "central differences are second-order in the interior."
    )

    with st.sidebar:
        st.subheader("Convergence Inputs")
        max_points = st.slider("Finest grid points", 81, 641, 321, 80)
        frequency = st.slider("Sine frequency multiplier", 1, 4, 1, 1)

    base_points = [21, 41, 81, 161, 321, 641]
    num_points_values = [value for value in base_points if value <= max_points]
    dx_values = []
    forward_errors = []
    backward_errors = []
    central_errors = []

    for num_points in num_points_values:
        x, dx = uniform_grid_1d(0.0, 1.0, num_points)
        y = np.sin(2.0 * np.pi * frequency * x)
        exact = 2.0 * np.pi * frequency * np.cos(2.0 * np.pi * frequency * x)
        interior = slice(1, -1)

        forward = forward_difference(y, dx)
        backward = backward_difference(y, dx)
        central = central_difference(y, dx)

        dx_values.append(dx)
        forward_errors.append(rms_error(forward[interior], exact[interior]))
        backward_errors.append(rms_error(backward[interior], exact[interior]))
        central_errors.append(rms_error(central[interior], exact[interior]))

    forward_order = estimate_convergence_order(dx_values, forward_errors)
    backward_order = estimate_convergence_order(dx_values, backward_errors)
    central_order = estimate_convergence_order(dx_values, central_errors)

    cols = st.columns(3)
    cols[0].metric("Forward order", format_value(forward_order))
    cols[1].metric("Backward order", format_value(backward_order))
    cols[2].metric("Central order", format_value(central_order))

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.loglog(dx_values, forward_errors, "o-", label="Forward")
    axis.loglog(dx_values, backward_errors, "s-", label="Backward")
    axis.loglog(dx_values, central_errors, "^-", label="Central")
    reference_dx = np.asarray(dx_values)
    axis.loglog(
        reference_dx,
        forward_errors[-1] * (reference_dx / reference_dx[-1]),
        ":",
        label="O(dx)",
    )
    axis.loglog(
        reference_dx,
        central_errors[-1] * (reference_dx / reference_dx[-1]) ** 2,
        "--",
        label="O(dx^2)",
    )
    axis.set_title("Finite Difference Error Convergence")
    axis.set_xlabel("Grid spacing dx")
    axis.set_ylabel("RMS derivative error")
    axis.grid(True, which="both")
    axis.legend()
    axis.invert_xaxis()
    figure.tight_layout()
    show_figure(figure)

    st.dataframe(
        {
            "grid points": num_points_values,
            "dx": dx_values,
            "forward RMS error": forward_errors,
            "backward RMS error": backward_errors,
            "central RMS error": central_errors,
        },
        use_container_width=True,
    )


def render_axial_bar_fem():
    """Render the 1D axial bar FEM demo."""
    st.header("1D FEM Axial Bar")
    st.write(
        "A fixed-free axial bar is split into linear finite elements. The app "
        "assembles the stiffness matrix, applies the fixed support, solves "
        "nodal displacements, and compares the tip displacement with "
        "`F*L/(E*A)`."
    )

    with st.sidebar:
        st.subheader("FEM Inputs")
        length = st.slider("Bar length L (m)", 0.2, 5.0, 1.0, 0.1)
        youngs_modulus_gpa = st.slider("Young's modulus E (GPa)", 1.0, 250.0, 200.0, 1.0)
        area_mm2 = st.slider("Area A (mm^2)", 10.0, 1000.0, 100.0, 10.0)
        force = st.slider("End force F (N)", 100.0, 10000.0, 1000.0, 100.0)
        num_elements = st.slider("Number of elements", 1, 40, 8, 1)

    E = youngs_modulus_gpa * 1e9
    A = area_mm2 * 1e-6
    result = simulate_axial_bar_fem(
        length=length,
        E=E,
        A=A,
        force=force,
        num_elements=num_elements,
    )

    nodes = result["nodes"]
    elements = result["elements"]
    displacements = result["displacements"]
    stresses = result["stresses"]
    analytical = force * nodes / (E * A)
    fixed_reaction = result["reactions"][0]
    average_stress = float(np.mean(stresses))

    cols = st.columns(4)
    cols[0].metric("FEM tip displacement", format_value(result["tip_displacement"], "m", 6))
    cols[1].metric(
        "Analytical tip",
        format_value(result["analytical_tip_displacement"], "m", 6),
    )
    cols[2].metric("Reaction force", format_value(fixed_reaction, "N"))
    cols[3].metric("Average stress", format_value(average_stress / 1e6, "MPa"))

    element_centers = 0.5 * (nodes[elements[:, 0]] + nodes[elements[:, 1]])
    max_displacement = max(float(np.max(np.abs(displacements))), 1e-15)
    scale = 0.1 * length / max_displacement
    deformed_nodes = nodes + scale * displacements

    figure, axes = plt.subplots(3, 1, figsize=(8, 9))
    axes[0].plot(nodes, displacements, "o-", label="FEM displacement")
    axes[0].plot(nodes, analytical, "--", label="Analytical displacement")
    axes[0].set_title("Nodal Displacement")
    axes[0].set_xlabel("Position x (m)")
    axes[0].set_ylabel("Displacement u (m)")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].step(element_centers, stresses / 1e6, where="mid", label="Element stress")
    axes[1].axhline(
        force / A / 1e6,
        color="tab:orange",
        linestyle="--",
        label="Analytical stress",
    )
    axes[1].set_title("Element Stress")
    axes[1].set_xlabel("Element center x (m)")
    axes[1].set_ylabel("Stress (MPa)")
    axes[1].grid(True)
    axes[1].legend()

    axes[2].plot(nodes, np.zeros_like(nodes), "o-", label="Original bar")
    axes[2].plot(
        deformed_nodes,
        np.zeros_like(deformed_nodes) + 0.02,
        "o-",
        label=f"Deformed shape ({scale:.2e}x)",
    )
    axes[2].set_title("Exaggerated Deformed Shape")
    axes[2].set_xlabel("Position x (m)")
    axes[2].set_yticks([])
    axes[2].grid(True, axis="x")
    axes[2].legend()
    figure.tight_layout()
    show_figure(figure)

    st.write(
        f"Relative tip displacement error: "
        f"`{result['relative_tip_error']:.3e}`. "
        "For this uniform axial bar with a tip load, the linear FEM solution "
        "matches the analytical linear displacement field up to numerical "
        "roundoff."
    )


def main():
    """Run the Streamlit application."""
    st.set_page_config(page_title="Engineering Simulation Toolkit", layout="wide")
    st.sidebar.title("Engineering Simulation Toolkit")
    domain = st.sidebar.radio("Domain", tuple(DOMAIN_DEMOS.keys()))
    demo = st.sidebar.selectbox("Demo", DOMAIN_DEMOS[domain])

    st.title("Engineering Simulation Toolkit")

    if domain == "Home / overview":
        render_home()
    elif demo == "RC Circuit":
        render_rc_circuit()
    elif demo == "RLC Circuit":
        render_rlc_circuit()
    elif demo == "DC Motor PID Control":
        render_dc_motor_pid_control()
    elif demo == "1D Heat Equation":
        render_heat_equation_1d()
    elif demo == "1D Wave Equation":
        render_wave_equation_1d()
    elif demo == "2D Heat Equation":
        render_heat_equation_2d()
    elif demo == "2D Wave Equation":
        render_wave_equation_2d()
    elif demo == "Finite Difference Convergence":
        render_finite_difference_convergence()
    elif demo == "1D Axial Bar FEM":
        render_axial_bar_fem()
    else:
        render_portfolio_examples(domain)


if __name__ == "__main__":
    main()
