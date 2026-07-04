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
from visualization.plot_style import (
    add_clean_colorbar,
    apply_engineering_plot_style,
    finalize_streamlit_figure,
    format_engineering_axes,
    format_heatmap_axes,
    set_equal_2d_axes,
)


APP_TITLE = "Engineering Simulation Toolkit"
APP_SUBTITLE = (
    "Python scientific computing, control systems, PDEs, FEM, and UAV simulation"
)

DOMAIN_DEMOS = {
    "Home": ("Overview",),
    "Control Systems": ("RC Circuit", "RLC Circuit", "DC Motor PID Control"),
    "State Estimation": ("Portfolio Examples",),
    "UAV / Quadcopter": ("Portfolio Examples",),
    "PDE Solvers": (
        "1D Heat Equation",
        "1D Wave Equation",
        "2D Heat Equation",
        "2D Wave Equation",
    ),
    "Numerical Methods": ("Finite Difference Convergence",),
    "FEM Basics": ("1D Axial Bar FEM",),
    "About": ("Project Scope",),
}

FEATURE_CARDS = (
    (
        "Control Systems",
        "RC/RLC transients, discrete PID motor control, step-response thinking, "
        "and classical engineering feedback concepts.",
        "Interactive demos",
    ),
    (
        "State Estimation",
        "Kalman, Extended Kalman, Unscented Kalman, and Particle Filter examples "
        "available through repository scripts and plots.",
        "Portfolio examples",
    ),
    (
        "UAV / Quadcopter",
        "Altitude, attitude, 6-DOF motion, trajectory tracking, waypoint following, "
        "and obstacle-avoidance examples.",
        "Portfolio examples",
    ),
    (
        "PDE Solvers",
        "1D/2D heat diffusion and wave propagation with explicit finite-difference "
        "stability checks.",
        "Interactive demos",
    ),
    (
        "Numerical Methods",
        "Finite differences, RMS error, convergence-order estimation, and "
        "stability-aware simulation workflows.",
        "Interactive demo",
    ),
    (
        "FEM Basics",
        "A clear 1D axial bar finite element workflow with stiffness assembly, "
        "displacement, reaction, and stress recovery.",
        "Interactive demo",
    ),
    (
        "Visualization / Streamlit",
        "Matplotlib figures, saved portfolio screenshots, and a browser UI for "
        "hands-on simulation exploration.",
        "Presentation layer",
    ),
)


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


def inject_custom_css():
    """Apply a lightweight local visual system for the Streamlit UI."""
    st.markdown(
        """
        <style>
        :root {
            --est-bg-soft: #f8fafc;
            --est-border: #d8dee9;
            --est-text-muted: #475569;
            --est-accent: #2563eb;
            --est-accent-soft: #eff6ff;
        }

        .block-container {
            padding-top: 1.75rem;
            padding-bottom: 3rem;
            max-width: 1240px;
        }

        .est-hero {
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
            border: 1px solid var(--est-border);
            border-radius: 8px;
            padding: 1.25rem 1.35rem;
            margin-bottom: 1.25rem;
        }

        .est-eyebrow {
            color: var(--est-accent);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            margin-bottom: 0.3rem;
            text-transform: uppercase;
        }

        .est-hero h1 {
            margin: 0;
            font-size: 2.1rem;
            line-height: 1.15;
        }

        .est-subtitle {
            color: var(--est-text-muted);
            font-size: 1.02rem;
            margin-top: 0.5rem;
            max-width: 850px;
        }

        .est-card {
            background: #ffffff;
            border: 1px solid var(--est-border);
            border-radius: 8px;
            padding: 1rem;
            min-height: 165px;
            margin-bottom: 1rem;
        }

        .est-card-title {
            font-weight: 700;
            font-size: 1.02rem;
            margin-bottom: 0.4rem;
        }

        .est-card-body {
            color: var(--est-text-muted);
            font-size: 0.94rem;
            line-height: 1.48;
        }

        .est-tag {
            display: inline-block;
            background: var(--est-accent-soft);
            border: 1px solid #bfdbfe;
            border-radius: 999px;
            color: #1d4ed8;
            font-size: 0.76rem;
            font-weight: 650;
            margin-top: 0.8rem;
            padding: 0.16rem 0.55rem;
        }

        .est-info-box {
            background: var(--est-bg-soft);
            border: 1px solid var(--est-border);
            border-radius: 8px;
            padding: 0.9rem 1rem;
            margin: 0.75rem 0 1rem;
        }

        .est-info-title {
            font-weight: 700;
            margin-bottom: 0.25rem;
        }

        .est-info-body {
            color: var(--est-text-muted);
            line-height: 1.5;
        }

        .est-divider {
            border-top: 1px solid var(--est-border);
            margin: 1.35rem 0 0.9rem;
        }

        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid var(--est-border);
            border-radius: 8px;
            padding: 0.75rem 0.85rem;
        }

        [data-testid="stMetricLabel"] {
            color: var(--est-text-muted);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(section=None):
    """Render the app-level title and subtitle."""
    eyebrow = "Portfolio engineering demo" if section is None else section
    st.markdown(
        f"""
        <section class="est-hero">
            <div class="est-eyebrow">{eyebrow}</div>
            <h1>{APP_TITLE}</h1>
            <div class="est-subtitle">{APP_SUBTITLE}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_feature_card(title, body, tag=None):
    """Render a compact visual feature card."""
    tag_html = f'<div class="est-tag">{tag}</div>' if tag else ""
    st.markdown(
        f"""
        <div class="est-card">
            <div class="est-card-title">{title}</div>
            <div class="est-card-body">{body}</div>
            {tag_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_row(metrics):
    """Render Streamlit metrics with consistent spacing."""
    columns = st.columns(len(metrics))
    for column, metric in zip(columns, metrics):
        label, value = metric[:2]
        delta = metric[2] if len(metric) > 2 else None
        column.metric(label, value, delta=delta)


def render_section_divider():
    """Render a subtle horizontal section divider."""
    st.markdown('<div class="est-divider"></div>', unsafe_allow_html=True)


def render_info_box(title, body):
    """Render a restrained information box."""
    st.markdown(
        f"""
        <div class="est-info-box">
            <div class="est-info-title">{title}</div>
            <div class="est-info-body">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_parameter_summary(parameters):
    """Render selected user inputs as a compact metric row."""
    st.caption("Selected setup")
    render_metric_row(parameters)


def show_figure(figure, caption=None):
    """Display a Matplotlib figure in Streamlit, then close it."""
    finalize_streamlit_figure(figure)
    st.pyplot(figure, use_container_width=True)
    if caption is not None:
        st.caption(caption)
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
    st.header("Home / Overview")
    render_info_box(
        "What this toolkit is",
        "A portfolio-oriented Python engineering simulation project for "
        "dynamics, control, state estimation, UAV motion, finite differences, "
        "PDEs, FEM basics, and scientific visualization. The Streamlit app "
        "reuses the tested project modules and keeps interactive demos small "
        "enough for responsive exploration.",
    )

    st.subheader("Domains covered")
    for row_start in range(0, len(FEATURE_CARDS), 3):
        columns = st.columns(3)
        for column, card in zip(columns, FEATURE_CARDS[row_start : row_start + 3]):
            with column:
                render_feature_card(*card)

    render_section_divider()
    st.subheader("Technologies and skills demonstrated")
    col1, col2 = st.columns(2)
    with col1:
        render_info_box(
            "Python scientific stack",
            "NumPy, SciPy, Matplotlib, Streamlit, pytest, reusable model modules, "
            "analysis helpers, and runnable example scripts.",
        )
    with col2:
        render_info_box(
            "Engineering workflow",
            "Equation-based modeling, numerical stability checks, controller "
            "experiments, estimator examples, FEM assembly, tests, documentation, "
            "and portfolio-ready plots.",
        )

    st.subheader("Best interactive demos")
    render_metric_row(
        (
            ("PDE pages", "4"),
            ("Control pages", "3"),
            ("Numerical demo", "1"),
            ("FEM demo", "1"),
        )
    )

    render_info_box(
        "Performance guardrails",
        "2D grids default to modest sizes, stored frames are capped, and "
        "explicit PDE solvers are blocked when the selected time step violates "
        "the heat-equation stability limit or wave-equation CFL limit.",
    )


def render_portfolio_examples(domain):
    """Render a lightweight page for domains available through scripts."""
    st.header(domain)
    render_info_box(
        "Repository-backed examples",
        "This domain is represented in the repository through command-line "
        "examples, tests, and generated figures. This page stays lightweight "
        "while the interactive app focuses on control, PDE, numerical-methods, "
        "and FEM demos.",
    )

    examples_by_domain = {
        "State Estimation": (
            "Kalman filter for DC motor and RLC systems",
            "Extended Kalman Filter pendulum observer",
            "Unscented Kalman Filter and Particle Filter pendulum observers",
        ),
        "UAV / Quadcopter": (
            "Altitude and attitude dynamics",
            "Full 6-DOF quadcopter model",
            "Trajectory tracking, waypoint following, and obstacle avoidance",
        ),
    }

    columns = st.columns(3)
    for column, item in zip(columns, examples_by_domain.get(domain, ())):
        with column:
            render_feature_card(item, "Available through tested modules, examples, and saved figures.")


def render_about():
    """Render project scope and implementation notes."""
    st.header("About")
    render_info_box(
        "Project scope",
        "Engineering Simulation Toolkit is an educational and portfolio-focused "
        "simulation project. It demonstrates readable equations, reproducible "
        "examples, tested numerical behavior, and a practical Streamlit UI.",
    )

    st.subheader("What is intentionally included")
    columns = st.columns(3)
    included = (
        (
            "Models and analysis",
            "ODE/PDE models, controls, estimators, MPC, finite differences, and "
            "introductory FEM utilities.",
        ),
        (
            "Presentation",
            "Matplotlib plots, saved example figures, Streamlit controls, metrics, "
            "and concise explanatory text.",
        ),
        (
            "Quality signals",
            "Focused tests, documentation, examples, and stability checks for "
            "explicit numerical methods.",
        ),
    )
    for column, card in zip(columns, included):
        with column:
            render_feature_card(card[0], card[1])

    render_section_divider()
    render_info_box(
        "Educational limits",
        "The app is not a certified control system, flight stack, CFD package, "
        "or industrial FEM solver. The goal is clear engineering intuition and "
        "credible software structure for demos, discussions, and portfolio use.",
    )


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

    render_metric_row(
        (
            ("Time constant", format_value(tau, "s")),
            ("Final capacitor voltage", format_value(capacitor_voltage[-1], "V")),
        )
    )

    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(t, capacitor_voltage, label="Numerical solution")
    axis.plot(t, analytical_voltage, "--", label="Analytical solution")
    format_engineering_axes(
        axis,
        title="RC Circuit Charging",
        xlabel="Time (s)",
        ylabel="Capacitor voltage (V)",
    )
    show_figure(
        figure,
        "Capacitor voltage approaches the input voltage with the expected "
        "first-order charging curve.",
    )

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

    render_metric_row(
        (
            ("Natural frequency", format_value(omega_n, "rad/s")),
            ("Damping ratio", format_value(zeta)),
            ("Response type", response_type(zeta)),
        )
    )

    figure, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    axes[0].plot(t, capacitor_voltage, label="Capacitor voltage")
    axes[0].axhline(input_voltage, color="gray", linestyle=":", label="DC steady state")
    format_engineering_axes(
        axes[0],
        title="RLC Capacitor Voltage Step Response",
        ylabel="Voltage (V)",
    )

    axes[1].plot(t, current, color="tab:orange", label="Current")
    format_engineering_axes(
        axes[1],
        title="RLC Circuit Current",
        xlabel="Time (s)",
        ylabel="Current (A)",
    )
    show_figure(
        figure,
        "Voltage and current show the second-order transient behavior of the "
        "series RLC circuit.",
    )

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

    render_metric_row(
        (
            ("Final speed", format_value(speed[-1], "rad/s")),
            ("Final error", format_value(final_error, "rad/s")),
            ("Overshoot", format_value(overshoot, "%")),
        )
    )

    render_metric_row(
        (
            ("Peak speed", format_value(peak_speed, "rad/s")),
            (
                "Final speed in rpm",
                format_value(rad_per_sec_to_rpm(speed[-1]), "rpm"),
            ),
            ("Max voltage", format_value(np.max(voltage), "V")),
        )
    )

    figure, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 7))
    axes[0].plot(t, speed, label="Motor speed")
    axes[0].axhline(target_speed, color="gray", linestyle=":", label="Target speed")
    format_engineering_axes(
        axes[0],
        title="Discrete PID DC Motor Speed Control",
        ylabel="Speed (rad/s)",
    )

    axes[1].plot(t, voltage, color="tab:orange", label="Control voltage")
    axes[1].axhline(voltage_limit, color="gray", linestyle=":", label="Voltage limit")
    format_engineering_axes(
        axes[1],
        title="Held Voltage Command",
        ylabel="Voltage (V)",
    )

    axes[2].plot(t, current, color="tab:green", label="Armature current")
    format_engineering_axes(
        axes[2],
        title="Armature Current",
        xlabel="Time (s)",
        ylabel="Current (A)",
    )
    show_figure(
        figure,
        "The PID controller tracks motor speed while the voltage command and "
        "armature current show actuator effort.",
    )

    st.write(
        "The controller compares target speed with measured speed, computes a "
        "voltage command from proportional, integral, and derivative terms, "
        "then clips that command to the configured voltage limit."
    )


def render_heat_equation_1d():
    """Render the 1D heat equation simulation."""
    st.header("1D Heat Equation")
    render_info_box(
        "Diffusion on a rod",
        "Heat diffusion smooths temperature gradients. A localized hot pulse "
        "spreads along the rod while its peak temperature decays over time.",
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
    render_parameter_summary(
        (
            ("Length", format_value(length, "m")),
            ("Grid points", str(num_points)),
            ("alpha", format_value(alpha, precision=4)),
            ("Final time", format_value(t_final, "s")),
        )
    )
    st.subheader("Stability and numerical metrics")
    render_metric_row(
        (
            ("dx", format_value(dx, "m", 4)),
            ("Requested dt", format_value(dt, "s", 4)),
            ("Stability r", format_value(r, precision=4)),
            ("Limit", "<= 0.5000"),
        )
    )

    if not show_stability_status("Explicit heat stability r", r, 0.5):
        with st.expander("Why this was blocked"):
            st.write(
                "The explicit 1D heat scheme is stable only when "
                "`r = alpha*dt/dx^2 <= 0.5`. Reducing `dt`, reducing `alpha`, "
                "or using a coarser grid lowers the stability number."
            )
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

    render_metric_row(
        (
            ("Actual dt", format_value(result["dt"], "s", 4)),
            ("Initial max", format_value(np.max(temperature[0]))),
            ("Final max", format_value(np.max(temperature[-1]))),
            (
                "Peak decay",
                format_value(np.max(temperature[0]) - np.max(temperature[-1])),
            ),
        )
    )

    st.subheader("Main plot")
    x = result["x"]
    t = result["t"]
    figure, axes = plt.subplots(2, 1, figsize=(8, 7))
    for index in profile_indices(t, [0.0, 0.25, 0.5, 1.0]):
        axes[0].plot(x, temperature[index], label=f"t = {t[index]:.2f} s")
    format_engineering_axes(
        axes[0],
        title="Temperature Profiles",
        xlabel="Position x (m)",
        ylabel="Temperature",
    )

    heatmap = axes[1].imshow(
        temperature,
        aspect="auto",
        origin="lower",
        extent=[x[0], x[-1], t[0], t[-1]],
        cmap="inferno",
    )
    format_heatmap_axes(
        axes[1],
        title="Temperature History",
        xlabel="Position x (m)",
        ylabel="Time (s)",
    )
    add_clean_colorbar(figure, heatmap, axes[1], label="Temperature")
    show_figure(
        figure,
        "The upper plot compares temperature profiles over time; the heatmap "
        "shows diffusion across the rod history.",
    )
    render_info_box(
        "Engineering interpretation",
        "The hot pulse spreads because heat flows from high-temperature regions "
        "toward cooler neighboring points. The peak temperature decays while "
        "the profile becomes smoother.",
    )
    with st.expander("Limitations and assumptions"):
        st.write(
            "This page uses a constant thermal diffusivity, a uniform grid, and "
            "an explicit finite-difference scheme with simple boundary handling. "
            "The stability check only validates the implemented numerical "
            "scheme; it is not a material-model fidelity check."
        )


def render_wave_equation_1d():
    """Render the 1D wave equation simulation."""
    st.header("1D Wave Equation")
    render_info_box(
        "Propagation on a string",
        "A displacement pulse propagates through the domain, oscillates, and "
        "reflects from fixed boundaries. This is wave motion, not diffusion: "
        "the pulse travels and changes sign instead of simply smoothing out.",
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
    render_parameter_summary(
        (
            ("Length", format_value(length, "m")),
            ("Wave speed", format_value(c, "m/s")),
            ("Grid points", str(num_points)),
            ("Final time", format_value(t_final, "s")),
        )
    )
    st.subheader("CFL and numerical metrics")
    render_metric_row(
        (
            ("dx", format_value(dx, "m", 4)),
            ("Requested dt", format_value(dt, "s", 4)),
            ("CFL lambda", format_value(cfl, precision=4)),
            ("Limit", "<= 1.0000"),
        )
    )

    if not show_stability_status("Wave CFL lambda", cfl, 1.0):
        with st.expander("Why this was blocked"):
            st.write(
                "The explicit 1D wave scheme is stable only when "
                "`lambda = c*dt/dx <= 1`. Reducing `dt`, reducing wave speed, "
                "or using a coarser grid lowers the CFL number."
            )
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

    render_metric_row(
        (
            ("Actual dt", format_value(result["dt"], "s", 4)),
            ("Max |u|", format_value(np.max(np.abs(displacement)))),
            ("Final max", format_value(np.max(displacement[-1]))),
            ("Final min", format_value(np.min(displacement[-1]))),
        )
    )

    st.subheader("Main plot")
    x = result["x"]
    t = result["t"]
    figure, axes = plt.subplots(2, 1, figsize=(8, 7))
    for index in profile_indices(t, [0.0, 0.25, 0.5, 0.75, 1.0]):
        axes[0].plot(x, displacement[index], label=f"t = {t[index]:.2f} s")
    format_engineering_axes(
        axes[0],
        title="Displacement Profiles",
        xlabel="Position x (m)",
        ylabel="Displacement",
    )

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
    format_heatmap_axes(
        axes[1],
        title="Displacement History",
        xlabel="Position x (m)",
        ylabel="Time (s)",
    )
    add_clean_colorbar(figure, heatmap, axes[1], label="Displacement")
    show_figure(
        figure,
        "The line profiles show wave motion at selected times; the diverging "
        "heatmap is centered around zero displacement.",
    )
    render_info_box(
        "Engineering interpretation",
        "The displacement pulse moves through the string and reflects from the "
        "boundaries. Positive and negative displacement are shown symmetrically "
        "because oscillatory wave motion is centered around zero.",
    )
    with st.expander("Limitations and assumptions"):
        st.write(
            "This page uses a constant wave speed, a uniform grid, zero initial "
            "velocity, and an explicit finite-difference scheme. The CFL check "
            "guards numerical stability; it does not model damping, forcing, or "
            "material heterogeneity."
        )


def render_heat_equation_2d():
    """Render the 2D heat equation simulation."""
    st.header("2D Heat Equation")
    render_info_box(
        "Diffusion on a plate",
        "A hot spot diffuses across a rectangular plate. The peak temperature "
        "decays as the temperature field spreads outward from the initial "
        "localized source.",
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
    render_parameter_summary(
        (
            ("Plate", f"{width:.1f} x {height:.1f} m"),
            ("Grid", f"{grid_size} x {grid_size}"),
            ("alpha", format_value(alpha, precision=4)),
            ("Final time", format_value(t_final, "s")),
        )
    )
    st.subheader("Stability and numerical metrics")
    render_metric_row(
        (
            ("dx", format_value(dx, "m", 4)),
            ("dy", format_value(dy, "m", 4)),
            ("rx", format_value(rx, precision=4)),
            ("ry", format_value(ry, precision=4)),
        )
    )
    render_metric_row(
        (
            ("Requested dt", format_value(dt, "s", 4)),
            ("Stability rx + ry", format_value(stability_sum, precision=4)),
            ("Limit", "<= 0.5000"),
        )
    )

    if not show_stability_status("2D heat stability rx + ry", stability_sum, 0.5):
        with st.expander("Why this was blocked"):
            st.write(
                "The explicit 2D heat scheme is stable only when "
                "`rx + ry <= 0.5`, where `rx = alpha*dt/dx^2` and "
                "`ry = alpha*dt/dy^2`. Lower `dt`, lower `alpha`, or a coarser "
                "grid will reduce the stability sum."
            )
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

    render_metric_row(
        (
            ("Actual dt", format_value(result["dt"], "s", 4)),
            ("Stored frames", str(len(result["t"]))),
            ("Initial max", format_value(np.max(temperature[0]))),
            ("Final max", format_value(np.max(temperature[-1]))),
        )
    )

    st.subheader("Main plot")
    draw_2d_heat_plots(result)
    render_info_box(
        "Engineering interpretation",
        "The hot spot loses peak intensity as heat diffuses in both spatial "
        "directions. The shared color scale makes the decay from the initial "
        "maximum to the final snapshot easy to compare.",
    )
    with st.expander("Limitations, assumptions, and performance"):
        st.write(
            "This demo uses a square uniform grid, constant thermal diffusivity, "
            "and an explicit finite-difference update. Stored frames are capped "
            "so the Streamlit page remains responsive with the default 41 x 41 "
            "grid."
        )


def render_wave_equation_2d():
    """Render the 2D wave equation simulation."""
    st.header("2D Wave Equation")
    render_info_box(
        "Propagation on a membrane",
        "A membrane displacement propagates outward, oscillates, and reflects "
        "from fixed edges. This is not diffusion: displacement travels through "
        "the domain and changes sign around zero.",
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
    render_parameter_summary(
        (
            ("Membrane", f"{width:.1f} x {height:.1f} m"),
            ("Grid", f"{grid_size} x {grid_size}"),
            ("Wave speed", format_value(c, "m/s")),
            ("Final time", format_value(t_final, "s")),
        )
    )
    st.subheader("CFL and numerical metrics")
    render_metric_row(
        (
            ("lambda x", format_value(lambda_x, precision=4)),
            ("lambda y", format_value(lambda_y, precision=4)),
            ("rx", format_value(rx, precision=4)),
            ("ry", format_value(ry, precision=4)),
        )
    )
    render_metric_row(
        (
            ("Requested dt", format_value(dt, "s", 4)),
            ("Stability rx + ry", format_value(stability_sum, precision=4)),
            ("Limit", "<= 1.0000"),
        )
    )

    if not show_stability_status("2D wave stability rx + ry", stability_sum, 1.0):
        with st.expander("Why this was blocked"):
            st.write(
                "The explicit 2D wave scheme is stable only when "
                "`rx + ry <= 1`, where `rx = (c*dt/dx)^2` and "
                "`ry = (c*dt/dy)^2`. Lower `dt`, lower wave speed, or a coarser "
                "grid will reduce the stability sum."
            )
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

    render_metric_row(
        (
            ("Actual dt", format_value(result["dt"], "s", 4)),
            ("Stored frames", str(len(result["t"]))),
            ("Max |u|", format_value(np.max(np.abs(displacement)))),
            ("Final max", format_value(np.max(displacement[-1]))),
            ("Final min", format_value(np.min(displacement[-1]))),
        )
    )

    st.subheader("Main plot")
    draw_2d_wave_plots(result)
    render_info_box(
        "Engineering interpretation",
        "The displacement field remains oscillatory rather than smoothing out. "
        "The plot uses symmetric color limits around zero so positive and "
        "negative membrane motion have equal visual weight.",
    )
    with st.expander("Limitations, assumptions, and performance"):
        st.write(
            "This demo uses a square uniform grid, constant wave speed, zero "
            "initial velocity, fixed boundaries, and an explicit update. Stored "
            "frames are capped to keep the default 41 x 41 simulation responsive."
        )


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
        format_heatmap_axes(
            axis,
            title=f"Temperature at t = {t[index]:.2f} s",
            xlabel="x (m)",
            ylabel="y (m)",
        )
        set_equal_2d_axes(axis)
        add_clean_colorbar(figure, heatmap, axis, label="Temperature")

    centerline_axis = flat_axes[3]
    center_y_index = len(y) // 2
    for index in profile_indices(t, [0.0, 0.33, 0.67, 1.0]):
        centerline_axis.plot(
            x,
            temperature[index, center_y_index, :],
            label=f"t = {t[index]:.2f} s",
        )
    format_engineering_axes(
        centerline_axis,
        title="Centerline Temperature",
        xlabel="x (m)",
        ylabel="Temperature",
    )
    show_figure(
        figure,
        "Snapshots show the hot spot diffusing across the plate; the centerline "
        "plot compares temperature profiles through the plate midpoint.",
    )


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
        format_heatmap_axes(
            axis,
            title=f"Displacement at t = {t[index]:.2f} s",
            xlabel="x (m)",
            ylabel="y (m)",
        )
        set_equal_2d_axes(axis)
        add_clean_colorbar(figure, heatmap, axis, label="Displacement")

    centerline_axis = flat_axes[4]
    center_y_index = len(y) // 2
    for index in profile_indices(t, [0.0, 0.25, 0.5, 0.75, 1.0]):
        centerline_axis.plot(
            x,
            displacement[index, center_y_index, :],
            label=f"t = {t[index]:.2f} s",
        )
    format_engineering_axes(
        centerline_axis,
        title="Centerline Displacement",
        xlabel="x (m)",
        ylabel="Displacement",
    )
    figure.delaxes(flat_axes[5])
    show_figure(
        figure,
        "Wave snapshots use a symmetric diverging color scale so positive and "
        "negative membrane displacement are visually balanced.",
    )


def render_finite_difference_convergence():
    """Render finite-difference convergence analysis."""
    st.header("Finite Difference Methods")
    render_info_box(
        "Grid refinement and derivative error",
        "This demo differentiates `f(x) = sin(2*pi*x)` on successively finer "
        "uniform grids. Forward and backward differences are first-order; "
        "central differences are second-order in the interior.",
    )

    with st.sidebar:
        st.subheader("Convergence Inputs")
        max_points = st.slider("Finest grid points", 81, 641, 321, 80)
        frequency = st.slider("Sine frequency multiplier", 1, 4, 1, 1)

    render_parameter_summary(
        (
            ("Finest grid", str(max_points)),
            ("Frequency", f"{frequency}x"),
        )
    )

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

    st.subheader("Estimated convergence order")
    render_metric_row(
        (
            ("Forward difference", format_value(forward_order)),
            ("Backward difference", format_value(backward_order)),
            ("Central difference", format_value(central_order)),
        )
    )
    st.info(
        "Central difference is usually more accurate for smooth interior points "
        "because its leading truncation error scales with `dx^2`, while forward "
        "and backward differences scale with `dx`."
    )

    st.subheader("Main plot")
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
    format_engineering_axes(
        axis,
        title="Finite Difference Error Convergence",
        xlabel="Grid spacing dx",
        ylabel="RMS derivative error",
    )
    axis.grid(True, which="both", linestyle="--", linewidth=0.65, alpha=0.85)
    axis.invert_xaxis()
    show_figure(
        figure,
        "Log-log error trends show first-order forward/backward differences "
        "and second-order central differences.",
    )

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
    with st.expander("Limitations and assumptions"):
        st.write(
            "The convergence estimate uses smooth analytical data on uniform 1D "
            "grids. Boundary effects are avoided by comparing derivative errors "
            "on interior points only."
        )


def render_axial_bar_fem():
    """Render the 1D axial bar FEM demo."""
    st.header("1D FEM Axial Bar")
    render_info_box(
        "Fixed-free axial bar",
        "A fixed-free axial bar is split into linear finite elements. The app "
        "assembles the stiffness matrix, applies the fixed support, solves "
        "nodal displacements, and compares the tip displacement with "
        "`F*L/(E*A)`.",
    )

    with st.sidebar:
        st.subheader("FEM Inputs")
        length = st.slider("Bar length L (m)", 0.2, 5.0, 1.0, 0.1)
        youngs_modulus_gpa = st.slider("Young's modulus E (GPa)", 1.0, 250.0, 200.0, 1.0)
        area_mm2 = st.slider("Area A (mm^2)", 10.0, 1000.0, 100.0, 10.0)
        force = st.slider("End force F (N)", 100.0, 10000.0, 1000.0, 100.0)
        num_elements = st.slider("Number of elements", 1, 40, 8, 1)

    render_parameter_summary(
        (
            ("Length", format_value(length, "m")),
            ("Young's modulus", format_value(youngs_modulus_gpa, "GPa")),
            ("Area", format_value(area_mm2, "mm^2")),
            ("Applied force", format_value(force, "N")),
            ("Elements", str(num_elements)),
        )
    )

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

    st.subheader("FEM solution metrics")
    render_metric_row(
        (
            (
                "FEM tip displacement",
                format_value(result["tip_displacement"], "m", 6),
            ),
            (
                "Analytical tip",
                format_value(result["analytical_tip_displacement"], "m", 6),
            ),
            ("Relative error", f"{result['relative_tip_error']:.3e}"),
            ("Reaction force", format_value(fixed_reaction, "N")),
            ("Average stress", format_value(average_stress / 1e6, "MPa")),
        )
    )

    element_centers = 0.5 * (nodes[elements[:, 0]] + nodes[elements[:, 1]])
    max_displacement = max(float(np.max(np.abs(displacements))), 1e-15)
    scale = 0.1 * length / max_displacement
    deformed_nodes = nodes + scale * displacements

    st.subheader("Main plot")
    figure, axes = plt.subplots(3, 1, figsize=(8, 9))
    axes[0].plot(nodes, displacements, "o-", label="FEM displacement")
    axes[0].plot(nodes, analytical, "--", label="Analytical displacement")
    format_engineering_axes(
        axes[0],
        title="Nodal Displacement",
        xlabel="Position x (m)",
        ylabel="Displacement u (m)",
    )

    axes[1].step(element_centers, stresses / 1e6, where="mid", label="Element stress")
    axes[1].axhline(
        force / A / 1e6,
        color="tab:orange",
        linestyle="--",
        label="Analytical stress",
    )
    format_engineering_axes(
        axes[1],
        title="Element Stress",
        xlabel="Element center x (m)",
        ylabel="Stress (MPa)",
    )

    axes[2].plot(nodes, np.zeros_like(nodes), "o-", label="Original bar")
    axes[2].plot(
        deformed_nodes,
        np.zeros_like(deformed_nodes) + 0.02,
        "o-",
        label=f"Deformed shape ({scale:.2e}x)",
    )
    axes[2].set_yticks([])
    format_engineering_axes(
        axes[2],
        title="Exaggerated Deformed Shape",
        xlabel="Position x (m)",
        ylabel=None,
    )
    axes[2].grid(True, axis="x", linestyle="--", linewidth=0.65, alpha=0.85)
    show_figure(
        figure,
        "The FEM plots compare displacement with the analytical solution, show "
        "element stress in MPa, and exaggerate the deformed shape for visibility.",
    )

    render_info_box(
        "Engineering interpretation",
        "FEM assembles element stiffness matrices into a global system, applies "
        "the fixed displacement boundary condition, solves nodal displacements, "
        "and recovers element stress. For this uniform axial bar with a tip "
        "load, the linear FEM solution matches the analytical linear "
        "displacement field up to numerical roundoff.",
    )
    with st.expander("Limitations and assumptions"):
        st.write(
            "This is a small-displacement, linear-elastic, static 1D axial bar "
            "model with constant `E` and `A`. It does not include beam bending, "
            "2D trusses, nonlinear materials, buckling, dynamics, or contact."
        )


def main():
    """Run the Streamlit application."""
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=":gear:",
        layout="wide",
    )
    apply_engineering_plot_style()
    inject_custom_css()

    st.sidebar.title(APP_TITLE)
    st.sidebar.caption("Interactive engineering simulations and portfolio examples.")
    domain = st.sidebar.radio("Section", tuple(DOMAIN_DEMOS.keys()))
    demo = st.sidebar.selectbox("Demo", DOMAIN_DEMOS[domain])

    render_header(domain)

    if domain == "Home":
        render_home()
    elif domain == "About":
        render_about()
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
