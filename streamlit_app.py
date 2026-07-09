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
from analysis.kalman_filter import KalmanFilter, discretize_state_space
from analysis.particle_filter import ParticleFilter
from analysis.quadcopter_altitude_control import (
    simulate_altitude_pid_control,
    summarize_altitude_response,
)
from analysis.quadcopter_obstacle_avoidance import (
    SphericalObstacle,
    simulate_quadcopter_obstacle_avoidance,
)
from analysis.quadcopter_trajectory_tracking import (
    circular_trajectory,
    hover_trajectory,
    simulate_quadcopter_trajectory_tracking,
)
from analysis.quadcopter_waypoint_following import (
    simulate_quadcopter_waypoint_following,
    waypoint_trajectory,
)
from analysis.state_space import dc_motor_state_space
from analysis.unscented_kalman_filter import UnscentedKalmanFilter
from examples.run_particle_filter_pendulum import (
    _rk4_pendulum_step as pf_rk4_pendulum_step,
)
from examples.run_particle_filter_pendulum import (
    _simulate_true_pendulum as simulate_pf_true_pendulum,
)
from examples.run_ukf_pendulum import (
    _pendulum_measurement as ukf_pendulum_measurement,
)
from examples.run_ukf_pendulum import (
    _rk4_pendulum_step as ukf_rk4_pendulum_step,
)
from examples.run_ukf_pendulum import (
    _simulate_true_pendulum as simulate_ukf_true_pendulum,
)
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
    create_streamlit_subplots,
    display_streamlit_figure,
    format_engineering_axes,
    format_heatmap_axes,
    place_legend_outside,
    set_equal_2d_axes,
    set_xy_plot_limits_with_margin,
)

APP_TITLE = "Engineering Simulation Toolkit"
APP_SUBTITLE = (
    "Python scientific computing, control systems, PDEs, FEM, and UAV simulation"
)

DOMAIN_DEMOS = {
    "Home": ("Overview",),
    "Control Systems": ("RC Circuit", "RLC Circuit", "DC Motor PID Control"),
    "State Estimation": ("Interactive Filters",),
    "UAV / Quadcopter": ("UAV / Quadcopter Simulation",),
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
        "Interactive Kalman, Unscented Kalman, and Particle Filter demos for "
        "reconstructing hidden states from noisy measurements.",
        "Interactive demos",
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


def render_page_header(section=None):
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


def render_metric_row(metrics, columns=None):
    """Render Streamlit metrics with consistent spacing."""
    metrics = tuple(metrics)
    if not metrics:
        return

    column_count = columns or len(metrics)
    metric_columns = st.columns(column_count)
    for index, metric in enumerate(metrics):
        label, value = metric[:2]
        delta = metric[2] if len(metric) > 2 else None
        metric_columns[index % column_count].metric(label, value, delta=delta)


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


def render_assumptions_expander(body, title="Limitations and assumptions"):
    """Render a standard explanatory expander without changing copy."""
    with st.expander(title):
        st.write(body)


def render_parameter_summary(parameters):
    """Render selected user inputs as a compact metric row."""
    st.caption("Selected setup")
    render_metric_row(parameters)


def safe_display_dataframe(data, **kwargs):
    """Display tabular data with current Streamlit stretch-width semantics."""
    kwargs.setdefault("width", "stretch")
    return st.dataframe(data, **kwargs)


def show_figure(figure, caption=None):
    """Display a Matplotlib figure in Streamlit, then close it."""
    display_streamlit_figure(figure, caption=caption)


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
        for column, card in zip(
            columns,
            FEATURE_CARDS[row_start : row_start + 3],
            strict=False,
        ):
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
            "Simplified 6-DOF quadcopter model",
            "Trajectory tracking, waypoint following, and obstacle avoidance",
        ),
    }

    columns = st.columns(3)
    for column, item in zip(columns, examples_by_domain.get(domain, ()), strict=False):
        with column:
            render_feature_card(item, "Available through tested modules, examples, and saved figures.")


def render_metric_grid(metrics, columns=2):
    """Render compact metrics inside a constrained column."""
    render_metric_row(metrics, columns=columns)


def render_inactive_simulation_hint(demo_name):
    """Tell users why an inactive eager-rendered tab has no metrics yet."""
    st.info(f"Select {demo_name} as the active simulation to run it.")


def settling_time_label(settling_time):
    """Return a readable settling-time label."""
    if settling_time is None:
        return "Not settled"

    return format_value(settling_time, "s")


@st.cache_data(show_spinner=False)
def run_uav_altitude_demo(
    target_altitude,
    mass,
    Kp,
    Ki,
    Kd,
    duration,
    dt,
):
    """Run the existing altitude PID helper for Streamlit."""
    result = simulate_altitude_pid_control(
        target_altitude=target_altitude,
        t_span=(0.0, duration),
        dt=dt,
        m=mass,
        Kp=Kp,
        Ki=Ki,
        Kd=Kd,
        thrust_limits=(0.0, 25.0),
    )
    metrics = summarize_altitude_response(result)
    return result, metrics


@st.cache_data(show_spinner=False)
def run_uav_trajectory_demo(
    trajectory_kind,
    radius,
    altitude,
    angular_speed,
    duration,
    dt,
):
    """Run the existing 6-DOF trajectory tracker for Streamlit."""
    initial_state = np.zeros(12)
    if trajectory_kind == "Hover":
        trajectory_func = hover_trajectory((0.0, 0.0, altitude))
        initial_state[0:3] = [0.45, -0.35, 0.0]
    else:
        trajectory_func = circular_trajectory(
            radius=radius,
            altitude=altitude,
            angular_speed=angular_speed,
        )
        initial_state[0:3] = [radius, 0.0, altitude]

    return simulate_quadcopter_trajectory_tracking(
        trajectory_func,
        initial_state=initial_state,
        t_span=(0.0, duration),
        dt=dt,
    )


@st.cache_data(show_spinner=False)
def run_uav_waypoint_demo(altitude, segment_time, smoothing, dt):
    """Run the existing waypoint-following helper for Streamlit."""
    waypoints = np.array(
        [
            [0.0, 0.0, altitude],
            [1.0, 0.0, altitude + 0.35],
            [1.0, 1.0, altitude + 0.15],
            [0.2, 1.2, altitude + 0.45],
        ]
    )
    return simulate_quadcopter_waypoint_following(
        waypoints,
        segment_time=segment_time,
        smoothing=smoothing,
        dt=dt,
        hold_time=1.0,
    )


@st.cache_data(show_spinner=False)
def run_uav_obstacle_demo(
    altitude,
    duration,
    obstacle_y,
    obstacle_radius,
    influence_radius,
    avoidance_gain,
    max_avoidance_acceleration,
    dt,
):
    """Run the existing static obstacle-avoidance helper for Streamlit."""
    waypoints = np.array(
        [
            [0.0, 0.0, altitude],
            [2.0, 0.0, altitude],
        ]
    )
    obstacle = SphericalObstacle(
        center=np.array([1.0, obstacle_y, altitude]),
        radius=obstacle_radius,
        influence_radius=influence_radius,
    )
    initial_state = np.zeros(12)
    initial_state[0:3] = waypoints[0]
    trajectory_func = waypoint_trajectory(
        waypoints,
        segment_time=max(0.5, duration - 1.0),
        smoothing="smoothstep",
    )

    result = simulate_quadcopter_obstacle_avoidance(
        trajectory_func,
        [obstacle],
        initial_state=initial_state,
        t_span=(0.0, duration),
        dt=dt,
        Kd_pos=(2.0, 2.0, 3.2),
        Kp_att=(0.25, 0.25, 0.16),
        Kd_att=(0.12, 0.12, 0.08),
        avoidance_gain=avoidance_gain,
        max_avoidance_acceleration=max_avoidance_acceleration,
    )
    result["waypoints"] = waypoints
    return result


def plot_uav_altitude_response(result):
    """Plot altitude tracking and thrust command."""
    time = result["time"]
    altitude = result["altitude"]
    thrust = result["thrust"]
    target = result["target_altitude"]
    hover_thrust_value = result["hover_thrust"]

    figure, axes = create_streamlit_subplots(2, 1, width=9, height=6.8, sharex=True)
    axes[0].plot(time, altitude, label="Altitude")
    axes[0].axhline(target, color="tab:orange", linestyle="--", label="Target")
    format_engineering_axes(
        axes[0],
        title="Altitude PID Tracking",
        ylabel="Altitude z (m)",
    )

    axes[1].plot(time, thrust, color="tab:green", label="Thrust command")
    axes[1].axhline(
        hover_thrust_value,
        color="gray",
        linestyle=":",
        label="Hover thrust",
    )
    format_engineering_axes(
        axes[1],
        title="Control Effort",
        xlabel="Time (s)",
        ylabel="Thrust (N)",
    )
    return figure


def plot_uav_tracking_response(result, title):
    """Plot XY path and tracking error for a 6-DOF result."""
    time = result["time"]
    positions = result["states"][:, 0:3]
    references = result["reference_positions"]
    error_norm = result["tracking_error_norm"]

    figure, axes = create_streamlit_subplots(2, 1, width=9, height=7.2)
    axes[0].plot(references[:, 0], references[:, 1], "--", label="Reference")
    axes[0].plot(positions[:, 0], positions[:, 1], label="Actual")
    axes[0].scatter(
        references[0, 0],
        references[0, 1],
        color="tab:green",
        s=35,
        label="Start",
        zorder=3,
    )
    axes[0].scatter(
        references[-1, 0],
        references[-1, 1],
        color="tab:red",
        s=35,
        label="Final reference",
        zorder=3,
    )
    format_engineering_axes(
        axes[0],
        title=f"{title} XY Path",
        xlabel="x (m)",
        ylabel="y (m)",
    )
    set_xy_plot_limits_with_margin(
        axes[0],
        np.concatenate((references[:, 0], positions[:, 0])),
        np.concatenate((references[:, 1], positions[:, 1])),
        margin_fraction=0.12,
    )

    axes[1].plot(time, error_norm, color="tab:red", label="Position error norm")
    axes[1].plot(
        time,
        positions[:, 2] - references[:, 2],
        color="tab:purple",
        linestyle="--",
        label="z error",
    )
    axes[1].axhline(0.0, color="gray", linestyle=":")
    format_engineering_axes(
        axes[1],
        title="Tracking Error",
        xlabel="Time (s)",
        ylabel="Error (m)",
    )
    return figure


def place_uav_xy_legend(axis):
    """Place UAV XY plot legends outside the data area."""
    place_legend_outside(axis, location="bottom", ncol=3)


def plot_uav_waypoint_response(result):
    """Plot waypoint-following path and position error."""
    figure = plot_uav_tracking_response(result, "Waypoint Following")
    axis = figure.axes[0]
    waypoints = result["waypoints"]
    axis.plot(
        waypoints[:, 0],
        waypoints[:, 1],
        "o:",
        color="tab:orange",
        label="Waypoints",
    )
    positions = result["states"][:, 0:3]
    references = result["reference_positions"]
    set_xy_plot_limits_with_margin(
        axis,
        np.concatenate((references[:, 0], positions[:, 0], waypoints[:, 0])),
        np.concatenate((references[:, 1], positions[:, 1], waypoints[:, 1])),
        margin_fraction=0.12,
    )
    place_uav_xy_legend(axis)
    return figure


def plot_uav_obstacle_response(result):
    """Plot obstacle-avoidance path, clearance, and avoidance command."""
    time = result["time"]
    positions = result["states"][:, 0:3]
    references = result["reference_positions"]
    clearances = result["nearest_clearances"]
    avoidance_norm = result["avoidance_acceleration_norm"]
    obstacle = result["obstacles"][0]
    waypoints = result["waypoints"]

    figure, axes = create_streamlit_subplots(3, 1, width=9.5, height=9.5)
    axes[0].plot(references[:, 0], references[:, 1], "--", label="Reference")
    axes[0].plot(positions[:, 0], positions[:, 1], label="Actual")
    axes[0].plot(
        waypoints[:, 0],
        waypoints[:, 1],
        "o:",
        color="tab:green",
        label="Waypoints",
    )
    obstacle_patch = plt.Circle(
        (obstacle.center[0], obstacle.center[1]),
        obstacle.radius,
        color="tab:red",
        alpha=0.22,
        label="Obstacle",
    )
    influence_patch = plt.Circle(
        (obstacle.center[0], obstacle.center[1]),
        obstacle.influence_radius,
        color="tab:red",
        alpha=0.08,
        label="Influence radius",
    )
    axes[0].add_patch(influence_patch)
    axes[0].add_patch(obstacle_patch)
    format_engineering_axes(
        axes[0],
        title="Obstacle Avoidance XY Projection",
        xlabel="x (m)",
        ylabel="y (m)",
    )
    obstacle_x = np.array(
        [
            obstacle.center[0] - obstacle.influence_radius,
            obstacle.center[0] + obstacle.influence_radius,
        ]
    )
    obstacle_y = np.array(
        [
            obstacle.center[1] - obstacle.influence_radius,
            obstacle.center[1] + obstacle.influence_radius,
        ]
    )
    set_xy_plot_limits_with_margin(
        axes[0],
        np.concatenate((references[:, 0], positions[:, 0], waypoints[:, 0], obstacle_x)),
        np.concatenate((references[:, 1], positions[:, 1], waypoints[:, 1], obstacle_y)),
        margin_fraction=0.12,
    )
    place_uav_xy_legend(axes[0])

    axes[1].plot(time, clearances, color="tab:red", label="Nearest clearance")
    axes[1].axhline(0.0, color="black", linestyle=":", label="Obstacle surface")
    format_engineering_axes(
        axes[1],
        title="Obstacle Clearance",
        ylabel="Clearance (m)",
    )

    axes[2].plot(
        time,
        avoidance_norm,
        color="tab:purple",
        label="Avoidance acceleration",
    )
    format_engineering_axes(
        axes[2],
        title="Repulsive Avoidance Command",
        xlabel="Time (s)",
        ylabel="Acceleration (m/s^2)",
    )
    return figure


def render_uav_altitude_tab(is_active):
    """Render the altitude-control Streamlit tab."""
    st.subheader("Altitude Control Demo")
    controls, metrics_column = st.columns([1, 1])
    with controls:
        st.caption("Controls")
        target_altitude = st.slider(
            "Target altitude (m)",
            min_value=0.5,
            max_value=8.0,
            value=3.0,
            step=0.25,
            key="uav_altitude_target",
        )
        mass = st.slider("Mass (kg)", 0.6, 1.8, 1.0, 0.1, key="uav_altitude_mass")
        Kp = st.slider("PID Kp", 0.0, 8.0, 4.0, 0.25, key="uav_altitude_kp")
        Ki = st.slider("PID Ki", 0.0, 3.0, 1.0, 0.1, key="uav_altitude_ki")
        Kd = st.slider("PID Kd", 0.0, 6.0, 3.0, 0.25, key="uav_altitude_kd")
        duration = st.slider(
            "Simulation duration (s)",
            4.0,
            12.0,
            8.0,
            1.0,
            key="uav_altitude_duration",
        )
        st.caption("Fixed sample time: 0.02 s; thrust is saturated at 25 N.")

    with metrics_column:
        st.caption("Metrics")
        if not is_active:
            render_inactive_simulation_hint("Altitude Control")
            return

        result, metrics = run_uav_altitude_demo(
            target_altitude,
            mass,
            Kp,
            Ki,
            Kd,
            duration,
            0.02,
        )
        render_metric_grid(
            (
                ("Final altitude", format_value(metrics.final_altitude, "m")),
                ("Final error", format_value(metrics.final_error, "m")),
                ("Overshoot", format_value(metrics.overshoot_percent, "%", 2)),
                ("Settling time", settling_time_label(metrics.settling_time)),
                ("Max thrust", format_value(metrics.max_thrust, "N")),
                ("Min thrust", format_value(metrics.min_thrust, "N")),
            )
        )

    figure = plot_uav_altitude_response(result)
    show_figure(
        figure,
        "Altitude is simulated with the existing 1D quadcopter altitude plant "
        "and discrete PID controller; thrust is shown as the control effort.",
    )


def render_uav_trajectory_tab(is_active):
    """Render the trajectory-tracking Streamlit tab."""
    st.subheader("Trajectory Tracking Demo")
    controls, metrics_column = st.columns([1, 1])
    with controls:
        st.caption("Controls")
        trajectory_kind = st.radio(
            "Reference trajectory",
            ("Hover", "Circle"),
            horizontal=True,
            key="uav_trajectory_kind",
        )
        radius = st.slider(
            "Circle radius (m)",
            0.5,
            2.0,
            1.0,
            0.1,
            key="uav_trajectory_radius",
        )
        altitude = st.slider(
            "Reference altitude (m)",
            1.0,
            4.0,
            2.0,
            0.25,
            key="uav_trajectory_altitude",
        )
        angular_speed = st.slider(
            "Angular speed (rad/s)",
            0.1,
            0.6,
            0.3,
            0.05,
            key="uav_trajectory_angular_speed",
        )
        duration = st.slider(
            "Simulation duration (s)",
            4.0,
            14.0,
            8.0,
            1.0,
            key="uav_trajectory_duration",
        )
        st.caption("Fixed sample time: 0.04 s for responsive simplified 6-DOF simulation.")

    with metrics_column:
        st.caption("Metrics")
        if not is_active:
            render_inactive_simulation_hint("Trajectory Tracking")
            return

        result = run_uav_trajectory_demo(
            trajectory_kind,
            radius,
            altitude,
            angular_speed,
            duration,
            0.04,
        )
        metrics = result["tracking_metrics"]
        render_metric_grid(
            (
                (
                    "Final position error",
                    format_value(metrics["final_position_error_norm"], "m"),
                ),
                ("RMS position error", format_value(metrics["rms_position_error"], "m")),
                ("Max position error", format_value(metrics["max_position_error"], "m")),
                ("Max thrust", format_value(metrics["max_thrust"], "N")),
                ("Max torque", format_value(metrics["max_abs_torque"], "N*m", 4)),
            )
        )

    figure = plot_uav_tracking_response(result, f"{trajectory_kind} Tracking")
    show_figure(
        figure,
        "The path and error plots use the repository's cascaded PD tracker "
        "around the simplified 6-DOF quadcopter model.",
    )


def render_uav_waypoint_tab(is_active):
    """Render the waypoint-following Streamlit tab."""
    st.subheader("Waypoint Following Demo")
    controls, metrics_column = st.columns([1, 1])
    with controls:
        st.caption("Controls")
        altitude = st.slider(
            "Base waypoint altitude (m)",
            0.8,
            3.0,
            1.2,
            0.1,
            key="uav_waypoint_altitude",
        )
        segment_time = st.slider(
            "Segment time (s)",
            2.0,
            5.0,
            3.0,
            0.5,
            key="uav_waypoint_segment_time",
        )
        smoothing = st.radio(
            "Waypoint interpolation",
            ("smoothstep", "linear"),
            horizontal=True,
            key="uav_waypoint_smoothing",
        )
        st.caption(
            "A predefined four-waypoint path keeps this tab lightweight while "
            "still exercising the waypoint helper."
        )

    with metrics_column:
        st.caption("Metrics")
        if not is_active:
            render_inactive_simulation_hint("Waypoint Following")
            return

        result = run_uav_waypoint_demo(altitude, segment_time, smoothing, 0.05)
        metrics = result["waypoint_metrics"]
        render_metric_grid(
            (
                ("Final waypoint error", format_value(metrics["final_waypoint_error"], "m")),
                ("RMS position error", format_value(metrics["rms_position_error"], "m")),
                ("Max position error", format_value(metrics["max_position_error"], "m")),
                ("Waypoints", str(metrics["number_of_waypoints"])),
                ("Max thrust", format_value(metrics["max_thrust"], "N")),
                ("Max torque", format_value(metrics["max_abs_torque"], "N*m", 4)),
            )
        )

    figure = plot_uav_waypoint_response(result)
    show_figure(
        figure,
        "The waypoint follower converts fixed 3D waypoints into a time-based "
        "reference and reuses the 6-DOF trajectory tracker.",
    )


def render_uav_obstacle_tab(is_active):
    """Render the obstacle-avoidance Streamlit tab."""
    st.subheader("Obstacle Avoidance Demo")
    controls, metrics_column = st.columns([1, 1])
    with controls:
        st.caption("Controls")
        altitude = st.slider(
            "Path altitude (m)",
            1.0,
            3.0,
            1.5,
            0.1,
            key="uav_obstacle_altitude",
        )
        duration = st.slider(
            "Simulation duration (s)",
            5.0,
            10.0,
            8.0,
            0.5,
            key="uav_obstacle_duration",
        )
        obstacle_y = st.slider(
            "Obstacle y offset (m)",
            0.0,
            0.6,
            0.28,
            0.02,
            key="uav_obstacle_y",
        )
        obstacle_radius = st.slider(
            "Obstacle radius (m)",
            0.15,
            0.4,
            0.25,
            0.01,
            key="uav_obstacle_radius",
        )
        influence_radius = st.slider(
            "Influence radius (m)",
            0.6,
            1.4,
            1.0,
            0.05,
            key="uav_obstacle_influence",
        )
        avoidance_gain = st.slider(
            "Avoidance gain",
            0.0,
            0.08,
            0.03,
            0.005,
            key="uav_obstacle_gain",
        )
        max_avoidance_acceleration = st.slider(
            "Max avoidance acceleration (m/s^2)",
            0.5,
            3.0,
            1.5,
            0.25,
            key="uav_obstacle_max_accel",
        )
        st.caption("Fixed sample time: 0.04 s; obstacle is a static sphere.")

    with metrics_column:
        st.caption("Metrics")
        if not is_active:
            render_inactive_simulation_hint("Obstacle Avoidance")
            return

        result = run_uav_obstacle_demo(
            altitude,
            duration,
            obstacle_y,
            obstacle_radius,
            influence_radius,
            avoidance_gain,
            max_avoidance_acceleration,
            0.04,
        )
        metrics = result["obstacle_metrics"]
        render_metric_grid(
            (
                (
                    "Final position error",
                    format_value(metrics["final_position_error_norm"], "m"),
                ),
                ("Minimum clearance", format_value(metrics["min_clearance"], "m")),
                ("RMS position error", format_value(metrics["rms_position_error"], "m")),
                (
                    "Max avoidance accel.",
                    format_value(metrics["max_avoidance_acceleration"], "m/s^2"),
                ),
                ("Max thrust", format_value(metrics["max_thrust"], "N")),
                ("Max torque", format_value(metrics["max_abs_torque"], "N*m", 4)),
            )
        )

    figure = plot_uav_obstacle_response(result)
    show_figure(
        figure,
        "The obstacle projection shows the static obstacle and its influence "
        "radius; clearance and avoidance acceleration are plotted over time.",
    )


def render_uav_quadcopter():
    """Render interactive UAV and quadcopter demos."""
    st.title("UAV / Quadcopter Simulation")
    st.write(
        "This section demonstrates dynamic simulation, control, trajectory "
        "tracking, and obstacle avoidance for simplified quadcopter models."
    )
    st.info(
        "These are simplified educational and prototype simulations, not "
        "flight-ready UAV control software."
    )

    tab_names = (
        "Altitude Control",
        "Trajectory Tracking",
        "Waypoint Following",
        "Obstacle Avoidance",
    )
    active_demo = st.radio(
        "Active simulation",
        tab_names,
        horizontal=True,
        key="uav_active_demo",
        help=(
            "Streamlit renders tab contents eagerly, so this selector keeps the "
            "page responsive by running only one UAV simulation at a time."
        ),
    )

    altitude_tab, trajectory_tab, waypoint_tab, obstacle_tab = st.tabs(tab_names)
    with altitude_tab:
        render_uav_altitude_tab(active_demo == "Altitude Control")
    with trajectory_tab:
        render_uav_trajectory_tab(active_demo == "Trajectory Tracking")
    with waypoint_tab:
        render_uav_waypoint_tab(active_demo == "Waypoint Following")
    with obstacle_tab:
        render_uav_obstacle_tab(active_demo == "Obstacle Avoidance")

    render_assumptions_expander(
        "The altitude demo uses the existing 1D vertical model and PID "
        "controller. The trajectory, waypoint, and obstacle demos use the "
        "repository's simplified 6-DOF model and cascaded educational "
        "tracking controller. The obstacle helper is local reactive "
        "avoidance around static spherical obstacles; it is not global path "
        "planning, certified guidance, rotor-level motor mixing, hardware "
        "validation, or real-world deployment software.",
        title="Assumptions and limitations",
    )


def _simulate_discrete_linear_system(A, B, x0, input_value, num_points):
    """Simulate a small discrete linear system for estimator demos."""
    states = np.zeros((num_points, len(x0)))
    states[0] = np.asarray(x0, dtype=float)
    input_vector = np.asarray([input_value], dtype=float)

    for index in range(num_points - 1):
        states[index + 1] = A @ states[index] + B @ input_vector

    return states


@st.cache_data(show_spinner=False)
def run_linear_kalman_motor_demo(
    voltage,
    measurement_noise_std,
    process_noise_scale,
    simulation_time,
    sample_time,
):
    """Run the existing linear Kalman filter on the DC motor state-space model."""
    R = 1.0
    L = 0.5
    J = 0.01
    b = 0.001
    Kt = 0.01
    Ke = 0.01
    num_points = int(simulation_time / sample_time) + 1
    rng = np.random.default_rng(0)

    A, B, _, _ = dc_motor_state_space(R, L, J, b, Kt, Ke)
    A_d, B_d = discretize_state_space(A, B, sample_time)
    measurement_matrix = np.array([[0.0, 1.0]])

    time = np.linspace(0.0, simulation_time, num_points)
    true_states = _simulate_discrete_linear_system(
        A_d,
        B_d,
        [0.0, 0.0],
        voltage,
        num_points,
    )
    measured_speed = true_states[:, 1] + rng.normal(
        0.0,
        measurement_noise_std,
        num_points,
    )

    kalman_filter = KalmanFilter(
        A=A_d,
        B=B_d,
        C=measurement_matrix,
        Q=process_noise_scale * np.diag([1e-4, 1e-3]),
        R=np.array([[measurement_noise_std**2]]),
        x_hat=np.array([0.0, 0.0]),
        P=np.diag([10.0, 100.0]),
        name="DC motor speed Kalman filter",
    )

    estimates = np.zeros_like(true_states)
    estimates[0], _ = kalman_filter.update(measured_speed[0], voltage)
    for index in range(1, num_points):
        estimates[index], _ = kalman_filter.step(measured_speed[index], voltage)

    errors = true_states - estimates

    return {
        "time": time,
        "true_current": true_states[:, 0],
        "true_speed": true_states[:, 1],
        "measured_speed": measured_speed,
        "estimated_current": estimates[:, 0],
        "estimated_speed": estimates[:, 1],
        "current_error": errors[:, 0],
        "speed_error": errors[:, 1],
        "measurement_noise_std": measurement_noise_std,
        "process_noise_scale": process_noise_scale,
    }


@st.cache_data(show_spinner=False)
def run_ukf_pendulum_demo(
    initial_angle_deg,
    measurement_noise_deg,
    process_noise_scale,
    initial_estimate_offset_deg,
    simulation_time,
):
    """Run the existing UKF class on the pendulum example dynamics."""
    L = 1.0
    g = 9.81
    damping = 0.08
    sample_time = 0.02
    num_points = int(simulation_time / sample_time) + 1
    time = np.linspace(0.0, simulation_time, num_points)
    rng = np.random.default_rng(4)
    measurement_noise_rad = np.radians(measurement_noise_deg)
    initial_state = np.array([np.radians(initial_angle_deg), 0.0])

    true_states = simulate_ukf_true_pendulum(
        initial_state,
        sample_time,
        num_points,
        L,
        g,
        damping,
    )
    measured_theta = true_states[:, 0] + rng.normal(
        0.0,
        measurement_noise_rad,
        len(time),
    )
    initial_estimate = np.array(
        [np.radians(initial_angle_deg + initial_estimate_offset_deg), 0.0]
    )

    ukf = UnscentedKalmanFilter(
        x0=initial_estimate,
        P0=np.diag([0.1, 1.0]),
        Q=process_noise_scale * np.diag([1e-7, 1e-5]),
        R=np.array([[measurement_noise_rad**2]]),
        process_model=lambda x, dt: ukf_rk4_pendulum_step(x, dt, L, g, damping),
        measurement_model=ukf_pendulum_measurement,
        dt=sample_time,
        alpha=1e-3,
        beta=2.0,
        kappa=0.0,
    )

    estimates = np.zeros_like(true_states)
    estimates[0] = ukf.update(measured_theta[0])
    for index in range(1, len(time)):
        estimates[index] = ukf.step(measured_theta[index])

    errors = true_states - estimates

    return {
        "time": time,
        "true_theta": true_states[:, 0],
        "true_omega": true_states[:, 1],
        "measured_theta": measured_theta,
        "estimated_theta": estimates[:, 0],
        "estimated_omega": estimates[:, 1],
        "angle_error": errors[:, 0],
        "omega_error": errors[:, 1],
        "measurement_noise_deg": measurement_noise_deg,
        "process_noise_scale": process_noise_scale,
        "sample_time": sample_time,
    }


@st.cache_data(show_spinner=False)
def run_particle_filter_pendulum_demo(
    initial_angle_deg,
    measurement_noise_deg,
    num_particles,
    simulation_time,
):
    """Run the existing ParticleFilter class on the pendulum example dynamics."""
    L = 1.0
    g = 9.81
    damping = 0.08
    sample_time = 0.02
    num_points = int(simulation_time / sample_time) + 1
    time = np.linspace(0.0, simulation_time, num_points)
    rng = np.random.default_rng(5)
    measurement_noise_rad = np.radians(measurement_noise_deg)
    process_noise_std = np.array([np.radians(0.04), 0.015])
    initial_state = np.array([np.radians(initial_angle_deg), 0.0])

    true_states = simulate_pf_true_pendulum(
        initial_state,
        sample_time,
        num_points,
        L,
        g,
        damping,
    )
    measured_theta = true_states[:, 0] + rng.normal(
        0.0,
        measurement_noise_rad,
        len(time),
    )
    particles = np.column_stack(
        [
            rng.normal(
                np.radians(initial_angle_deg - 10.0),
                np.radians(10.0),
                num_particles,
            ),
            rng.normal(0.0, 0.5, num_particles),
        ]
    )

    def process_model(particle_states, dt):
        return pf_rk4_pendulum_step(particle_states, dt, L, g, damping)

    def process_noise_sampler(n_particles, state_dim, random_generator):
        return random_generator.normal(
            0.0,
            process_noise_std,
            size=(n_particles, state_dim),
        )

    def measurement_likelihood(measurement, particle_states):
        measured_angle = np.asarray(measurement, dtype=float).reshape(-1)[0]
        angle_error = measured_angle - particle_states[:, 0]
        return np.exp(-0.5 * (angle_error / measurement_noise_rad) ** 2)

    particle_filter = ParticleFilter(
        particles=particles,
        process_model=process_model,
        measurement_likelihood=measurement_likelihood,
        process_noise_sampler=process_noise_sampler,
        rng=rng,
    )

    estimates = np.zeros_like(true_states)
    effective_sample_sizes = np.zeros(len(time))
    resample_threshold = 0.5

    particle_filter.update(measured_theta[0])
    effective_sample_sizes[0] = particle_filter.effective_sample_size()
    if effective_sample_sizes[0] < resample_threshold * num_particles:
        particle_filter.systematic_resample()
    estimates[0] = particle_filter.estimate()

    for index in range(1, len(time)):
        particle_filter.predict(sample_time)
        particle_filter.update(measured_theta[index])
        effective_sample_sizes[index] = particle_filter.effective_sample_size()
        if effective_sample_sizes[index] < resample_threshold * num_particles:
            particle_filter.systematic_resample()
        estimates[index] = particle_filter.estimate()

    errors = true_states - estimates

    return {
        "time": time,
        "true_theta": true_states[:, 0],
        "true_omega": true_states[:, 1],
        "measured_theta": measured_theta,
        "estimated_theta": estimates[:, 0],
        "estimated_omega": estimates[:, 1],
        "angle_error": errors[:, 0],
        "omega_error": errors[:, 1],
        "effective_sample_sizes": effective_sample_sizes,
        "measurement_noise_deg": measurement_noise_deg,
        "num_particles": num_particles,
    }


def render_linear_kalman_filter_demo():
    """Render the linear Kalman Filter Streamlit demo."""
    st.subheader("Linear Kalman Filter: DC Motor Speed")
    st.write(
        "A linear Kalman Filter estimates hidden armature current and smooths "
        "noisy speed measurements from the existing DC motor state-space model."
    )

    parameter_col, metric_col = st.columns([1, 1])
    with parameter_col:
        st.caption("Parameters")
        voltage = st.slider(
            "Input voltage (V)",
            2.0,
            24.0,
            12.0,
            1.0,
            key="kf_voltage",
        )
        measurement_noise = st.slider(
            "Measurement noise std (rad/s)",
            0.2,
            5.0,
            2.0,
            0.1,
            key="kf_measurement_noise",
        )
        process_noise_scale = st.slider(
            "Process noise scale",
            0.1,
            10.0,
            1.0,
            0.1,
            key="kf_process_noise_scale",
        )
        simulation_time = st.slider(
            "Simulation time (s)",
            2.0,
            10.0,
            8.0,
            0.5,
            key="kf_simulation_time",
        )
        sample_time = st.slider(
            "Sample time dt (s)",
            0.005,
            0.05,
            0.01,
            0.005,
            format="%.3f",
            key="kf_sample_time",
        )

    result = run_linear_kalman_motor_demo(
        voltage,
        measurement_noise,
        process_noise_scale,
        simulation_time,
        sample_time,
    )
    speed_error = result["speed_error"]
    current_error = result["current_error"]

    with metric_col:
        st.caption("Metrics")
        render_metric_row(
            (
                (
                    "RMS speed error",
                    format_value(np.sqrt(np.mean(speed_error**2)), "rad/s"),
                ),
                ("Final speed error", format_value(speed_error[-1], "rad/s")),
            )
        )
        render_metric_row(
            (
                (
                    "RMS current error",
                    format_value(np.sqrt(np.mean(current_error**2)), "A"),
                ),
                ("Measurement noise", format_value(measurement_noise, "rad/s")),
            )
        )
        render_metric_row(
            (
                ("Process noise scale", format_value(process_noise_scale)),
                (
                    "Final estimate",
                    format_value(result["estimated_speed"][-1], "rad/s"),
                ),
            )
        )

    time = result["time"]
    figure, axes = create_streamlit_subplots(3, 1, sharex=True, width=9, height=8.4)
    axes[0].plot(time, result["true_speed"], label="True speed")
    axes[0].plot(
        time,
        result["measured_speed"],
        ".",
        alpha=0.25,
        label="Noisy speed measurement",
    )
    axes[0].plot(time, result["estimated_speed"], "--", label="Kalman estimate")
    format_engineering_axes(
        axes[0],
        title="Speed Estimate from Noisy Measurements",
        ylabel="Speed (rad/s)",
    )

    axes[1].plot(time, result["true_current"], label="True current")
    axes[1].plot(
        time,
        result["estimated_current"],
        "--",
        label="Estimated hidden current",
    )
    format_engineering_axes(
        axes[1],
        title="Hidden Current Estimate",
        ylabel="Current (A)",
    )

    axes[2].plot(time, speed_error, label="Speed error")
    axes[2].plot(time, current_error, label="Current error")
    axes[2].axhline(0.0, color="gray", linestyle=":")
    format_engineering_axes(
        axes[2],
        title="Estimation Errors",
        xlabel="Time (s)",
        ylabel="True - estimate",
    )
    show_figure(
        figure,
        "The filter combines the motor model with noisy speed measurements to "
        "estimate both measured speed and hidden armature current.",
    )

    render_assumptions_expander(
        "The measurement only observes motor speed. The Kalman Filter uses "
        "the linear state-space model to infer current, while the process "
        "and measurement noise settings tune how strongly it trusts the "
        "model versus the sensor.",
        title="Engineering interpretation",
    )


def render_ukf_pendulum_demo():
    """Render the Unscented Kalman Filter pendulum demo."""
    st.subheader("Unscented Kalman Filter: Nonlinear Pendulum")
    st.write(
        "The UKF estimates pendulum angle and hidden angular velocity from "
        "noisy angle measurements using sigma points instead of Jacobians."
    )

    parameter_col, metric_col = st.columns([1, 1])
    with parameter_col:
        st.caption("Parameters")
        initial_angle = st.slider(
            "Initial angle (deg)",
            5.0,
            60.0,
            25.0,
            1.0,
            key="ukf_initial_angle",
        )
        measurement_noise = st.slider(
            "Measurement noise std (deg)",
            0.2,
            8.0,
            2.0,
            0.1,
            key="ukf_measurement_noise",
        )
        process_noise_scale = st.slider(
            "Process noise scale",
            0.1,
            10.0,
            1.0,
            0.1,
            key="ukf_process_noise_scale",
        )
        initial_estimate_offset = st.slider(
            "Initial estimate offset (deg)",
            -20.0,
            20.0,
            -10.0,
            1.0,
            key="ukf_initial_estimate_offset",
        )
        simulation_time = st.slider(
            "Simulation time (s)",
            3.0,
            10.0,
            8.0,
            0.5,
            key="ukf_simulation_time",
        )

    result = run_ukf_pendulum_demo(
        initial_angle,
        measurement_noise,
        process_noise_scale,
        initial_estimate_offset,
        simulation_time,
    )
    angle_error_deg = np.degrees(result["angle_error"])
    omega_error = result["omega_error"]

    with metric_col:
        st.caption("Metrics")
        render_metric_row(
            (
                (
                    "RMS angle error",
                    format_value(np.sqrt(np.mean(angle_error_deg**2)), "deg"),
                ),
                ("Final angle error", format_value(angle_error_deg[-1], "deg")),
            )
        )
        render_metric_row(
            (
                (
                    "RMS omega error",
                    format_value(np.sqrt(np.mean(omega_error**2)), "rad/s"),
                ),
                ("Final omega error", format_value(omega_error[-1], "rad/s")),
            )
        )
        render_metric_row(
            (
                ("Measurement noise", format_value(measurement_noise, "deg")),
                ("Sample time", format_value(result["sample_time"], "s", 3)),
            )
        )

    time = result["time"]
    figure, axes = create_streamlit_subplots(2, 1, sharex=True, width=9, height=6.5)
    axes[0].plot(time, np.degrees(result["true_theta"]), label="True angle")
    axes[0].plot(
        time,
        np.degrees(result["measured_theta"]),
        ".",
        alpha=0.25,
        label="Noisy angle measurement",
    )
    axes[0].plot(
        time,
        np.degrees(result["estimated_theta"]),
        "--",
        label="UKF estimate",
    )
    format_engineering_axes(
        axes[0],
        title="Nonlinear Pendulum Angle Estimate",
        ylabel="Angle (deg)",
    )

    axes[1].plot(time, angle_error_deg, label="Angle error")
    axes[1].axhline(0.0, color="gray", linestyle=":")
    format_engineering_axes(
        axes[1],
        title="Angle Estimation Error",
        xlabel="Time (s)",
        ylabel="True - estimate (deg)",
    )
    show_figure(
        figure,
        "The UKF tracks the nonlinear pendulum angle while estimating angular "
        "velocity internally from angle-only measurements.",
    )

    render_assumptions_expander(
        "The pendulum measurement observes angle only. The Unscented Kalman "
        "Filter propagates sigma points through the nonlinear pendulum "
        "dynamics, so it can estimate angular velocity without requiring "
        "a measured velocity signal.",
        title="Engineering interpretation",
    )


def render_particle_filter_pendulum_demo():
    """Render the optional Particle Filter pendulum demo."""
    st.subheader("Particle Filter: Nonlinear Pendulum")
    st.write(
        "The Particle Filter represents uncertainty with weighted pendulum "
        "state hypotheses. It is optional here because particle count directly "
        "affects runtime."
    )

    parameter_col, metric_col = st.columns([1, 1])
    with parameter_col:
        st.caption("Parameters")
        run_demo = st.checkbox(
            "Run Particle Filter simulation",
            value=False,
            key="pf_run_demo",
        )
        num_particles = st.slider(
            "Number of particles",
            200,
            2000,
            600,
            100,
            key="pf_num_particles",
        )
        initial_angle = st.slider(
            "Initial angle (deg)",
            5.0,
            60.0,
            25.0,
            1.0,
            key="pf_initial_angle",
        )
        measurement_noise = st.slider(
            "Measurement noise std (deg)",
            0.5,
            8.0,
            2.0,
            0.1,
            key="pf_measurement_noise",
        )
        simulation_time = st.slider(
            "Simulation time (s)",
            3.0,
            8.0,
            6.0,
            0.5,
            key="pf_simulation_time",
        )

    if not run_demo:
        with metric_col:
            st.info(
                "Enable the checkbox to run the particle simulation. The "
                "default is off so routine page reruns stay quick."
            )
        return

    result = run_particle_filter_pendulum_demo(
        initial_angle,
        measurement_noise,
        num_particles,
        simulation_time,
    )
    angle_error_deg = np.degrees(result["angle_error"])
    effective_sample_sizes = result["effective_sample_sizes"]

    with metric_col:
        st.caption("Metrics")
        render_metric_row(
            (
                (
                    "RMS angle error",
                    format_value(np.sqrt(np.mean(angle_error_deg**2)), "deg"),
                ),
                ("Final angle error", format_value(angle_error_deg[-1], "deg")),
            )
        )
        render_metric_row(
            (
                ("Average ESS", format_value(np.mean(effective_sample_sizes))),
                ("Minimum ESS", format_value(np.min(effective_sample_sizes))),
            )
        )
        render_metric_row(
            (
                ("Particles", str(num_particles)),
                ("Measurement noise", format_value(measurement_noise, "deg")),
            )
        )

    time = result["time"]
    figure, axes = create_streamlit_subplots(3, 1, sharex=True, width=9, height=8.4)
    axes[0].plot(time, np.degrees(result["true_theta"]), label="True angle")
    axes[0].plot(
        time,
        np.degrees(result["measured_theta"]),
        ".",
        alpha=0.2,
        label="Noisy angle measurement",
    )
    axes[0].plot(
        time,
        np.degrees(result["estimated_theta"]),
        "--",
        label="Particle estimate",
    )
    format_engineering_axes(
        axes[0],
        title="Particle Filter Angle Estimate",
        ylabel="Angle (deg)",
    )

    axes[1].plot(time, angle_error_deg, label="Angle error")
    axes[1].axhline(0.0, color="gray", linestyle=":")
    format_engineering_axes(
        axes[1],
        title="Angle Estimation Error",
        ylabel="True - estimate (deg)",
    )

    axes[2].plot(time, effective_sample_sizes, label="Effective sample size")
    axes[2].axhline(
        0.5 * num_particles,
        color="gray",
        linestyle=":",
        label="Resampling threshold",
    )
    format_engineering_axes(
        axes[2],
        title="Particle Diversity",
        xlabel="Time (s)",
        ylabel="Particles",
    )
    show_figure(
        figure,
        "Effective sample size drops when particle weights concentrate; "
        "systematic resampling restores diversity when it falls below the "
        "threshold.",
    )

    render_assumptions_expander(
        "Particle Filters can represent non-Gaussian uncertainty, but they "
        "trade that flexibility for compute cost. The particle cap and "
        "short default simulation keep this page responsive.",
        title="Engineering interpretation",
    )


def render_state_estimation():
    """Render interactive state-estimation demos."""
    st.header("State Estimation")
    st.info("State estimation reconstructs hidden system states from noisy measurements.")
    render_info_box(
        "Interactive filters",
        "These demos reuse the repository's tested estimator classes and "
        "example dynamics. They keep sample counts modest and use fixed random "
        "seeds so slider changes produce repeatable comparisons.",
    )

    kalman_tab, ukf_tab, particle_tab = st.tabs(
        ("Kalman Filter", "Unscented Kalman Filter", "Particle Filter")
    )

    with kalman_tab:
        render_linear_kalman_filter_demo()

    with ukf_tab:
        render_ukf_pendulum_demo()

    with particle_tab:
        render_particle_filter_pendulum_demo()


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
    for column, card in zip(columns, included, strict=True):
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

    figure, axis = create_streamlit_subplots(width=9, height=5)
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

    figure, axes = create_streamlit_subplots(2, 1, sharex=True, width=9, height=6.5)
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

    figure, axes = create_streamlit_subplots(3, 1, sharex=True, width=9, height=8)
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
        render_assumptions_expander(
            "The explicit 1D heat scheme is stable only when "
            "`r = alpha*dt/dx^2 <= 0.5`. Reducing `dt`, reducing `alpha`, "
            "or using a coarser grid lowers the stability number.",
            title="Why this was blocked",
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
    figure, axes = create_streamlit_subplots(2, 1, width=9, height=7.4)
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
    render_assumptions_expander(
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
        render_assumptions_expander(
            "The explicit 1D wave scheme is stable only when "
            "`lambda = c*dt/dx <= 1`. Reducing `dt`, reducing wave speed, "
            "or using a coarser grid lowers the CFL number.",
            title="Why this was blocked",
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
    figure, axes = create_streamlit_subplots(2, 1, width=9, height=7.4)
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
    render_assumptions_expander(
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
        render_assumptions_expander(
            "The explicit 2D heat scheme is stable only when "
            "`rx + ry <= 0.5`, where `rx = alpha*dt/dx^2` and "
            "`ry = alpha*dt/dy^2`. Lower `dt`, lower `alpha`, or a coarser "
            "grid will reduce the stability sum.",
            title="Why this was blocked",
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
    render_assumptions_expander(
        "This demo uses a square uniform grid, constant thermal diffusivity, "
        "and an explicit finite-difference update. Stored frames are capped "
        "so the Streamlit page remains responsive with the default 41 x 41 "
        "grid.",
        title="Limitations, assumptions, and performance",
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
        render_assumptions_expander(
            "The explicit 2D wave scheme is stable only when "
            "`rx + ry <= 1`, where `rx = (c*dt/dx)^2` and "
            "`ry = (c*dt/dy)^2`. Lower `dt`, lower wave speed, or a coarser "
            "grid will reduce the stability sum.",
            title="Why this was blocked",
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
    render_assumptions_expander(
        "This demo uses a square uniform grid, constant wave speed, zero "
        "initial velocity, fixed boundaries, and an explicit update. Stored "
        "frames are capped to keep the default 41 x 41 simulation responsive.",
        title="Limitations, assumptions, and performance",
    )


def draw_2d_heat_plots(result):
    """Draw 2D heat snapshots and a centerline profile."""
    x = result["x"]
    y = result["y"]
    t = result["t"]
    temperature = result["temperature"]
    snapshot_indices = [0, int(round(0.5 * (len(t) - 1))), len(t) - 1]

    figure, axes = create_streamlit_subplots(2, 2, width=12, height=8.8)
    flat_axes = axes.ravel()
    color_min = float(np.min(temperature))
    color_max = float(np.max(temperature))

    for axis, index in zip(flat_axes[:3], snapshot_indices, strict=True):
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

    figure, axes = create_streamlit_subplots(2, 3, width=13.5, height=8.8)
    flat_axes = axes.ravel()
    for axis, index in zip(flat_axes[:4], snapshot_indices, strict=True):
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
    figure, axis = create_streamlit_subplots(width=9, height=5.6)
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

    safe_display_dataframe(
        {
            "grid points": num_points_values,
            "dx": dx_values,
            "forward RMS error": forward_errors,
            "backward RMS error": backward_errors,
            "central RMS error": central_errors,
        }
    )
    render_assumptions_expander(
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
    figure, axes = create_streamlit_subplots(3, 1, width=9, height=9.4)
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
    render_assumptions_expander(
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

    render_page_header(domain)

    if domain == "Home":
        render_home()
    elif domain == "State Estimation":
        render_state_estimation()
    elif domain == "UAV / Quadcopter":
        render_uav_quadcopter()
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
