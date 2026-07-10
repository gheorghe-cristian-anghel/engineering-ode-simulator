# Streamlit Dependency Map

`streamlit_app.py` is the main interactive shell. It directly imports simulation models, analysis/control helpers, selected example helpers, and `visualization.plot_style` as `ps`.

## Size and UI surface

- File length: 3469 lines.
- Functions defined: 67.
- `@st.cache_data` functions: 8.
- Direct local imports: 21 modules.

## Direct local imports from the app

| Area | Modules | Imported modules |
| --- | --- | --- |
| analysis/ | 10 | analysis.finite_difference, analysis.finite_element_1d, analysis.kalman_filter, analysis.particle_filter, analysis.quadcopter_altitude_control, analysis.quadcopter_obstacle_avoidance, analysis.quadcopter_trajectory_tracking, analysis.quadcopter_waypoint_following, analysis.state_space, analysis.unscented_kalman_filter |
| examples/ | 2 | examples.run_particle_filter_pendulum, examples.run_ukf_pendulum |
| models/ | 8 | models.dc_motor, models.discrete_pid, models.heat_equation_1d, models.heat_equation_2d, models.rc_circuit, models.rlc_circuit, models.wave_equation_1d, models.wave_equation_2d |
| visualization/ | 1 | visualization.plot_style |

## Streamlit API calls in app

| st API | Calls |
| --- | --- |
| slider | 68 |
| caption | 23 |
| columns | 22 |
| subheader | 20 |
| number_input | 19 |
| cache_data | 8 |
| info | 6 |
| markdown | 5 |
| write | 5 |
| warning | 4 |
| sidebar | 4 |
| radio | 3 |
| header | 2 |
| success | 2 |
| tabs | 2 |
| container | 1 |
| expander | 1 |
| dataframe | 1 |
| checkbox | 1 |
| set_page_config | 1 |

## Plot-style coupling

The app imports `visualization.plot_style` as `ps` and calls these helpers directly:

| ps helper | Calls |
| --- | --- |
| format_engineering_axes | 30 |
| create_streamlit_subplots | 16 |
| place_legend_outside | 9 |
| place_legends_outside | 8 |
| format_heatmap_axes | 4 |
| add_clean_colorbar | 4 |
| set_xy_plot_limits_with_margin | 3 |
| THERMAL_COLORMAP | 2 |
| symmetric_color_limits | 2 |
| DIVERGING_COLORMAP | 2 |
| set_equal_2d_axes | 2 |
| display_streamlit_figure | 1 |
| apply_engineering_plot_style | 1 |

Direct `matplotlib.pyplot` calls in app:

| plt helper | Calls |
| --- | --- |
| Circle | 2 |

## Coupling observations

- The app owns UI layout, parameter widgets, simulation orchestration, metrics, and plotting assembly in one file.
- `visualization.plot_style.display_streamlit_figure()` imports Streamlit internally, so visualization has an optional UI dependency.
- The app imports selected helpers from `examples.run_particle_filter_pendulum` and `examples.run_ukf_pendulum`, which makes examples part of the runtime app dependency graph.
- Plot helper usage is centralized through `ps`, which is good, but the app still knows low-level figure/subplot/finalization helpers rather than rendering through feature-level view functions.

## Suggested Streamlit cleanup path

1. Extract one app adapter module per domain, for example `app_sections/uav.py`, `app_sections/state_estimation.py`, and `app_sections/pde.py`.
2. Move demo-only helper functions currently imported from `examples/` into `analysis/` or a small app utility module, then let examples and Streamlit both depend on that shared code.
3. Keep `visualization.plot_style` UI-light by separating pure Matplotlib style helpers from Streamlit display wrappers.
4. Add smoke tests for importability of app sections after decomposition, without running full Streamlit.
