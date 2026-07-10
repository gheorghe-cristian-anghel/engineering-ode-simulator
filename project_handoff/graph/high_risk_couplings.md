# High-Risk Couplings and Cleanup Opportunities

This is a static dependency review. It does not claim runtime bugs; it highlights places where future changes are likely to ripple.

## Highest-risk couplings

### 1. Streamlit app imports too much domain detail

- `streamlit_app.py` directly imports 8 model modules, 10 analysis modules, 2 example modules, and 1 visualization module.
- It has 3469 lines and 67 functions, so feature changes can easily mix UI, plotting, and solver orchestration.
- Cleanup opportunity: split domain-specific Streamlit sections first, preserving existing function calls and behavior.

### 2. Model layer has a reverse dependency on analysis

| Model module | Analysis dependency |
| --- | --- |
| models.inverted_pendulum_lqr | analysis.lqr |

- Most dependencies flow analysis -> models, but this edge points models -> analysis.
- Cleanup opportunity: move generic LQR math either into `models`-neutral shared utilities or keep the LQR-specific model in analysis.

### 3. Analysis modules depend on each other for shared helpers

| Analysis module | Analysis dependency |
| --- | --- |
| analysis.frequency_response | analysis.transfer_function |
| analysis.parameter_sweep | analysis.step_response |
| analysis.quadcopter_obstacle_avoidance | analysis.quadcopter_trajectory_tracking |
| analysis.quadcopter_waypoint_following | analysis.quadcopter_trajectory_tracking |
| analysis.rlc_sweep | analysis.step_response |

- These are reasonable compositions, but repeated shared metrics/trajectory helpers can become sticky if they live inside feature modules.
- Cleanup opportunity: create small shared analysis utility modules only when extracting existing repeated helpers.

### 4. Visualization reaches into model internals

| Visualization module | Model dependency |
| --- | --- |
| visualization.quadcopter_animation | models.quadcopter_6dof |

- `visualization.quadcopter_animation` imports the 6-DOF rotation helper from the model layer.
- Cleanup opportunity: keep the helper stable or move pure geometry transforms to a neutral utility if more visualizations need it.

### 5. Examples are both demos and helper providers

- `streamlit_app.py` imports: examples.run_particle_filter_pendulum, examples.run_ukf_pendulum.
- Many examples repeat `sys`/`Path` project-root bootstrapping and Matplotlib save/show patterns.
- Cleanup opportunity: keep examples as thin scripts, move reusable simulation setup/metrics functions into package modules.

## Repeated helper patterns

| Pattern | Count | Examples |
| --- | --- | --- |
| simulate_* | 31 | analysis.finite_element_1d.simulate_axial_bar_fem, analysis.model_predictive_control.simulate_mpc_tracking, analysis.quadcopter_altitude_control.simulate_altitude_pid_control, analysis.quadcopter_attitude_control.simulate_attitude_pid_control, analysis.quadcopter_obstacle_avoidance.simulate_quadcopter_obstacle_avoidance, analysis.quadcopter_trajectory_tracking.simulate_quadcopter_trajectory_tracking, analysis.quadcopter_waypoint_following.simulate_quadcopter_waypoint_following, analysis.state_space.simulate_state_space |
| validate_* | 17 | analysis.extended_kalman_filter.validate_ekf_matrices, analysis.kalman_filter.validate_kalman_matrices, analysis.lqr.validate_lqr_matrices, analysis.state_space.validate_state_space_matrices, models.dc_motor.validate_dc_motor_parameters, models.inverted_pendulum.validate_inverted_pendulum_parameters, models.mass_spring_damper.validate_mass_spring_damper_parameters, models.pendulum.validate_pendulum_parameters |
| analytical_* | 5 | analysis.finite_element_1d.analytical_tip_displacement_axial_bar, models.cooling.analytical_cooling, models.first_order_control.analytical_step_response, models.rc_circuit.analytical_rc, models.rl_circuit.analytical_rl |
| *_state_space | 8 | analysis.kalman_filter.discretize_state_space, analysis.state_space.simulate_state_space, analysis.state_space.mass_spring_damper_state_space, analysis.state_space.rlc_state_space, analysis.state_space.dc_motor_state_space, models.inverted_pendulum.linearized_inverted_pendulum_state_space, models.quadcopter_altitude.linearized_altitude_state_space, models.quadcopter_attitude.linearized_attitude_state_space |
| *_trajectory | 6 | analysis.model_predictive_control._reference_trajectory, analysis.model_predictive_control.predict_trajectory, analysis.quadcopter_trajectory_tracking.hover_trajectory, analysis.quadcopter_trajectory_tracking.circular_trajectory, analysis.quadcopter_waypoint_following.waypoint_trajectory, visualization.inverted_pendulum_animation._validate_trajectory |
| example scripts | 68 | examples.run_cooling, examples.run_dc_motor, examples.run_discrete_pid_motor, examples.run_ekf_pendulum, examples.run_fem_1d_bar, examples.run_first_order_control, examples.run_frequency_response_first_order, examples.run_heat_equation_1d |
| Streamlit plot helpers | 4 | visualization.plot_style.create_streamlit_figure, visualization.plot_style.create_streamlit_subplots, visualization.plot_style.finalize_streamlit_figure, visualization.plot_style.display_streamlit_figure |

## Modules with many incoming dependencies

| Module | Incoming modules | Imported by |
| --- | --- | --- |
| visualization.plot_style | 19 | examples.run_fem_1d_bar, examples.run_finite_difference_convergence, examples.run_finite_difference_derivatives, examples.run_heat_equation_1d, examples.run_heat_equation_2d, examples.run_kalman_dc_motor, examples.run_mpc_double_integrator, examples.run_particle_filter_pendulum |
| models.discrete_pid | 9 | analysis.pid_tuning, analysis.quadcopter_altitude_control, analysis.quadcopter_attitude_control, examples.run_dc_motor_disturbance_rejection_comparison, examples.run_discrete_pid_disturbance_response, examples.run_discrete_pid_motor, streamlit_app, tests.test_discrete_pid |
| models.dc_motor | 9 | examples.run_dc_motor, examples.run_dc_motor_disturbance_rejection_comparison, examples.run_dc_motor_open_loop_disturbance, examples.run_pid_motor_control, examples.run_state_space_dc_motor, models.discrete_pid, models.pid_motor_control, streamlit_app |
| analysis.quadcopter_trajectory_tracking | 8 | analysis.quadcopter_obstacle_avoidance, analysis.quadcopter_waypoint_following, examples.run_quadcopter_circle_animation, examples.run_quadcopter_trajectory_circle_tracking, examples.run_quadcopter_trajectory_hover_tracking, streamlit_app, tests.test_quadcopter_obstacle_avoidance, tests.test_quadcopter_trajectory_tracking |
| analysis.state_space | 8 | examples.run_inverted_pendulum_linearized, examples.run_kalman_dc_motor, examples.run_kalman_rlc, examples.run_state_space_dc_motor, examples.run_state_space_mass_spring_damper, examples.run_state_space_rlc, streamlit_app, tests.test_state_space |
| models.pid_motor_control | 7 | analysis.motor_disturbance, analysis.parameter_sweep, examples.run_dc_motor_disturbance_rejection_comparison, examples.run_motor_load_disturbance, examples.run_pid_motor_control, tests.test_motor_load_disturbance, tests.test_pid_motor_control |
| analysis.step_response | 7 | analysis.parameter_sweep, analysis.rlc_sweep, examples.run_first_order_control, examples.run_pid_motor_control, examples.run_rlc_circuit, examples.run_second_order_control, tests.test_step_response |
| models.quadcopter_6dof | 7 | analysis.quadcopter_obstacle_avoidance, analysis.quadcopter_trajectory_tracking, examples.run_quadcopter_6dof_hover, examples.run_quadcopter_6dof_tilted_thrust, examples.run_quadcopter_6dof_torque_response, tests.test_quadcopter_6dof, visualization.quadcopter_animation |
| models.rlc_circuit | 6 | analysis.rlc_sweep, examples.run_frequency_response_rlc, examples.run_rlc_circuit, examples.run_state_space_rlc, streamlit_app, tests.test_rlc_circuit |
| models.inverted_pendulum | 6 | examples.run_inverted_pendulum_animation_open_loop, examples.run_inverted_pendulum_linearized, examples.run_inverted_pendulum_lqr_comparison, examples.run_inverted_pendulum_open_loop, models.inverted_pendulum_lqr, tests.test_inverted_pendulum |

## Modules with many outgoing dependencies

| Module | Outgoing modules | Imports |
| --- | --- | --- |
| streamlit_app | 21 | analysis.finite_difference, analysis.finite_element_1d, analysis.kalman_filter, analysis.particle_filter, analysis.quadcopter_altitude_control, analysis.quadcopter_obstacle_avoidance, analysis.quadcopter_trajectory_tracking, analysis.quadcopter_waypoint_following |
| examples.run_dc_motor_disturbance_rejection_comparison | 4 | analysis.motor_disturbance, models.dc_motor, models.discrete_pid, models.pid_motor_control |
| examples.run_kalman_dc_motor | 3 | analysis.kalman_filter, analysis.state_space, visualization.plot_style |
| examples.run_pid_motor_control | 3 | analysis.step_response, models.dc_motor, models.pid_motor_control |
| examples.run_quadcopter_obstacle_avoidance | 3 | analysis.quadcopter_obstacle_avoidance, analysis.quadcopter_waypoint_following, visualization.plot_style |
| analysis.parameter_sweep | 2 | analysis.step_response, models.pid_motor_control |
| analysis.quadcopter_altitude_control | 2 | models.discrete_pid, models.quadcopter_altitude |
| analysis.quadcopter_attitude_control | 2 | models.discrete_pid, models.quadcopter_attitude |
| analysis.quadcopter_obstacle_avoidance | 2 | analysis.quadcopter_trajectory_tracking, models.quadcopter_6dof |
| analysis.rlc_sweep | 2 | analysis.step_response, models.rlc_circuit |

## Recommended next refactors

1. Split `streamlit_app.py` by domain while preserving existing imports and behavior.
2. Remove Streamlit runtime imports from pure visualization helpers by separating `display_streamlit_figure` from Matplotlib-only style functions.
3. Move reusable demo helpers out of `examples/` so Streamlit no longer depends on example scripts.
4. Resolve the `models.inverted_pendulum_lqr -> analysis.lqr` layering inversion.
5. Consolidate repeated example bootstrapping and figure save/show handling once the app split is stable.
