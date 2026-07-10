# Module Dependency Graph

Generated from Graphify plus a local AST import scan. Source code was not modified.

## Graphify status

- Graphify CLI: used via `graphify extract . --code-only --no-cluster --out project_handoff\graph`.
- Graphify code graph: 3117 nodes, 5619 edges, code-only/no-cluster.
- Local scan scope: 165 Python files excluding `.git`, `.venv`, caches, package metadata, and generated Graphify output.
- Internal import edges found by local scan: 199.

## Package shape

| Area | Python files |
| --- | --- |
| analysis/ | 23 |
| examples/ | 68 |
| models/ | 21 |
| streamlit_app.py | 1 |
| tests/ | 48 |
| visualization/ | 4 |

## Layer-level import edges

| From | To | Import edges |
| --- | --- | --- |
| examples/ | analysis/ | 47 |
| examples/ | models/ | 41 |
| tests/ | analysis/ | 23 |
| tests/ | models/ | 23 |
| examples/ | visualization/ | 18 |
| analysis/ | models/ | 10 |
| streamlit_app.py | analysis/ | 10 |
| streamlit_app.py | models/ | 8 |
| analysis/ | analysis/ | 5 |
| tests/ | visualization/ | 5 |
| models/ | models/ | 3 |
| streamlit_app.py | examples/ | 2 |
| models/ | analysis/ | 1 |
| streamlit_app.py | visualization/ | 1 |
| visualization/ | models/ | 1 |
| visualization/ | visualization/ | 1 |

## Most imported local modules

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
| analysis.transfer_function | 5 | analysis.frequency_response, examples.run_transfer_function_comparison, examples.run_transfer_function_impulse_response, examples.run_transfer_function_step_response, tests.test_transfer_function |
| analysis.pid_tuning | 5 | examples.run_pid_kd_tuning, examples.run_pid_ki_tuning, examples.run_pid_kp_tuning, examples.run_pid_p_pi_pid_comparison, tests.test_pid_tuning |
| analysis.quadcopter_waypoint_following | 5 | examples.run_quadcopter_obstacle_avoidance, examples.run_quadcopter_waypoint_animation, examples.run_quadcopter_waypoint_following, streamlit_app, tests.test_quadcopter_waypoint_following |
| models.quadcopter_altitude | 4 | analysis.quadcopter_altitude_control, examples.run_quadcopter_altitude_open_loop, examples.run_quadcopter_altitude_thrust_step, tests.test_quadcopter_altitude |
| models.quadcopter_attitude | 4 | analysis.quadcopter_attitude_control, examples.run_quadcopter_attitude_roll_torque, examples.run_quadcopter_attitude_torque_step, tests.test_quadcopter_attitude |

## Modules with most local outgoing dependencies

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
| examples.run_dc_motor_open_loop_disturbance | 2 | analysis.motor_disturbance, models.dc_motor |
| examples.run_discrete_pid_motor | 2 | analysis.export_utils, models.discrete_pid |
| examples.run_ekf_pendulum | 2 | analysis.extended_kalman_filter, models.pendulum |
| examples.run_fem_1d_bar | 2 | analysis.finite_element_1d, visualization.plot_style |
| examples.run_finite_difference_convergence | 2 | analysis.finite_difference, visualization.plot_style |

## Major external dependencies seen in imports

| Dependency root | Import statements |
| --- | --- |
| numpy | 131 |
| matplotlib | 78 |
| pathlib | 72 |
| sys | 68 |
| _tkinter | 50 |
| pytest | 46 |
| scipy | 26 |
| dataclasses | 12 |
| argparse | 4 |
| csv | 2 |
| warnings | 2 |
| streamlit | 2 |
| math | 1 |
| cycler | 1 |

## Core module adjacency

| Module | Area | Outgoing | Incoming | Local imports |
| --- | --- | --- | --- | --- |
| analysis.__init__ | analysis/ | 0 | 0 | - |
| analysis.export_utils | analysis/ | 0 | 2 | - |
| analysis.extended_kalman_filter | analysis/ | 0 | 2 | - |
| analysis.finite_difference | analysis/ | 0 | 4 | - |
| analysis.finite_element_1d | analysis/ | 0 | 3 | - |
| analysis.frequency_response | analysis/ | 1 | 4 | analysis.transfer_function |
| analysis.kalman_filter | analysis/ | 0 | 4 | - |
| analysis.lqr | analysis/ | 0 | 2 | - |
| analysis.model_predictive_control | analysis/ | 0 | 2 | - |
| analysis.motor_disturbance | analysis/ | 1 | 3 | models.pid_motor_control |
| analysis.parameter_sweep | analysis/ | 2 | 2 | analysis.step_response, models.pid_motor_control |
| analysis.particle_filter | analysis/ | 0 | 3 | - |
| analysis.pid_tuning | analysis/ | 1 | 5 | models.discrete_pid |
| analysis.quadcopter_altitude_control | analysis/ | 2 | 4 | models.discrete_pid, models.quadcopter_altitude |
| analysis.quadcopter_attitude_control | analysis/ | 2 | 3 | models.discrete_pid, models.quadcopter_attitude |
| analysis.quadcopter_obstacle_avoidance | analysis/ | 2 | 3 | analysis.quadcopter_trajectory_tracking, models.quadcopter_6dof |
| analysis.quadcopter_trajectory_tracking | analysis/ | 1 | 8 | models.quadcopter_6dof |
| analysis.quadcopter_waypoint_following | analysis/ | 1 | 5 | analysis.quadcopter_trajectory_tracking |
| analysis.rlc_sweep | analysis/ | 2 | 4 | analysis.step_response, models.rlc_circuit |
| analysis.state_space | analysis/ | 0 | 8 | - |
| analysis.step_response | analysis/ | 0 | 7 | - |
| analysis.transfer_function | analysis/ | 0 | 5 | - |
| analysis.unscented_kalman_filter | analysis/ | 0 | 3 | - |
| models.__init__ | models/ | 0 | 0 | - |
| models.cooling | models/ | 0 | 2 | - |
| models.dc_motor | models/ | 0 | 9 | - |
| models.discrete_pid | models/ | 1 | 9 | models.dc_motor |
| models.first_order_control | models/ | 0 | 3 | - |
| models.heat_equation_1d | models/ | 0 | 3 | - |
| models.heat_equation_2d | models/ | 0 | 3 | - |
| models.inverted_pendulum | models/ | 0 | 6 | - |
| models.inverted_pendulum_lqr | models/ | 2 | 4 | analysis.lqr, models.inverted_pendulum |
| models.mass_spring_damper | models/ | 0 | 3 | - |
| models.pendulum | models/ | 0 | 3 | - |
| models.pid_motor_control | models/ | 1 | 7 | models.dc_motor |
| models.quadcopter_6dof | models/ | 0 | 7 | - |
| models.quadcopter_altitude | models/ | 0 | 4 | - |
| models.quadcopter_attitude | models/ | 0 | 4 | - |
| models.rc_circuit | models/ | 0 | 3 | - |
| models.rl_circuit | models/ | 0 | 2 | - |
| models.rlc_circuit | models/ | 0 | 6 | - |
| models.second_order_control | models/ | 0 | 2 | - |
| models.wave_equation_1d | models/ | 0 | 3 | - |
| models.wave_equation_2d | models/ | 0 | 3 | - |
| streamlit_app | streamlit_app.py | 21 | 0 | analysis.finite_difference, analysis.finite_element_1d, analysis.kalman_filter, analysis.particle_filter, analysis.quadcopter_altitude_control, analysis.quadcopter_obstacle_avoidance |
| visualization.__init__ | visualization/ | 1 | 0 | visualization.plot_style |
| visualization.inverted_pendulum_animation | visualization/ | 0 | 3 | - |
| visualization.plot_style | visualization/ | 0 | 19 | - |
| visualization.quadcopter_animation | visualization/ | 1 | 3 | models.quadcopter_6dof |

## Coverage by tests/examples/docs

- Package modules scanned: 45.
- Package modules with at least one direct test import: 45 / 45.
- Package modules with at least one direct example import: 44 / 45.
- Package modules mentioned by docs text using module basename: 33 / 45.

### Package modules without direct test imports

None found.

### Package modules without direct example imports

analysis.lqr

### Package modules without obvious docs mention

analysis.export_utils, analysis.frequency_response, analysis.motor_disturbance, analysis.parameter_sweep, analysis.rlc_sweep, analysis.state_space, models.cooling, models.mass_spring_damper, models.pid_motor_control, models.rc_circuit, models.rl_circuit, models.rlc_circuit
