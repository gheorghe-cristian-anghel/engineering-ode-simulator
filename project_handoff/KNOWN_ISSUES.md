# Known Issues

## Streamlit Plot Helpers

- Recent commits fixed Streamlit plot style helpers and plot sizing.
- Treat `visualization/plot_style.py` and `streamlit_app.py` plot display paths as regression-sensitive.
- Watch for layout issues with legends outside axes, colorbars, constrained layout, `st.pyplot(width="stretch")`, and multi-panel figures.
- Current logs show Streamlit started successfully, but the active process was not verified due to a Windows permission denial during port inspection.

## Possible Scientific Audit Findings

- Many systems are intentionally simplified educational models.
- Quadcopter examples are not flight-ready and do not include rotor-level motor dynamics, real autopilot firmware, sensor fusion stacks, SLAM, global planning, or high-fidelity aerodynamics.
- Obstacle avoidance is local/reactive and static-obstacle focused.
- PDE solvers use explicit finite differences and require stability constraints.
- FEM is introductory and currently limited to a 1D axial bar.
- Control examples demonstrate concepts; they are not safety-critical or certified controller implementations.

## Areas That Are Simplified

- UAV dynamics and controllers.
- Obstacle avoidance.
- Explicit PDE methods.
- FEM coverage.
- Streamlit demo layer.
- Portfolio screenshots and visual outputs.

## Areas Not To Overclaim

- Production readiness.
- Industrial simulation accuracy.
- Flight readiness.
- Certified control behavior.
- Real-time embedded behavior.
- High-performance PDE/FEM solving.
- Full validation against external tools.

## Future Cleanup Opportunities

- Extract repeated Streamlit UI helper patterns only if `streamlit_app.py` becomes hard to maintain.
- Add more shared validation helpers only where repetition is clear.
- Curate screenshots and animations with a reproducible generation workflow.
- Add more analytical references and validation tests for selected models.
- Consider packaging/release polish after the app, tests, and docs are stable.
