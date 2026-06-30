# Portfolio Notes

## Project Pitch

Engineering ODE Simulator is a Python-based simulation toolkit for modeling
electrical, mechanical, thermal, and control systems using ordinary
differential equations. It demonstrates scientific computing, numerical
simulation, control-system analysis, and clean software engineering practices.

## Skills Demonstrated

- Python scientific computing
- NumPy
- SciPy `solve_ivp`
- Matplotlib visualization
- Streamlit browser UI for selected simulations
- frequency response and Bode plot analysis
- transfer function step and impulse response analysis
- state-space modeling for linear systems
- CSV export for Excel, MATLAB, LibreOffice Calc, or Python analysis
- saved plot screenshots for GitHub and README presentation
- pytest testing
- Git/GitHub workflow
- differential equation modeling
- control systems
- electromechanical modeling
- embedded-style digital PID control
- parameter validation
- documentation

## Best Models to Showcase

### RLC Circuit

Shows second-order electrical dynamics, damping, overshoot, and oscillation.

### Mass-Spring-Damper

Shows mechanical vibration and energy/damping concepts.

### Second-Order Control System

Shows damping ratio, natural frequency, overshoot, peak time, and settling time.

### DC Motor

Shows coupled electrical-mechanical dynamics.

### PI Motor Speed Control

Shows closed-loop feedback control, reference tracking, voltage saturation, and
motor speed regulation. The demo also includes load disturbance rejection,
where the controller increases voltage and current to recover speed after a
torque step.

### Discrete PID Motor Control

Shows embedded-style sampled control: fixed update time, held voltage commands,
output saturation, anti-windup, derivative-on-measurement, and DC motor speed
tracking.

### Streamlit GUI

Shows selected simulations in a browser UI with interactive inputs and
Matplotlib plots while reusing the tested model modules.

### Frequency Response

Shows Bode magnitude and phase plots for first-order, second-order, and RLC
low-pass transfer functions.

### Transfer Functions

Strengthens the control-systems toolkit with reusable transfer function
models, step response analysis, impulse response analysis, and common low-pass
helpers.

### State-Space Models

Shows modern control-system modeling using `x_dot = A*x + B*u` and
`y = C*x + D*u` for mechanical, electrical, and electromechanical systems.

### CSV Export

Shows practical engineering workflow support by exporting simulation results
for external analysis in Excel, MATLAB, LibreOffice Calc, or Python.

### Plot Screenshots

Selected simulation plots are saved as screenshots for GitHub, README, and
portfolio presentation.

## Freelance Relevance

The project supports offers such as:

- engineering simulation scripts
- ODE modeling
- plotting and visualization
- control-system response analysis
- converting equations into Python tools
- educational simulation apps

## Interview Talking Points

- I built a modular simulation library with one model per file.
- I used SciPy `solve_ivp` for numerical integration.
- I added tests that check engineering behavior.
- I implemented reusable step response metrics.
- I added reusable frequency response utilities and Bode plot examples.
- I added reusable state-space simulation utilities for linear systems.
- I added CSV export so simulation data can be reused in external analysis tools.
- I modeled both open-loop and closed-loop systems.
- I added an embedded-style discrete PID controller for motor speed control.
- I added a Streamlit GUI MVP so selected simulations can be explored from a
  browser.
- I documented the project for future extension.

## Future Portfolio Upgrades

- screenshots
- hosted demo
- blog post series
- LinkedIn posts
- short demo video
