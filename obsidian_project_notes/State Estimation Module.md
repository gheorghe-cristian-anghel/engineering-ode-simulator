---
title: State Estimation Module
tags:
  - engineering-simulation-toolkit
  - state-estimation
  - filters
---

# State Estimation Module

The state-estimation area demonstrates linear and nonlinear estimation patterns with reusable analysis helpers and examples.

## Kalman Filter

KF examples cover linear discrete-time estimation for systems such as DC motor and RLC models.

Relevant area:

- `analysis/kalman_filter.py`

## Extended Kalman Filter

EKF examples estimate nonlinear pendulum state using nonlinear prediction and local Jacobians.

Relevant area:

- `analysis/extended_kalman_filter.py`

## Unscented Kalman Filter

UKF examples estimate nonlinear pendulum state using sigma points, avoiding hand-derived Jacobians.

Relevant area:

- `analysis/unscented_kalman_filter.py`

## Particle Filter

PF examples use weighted particles, effective sample size, and resampling for nonlinear/non-Gaussian estimation.

Relevant area:

- `analysis/particle_filter.py`

## Plots

Plots should make these relationships clear:

- true state
- noisy measurement
- estimated state
- uncertainty or error where available

Use [[Visualization and Animations]] guidance for labels, colors, and readability.

## Metrics

Useful metrics:

- RMSE
- final error
- covariance sanity checks
- finite output checks
- comparison against raw measurements

## Limitations

- Examples are educational and parameter-specific.
- Filter tuning is illustrative, not production calibration.
- Do not imply robust sensor fusion for real systems.

Related: [[Scientific Correctness]], [[Streamlit App]].
