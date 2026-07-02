# State Estimation

State estimation examples reconstruct hidden states from noisy measurements.
They are compact demonstrations of filtering algorithms, not production sensor
fusion systems.

## Governing Ideas

The linear Kalman Filter assumes:

```text
x[k+1] = A x[k] + B u[k] + w[k]
y[k]   = C x[k] + D u[k] + v[k]
```

Prediction propagates state and covariance; update applies a Kalman gain from
the innovation covariance.

The Extended Kalman Filter uses nonlinear models:

```text
x[k+1] = f(x[k], u[k], dt) + w[k]
y[k]   = h(x[k]) + v[k]
```

It linearizes the models locally with Jacobians.

The Unscented Kalman Filter also uses nonlinear models, but propagates sigma
points through the process and measurement functions instead of requiring
hand-derived Jacobians.

The Particle Filter represents uncertainty with weighted particles. Prediction
propagates particles, update reweights them by measurement likelihood, and
resampling combats weight collapse.

## Assumptions

- Kalman-based filters use covariance matrices chosen by the example.
- Linear Kalman Filter examples assume linear dynamics and Gaussian-style
  process/measurement uncertainty.
- EKF accuracy depends on local linearization quality.
- UKF accuracy depends on sigma-point scaling and covariance choices.
- Particle Filter quality depends on particle count, process noise, likelihood
  model, and resampling strategy.

## Numerical Method

- Kalman Filter: matrix prediction and linear measurement update.
- EKF: nonlinear prediction plus Jacobian-based covariance propagation.
- UKF: unscented transform over sigma points.
- Particle Filter: Monte Carlo propagation, likelihood weighting, effective
  sample size, and systematic resampling.

## What the Repository Demonstrates

- Linear state estimation for DC motor and RLC examples.
- Nonlinear pendulum estimation with EKF and UKF.
- Particle-based pendulum estimation with hidden angular velocity.
- Practical filter outputs such as state estimates, gains, weights, and
  effective sample size.

## Relevant Files and Examples

- `analysis/kalman_filter.py`
- `analysis/extended_kalman_filter.py`
- `analysis/unscented_kalman_filter.py`
- `analysis/particle_filter.py`
- `examples/run_kalman_dc_motor.py`
- `examples/run_kalman_rlc.py`
- `examples/run_ekf_pendulum.py`
- `examples/run_ukf_pendulum.py`
- `examples/run_particle_filter_pendulum.py`

## Limitations

- Noise models are simplified and manually selected.
- No real sensor calibration, bias estimation, data association, or SLAM is
  included.
- Filters are demonstrated on small systems with known models.
- Results should be interpreted as educational algorithm behavior, not as
  validated estimator performance for deployed systems.
