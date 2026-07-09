import numpy as np
import pytest

from models.wave_equation_1d import (
    check_wave_stability,
    gaussian_displacement,
    simulate_wave_equation_1d,
    sine_displacement,
    triangular_displacement,
    wave_cfl_number,
    zero_initial_velocity,
)


def test_wave_cfl_number_matches_formula():
    """The CFL number should be c*dt/dx."""
    c = 2.0
    dt = 0.01
    dx = 0.05

    assert wave_cfl_number(c, dt, dx) == pytest.approx(c * dt / dx)


def test_stable_parameters_pass_stability_check():
    """Parameters with CFL <= 1 should pass the stability check."""
    assert check_wave_stability(c=1.0, dt=0.05, dx=0.1)


def test_unstable_parameters_fail_stability_check():
    """Parameters with CFL > 1 should fail when not raising."""
    assert not check_wave_stability(c=2.0, dt=0.1, dx=0.1)


def test_unstable_parameters_raise_when_requested():
    """The helper should raise for unstable parameters when configured."""
    with pytest.raises(ValueError):
        check_wave_stability(
            c=2.0,
            dt=0.1,
            dx=0.1,
            raise_error=True,
        )


def test_gaussian_displacement_returns_correct_shape():
    """The Gaussian helper should return one value per x point."""
    x = np.linspace(0.0, 1.0, 21)

    displacement = gaussian_displacement(x)

    assert displacement.shape == x.shape
    assert np.all(np.isfinite(displacement))


def test_sine_displacement_returns_correct_shape():
    """The sine helper should return one value per x point."""
    x = np.linspace(0.0, 1.0, 21)

    displacement = sine_displacement(x)

    assert displacement.shape == x.shape
    assert np.all(np.isfinite(displacement))


def test_triangular_displacement_returns_correct_shape():
    """The triangular helper should return one value per x point."""
    x = np.linspace(0.0, 1.0, 21)

    displacement = triangular_displacement(x)

    assert displacement.shape == x.shape
    assert np.all(np.isfinite(displacement))


def test_zero_initial_velocity_returns_correct_shape_and_zeros():
    """The zero-velocity helper should match x shape and contain zeros."""
    x = np.linspace(0.0, 1.0, 21)

    velocity = zero_initial_velocity(x)

    assert velocity.shape == x.shape
    assert np.allclose(velocity, 0.0)


def test_simulate_wave_equation_returns_expected_keys():
    """The solver should return the documented result dictionary fields."""
    result = simulate_wave_equation_1d(t_final=0.1, num_points=21)

    expected_keys = {
        "x",
        "t",
        "displacement",
        "c",
        "dx",
        "dt",
        "cfl_number",
        "boundary_type",
        "boundary_values",
    }

    assert expected_keys.issubset(result.keys())


def test_displacement_array_has_expected_shape():
    """Displacement should have shape (num_time_steps, num_points)."""
    num_points = 21

    result = simulate_wave_equation_1d(t_final=0.1, num_points=num_points)

    assert result["displacement"].shape == (len(result["t"]), num_points)


def test_x_and_t_arrays_are_one_dimensional():
    """The spatial and time grids should be one-dimensional arrays."""
    result = simulate_wave_equation_1d(t_final=0.1, num_points=21)

    assert result["x"].ndim == 1
    assert result["t"].ndim == 1


def test_dirichlet_boundaries_remain_fixed():
    """Fixed-displacement boundaries should remain fixed for all time."""
    result = simulate_wave_equation_1d(
        t_final=0.1,
        num_points=21,
        boundary_values=(0.5, -0.5),
    )

    displacement = result["displacement"]

    assert np.allclose(displacement[:, 0], 0.5)
    assert np.allclose(displacement[:, -1], -0.5)


def test_neumann_boundaries_have_zero_gradient():
    """Free boundaries should copy their nearest interior neighbors."""
    result = simulate_wave_equation_1d(
        t_final=0.1,
        num_points=21,
        boundary_type="neumann",
    )

    displacement = result["displacement"]

    assert np.allclose(displacement[:, 0], displacement[:, 1])
    assert np.allclose(displacement[:, -1], displacement[:, -2])


def test_unstable_dt_raises_value_error_when_enforced():
    """The simulator should reject unstable explicit time steps by default."""
    with pytest.raises(ValueError):
        simulate_wave_equation_1d(
            length=1.0,
            c=2.0,
            t_final=0.1,
            num_points=11,
            dt=0.1,
            enforce_stability=True,
        )


def test_invalid_length_raises_value_error():
    """String length must be positive."""
    with pytest.raises(ValueError):
        simulate_wave_equation_1d(length=0.0)


def test_invalid_wave_speed_raises_value_error():
    """Wave speed must be positive."""
    with pytest.raises(ValueError):
        simulate_wave_equation_1d(c=0.0)


def test_invalid_t_final_raises_value_error():
    """Final simulation time must be positive."""
    with pytest.raises(ValueError):
        simulate_wave_equation_1d(t_final=0.0)


def test_invalid_num_points_raises_value_error():
    """The grid needs at least two boundaries and one interior point."""
    with pytest.raises(ValueError):
        simulate_wave_equation_1d(num_points=2)


def test_invalid_dt_raises_value_error():
    """Provided time step must be positive."""
    with pytest.raises(ValueError):
        simulate_wave_equation_1d(dt=0.0)


def test_invalid_boundary_type_raises_value_error():
    """Only supported boundary condition types should be accepted."""
    with pytest.raises(ValueError):
        simulate_wave_equation_1d(boundary_type="periodic")


def test_sine_mode_stays_bounded_for_stable_cfl():
    """A stable fixed-end sine mode should remain bounded."""
    result = simulate_wave_equation_1d(
        t_final=0.5,
        num_points=101,
        initial_displacement=lambda x: sine_displacement(
            x,
            amplitude=1.0,
            mode=1,
            length=1.0,
        ),
        initial_velocity=zero_initial_velocity,
    )

    displacement = result["displacement"]

    assert np.max(np.abs(displacement)) <= 1.1


def test_solution_contains_only_finite_values():
    """The stable solver should produce finite displacement values."""
    result = simulate_wave_equation_1d(t_final=0.5, num_points=101)

    assert result["cfl_number"] <= 1.0
    assert np.all(np.isfinite(result["displacement"]))


def test_default_pulse_evolves_over_time():
    """A default displacement pulse should propagate, not remain static."""
    result = simulate_wave_equation_1d(t_final=0.5, num_points=101)

    displacement = result["displacement"]

    assert not np.allclose(displacement[0], displacement[-1])


def test_nonzero_initial_velocity_changes_first_step_in_expected_direction():
    """Positive initial velocity should move the first interior step upward."""
    result = simulate_wave_equation_1d(
        t_final=0.05,
        num_points=21,
        initial_displacement=lambda x: np.zeros_like(x),
        initial_velocity=lambda x: np.ones_like(x),
        boundary_values=(0.0, 0.0),
    )

    displacement = result["displacement"]

    assert np.all(displacement[1, 1:-1] > displacement[0, 1:-1])


def test_uniform_initial_velocity_first_step_matches_dt_for_flat_string():
    """With zero curvature, the first wave step should advance by v0*dt."""
    result = simulate_wave_equation_1d(
        t_final=0.02,
        num_points=21,
        initial_displacement=lambda x: np.zeros_like(x),
        initial_velocity=lambda x: np.full_like(x, 0.25),
        boundary_values=(0.0, 0.0),
    )
    displacement = result["displacement"]

    assert displacement[1, 1:-1] == pytest.approx(0.25 * result["dt"])


def test_initial_displacement_wrong_shape_raises_value_error():
    """Initial displacement must provide one value per grid point."""
    with pytest.raises(ValueError):
        simulate_wave_equation_1d(
            t_final=0.1,
            num_points=21,
            initial_displacement=lambda x: np.ones(len(x) - 1),
        )


def test_initial_velocity_wrong_shape_raises_value_error():
    """Initial velocity must provide one value per grid point."""
    with pytest.raises(ValueError):
        simulate_wave_equation_1d(
            t_final=0.1,
            num_points=21,
            initial_velocity=lambda x: np.ones(len(x) - 1),
        )
