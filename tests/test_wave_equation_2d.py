import numpy as np
import pytest

from models.wave_equation_2d import (
    apply_2d_wave_boundary_conditions,
    check_wave_stability_2d,
    circular_ring_displacement_2d,
    create_2d_grid,
    gaussian_displacement_2d,
    simulate_wave_equation_2d,
    sine_displacement_2d,
    wave_stability_numbers_2d,
    zero_initial_velocity_2d,
)


def test_create_2d_grid_returns_expected_shapes():
    """2D grid helper should return vectors, mesh arrays, and spacings."""
    x, y, X, Y, dx, dy = create_2d_grid(width=2.0, height=1.0, nx=5, ny=4)

    assert x.shape == (5,)
    assert y.shape == (4,)
    assert X.shape == (4, 5)
    assert Y.shape == (4, 5)
    assert dx == pytest.approx(0.5)
    assert dy == pytest.approx(1.0 / 3.0)


def test_wave_stability_numbers_2d_match_formula():
    """The stability numbers should match the 2D wave CFL formulas."""
    c = 2.0
    dt = 0.002
    dx = 0.02
    dy = 0.04

    lambda_x, lambda_y, rx, ry, stability_sum = wave_stability_numbers_2d(
        c,
        dt,
        dx,
        dy,
    )

    assert lambda_x == pytest.approx(c * dt / dx)
    assert lambda_y == pytest.approx(c * dt / dy)
    assert rx == pytest.approx(lambda_x**2)
    assert ry == pytest.approx(lambda_y**2)
    assert stability_sum == pytest.approx(rx + ry)


def test_stable_parameters_pass_stability_check():
    """Parameters with rx + ry <= 1 should pass the stability check."""
    assert check_wave_stability_2d(c=1.0, dt=0.05, dx=0.1, dy=0.1)


def test_unstable_parameters_fail_stability_check():
    """Parameters with rx + ry > 1 should fail when not raising."""
    assert not check_wave_stability_2d(c=1.0, dt=0.1, dx=0.1, dy=0.1)


def test_unstable_parameters_raise_when_requested():
    """The helper should raise for unstable parameters when configured."""
    with pytest.raises(ValueError):
        check_wave_stability_2d(
            c=1.0,
            dt=0.1,
            dx=0.1,
            dy=0.1,
            raise_error=True,
        )


def test_gaussian_displacement_2d_returns_shape_ny_by_nx():
    """Gaussian helper should return one displacement per grid point."""
    _, _, X, Y, _, _ = create_2d_grid(nx=7, ny=5)

    displacement = gaussian_displacement_2d(X, Y)

    assert displacement.shape == (5, 7)
    assert np.all(np.isfinite(displacement))


def test_circular_ring_displacement_2d_returns_shape_ny_by_nx():
    """Ring helper should return one displacement per grid point."""
    _, _, X, Y, _, _ = create_2d_grid(nx=7, ny=5)

    displacement = circular_ring_displacement_2d(X, Y)

    assert displacement.shape == (5, 7)
    assert np.all(np.isfinite(displacement))


def test_sine_displacement_2d_returns_shape_ny_by_nx():
    """Sine helper should return one displacement per grid point."""
    _, _, X, Y, _, _ = create_2d_grid(nx=7, ny=5)

    displacement = sine_displacement_2d(X, Y)

    assert displacement.shape == (5, 7)
    assert np.all(np.isfinite(displacement))


def test_zero_initial_velocity_2d_returns_shape_and_zeros():
    """Zero velocity helper should match grid shape and contain zeros."""
    _, _, X, Y, _, _ = create_2d_grid(nx=7, ny=5)

    velocity = zero_initial_velocity_2d(X, Y)

    assert velocity.shape == (5, 7)
    assert np.allclose(velocity, 0.0)


def test_apply_2d_wave_boundary_conditions_fixes_dirichlet_boundaries():
    """Dirichlet boundary values should be written to all four edges."""
    displacement = np.ones((5, 7))

    apply_2d_wave_boundary_conditions(
        displacement,
        boundary_type="dirichlet",
        boundary_values=(1.0, 2.0, 3.0, 4.0),
    )

    assert np.allclose(displacement[1:-1, 0], 1.0)
    assert np.allclose(displacement[1:-1, -1], 2.0)
    assert np.allclose(displacement[0, :], 3.0)
    assert np.allclose(displacement[-1, :], 4.0)


def test_simulate_wave_equation_2d_returns_expected_keys():
    """The solver should return the documented result dictionary fields."""
    result = simulate_wave_equation_2d(t_final=0.01, nx=11, ny=9)

    expected_keys = {
        "x",
        "y",
        "X",
        "Y",
        "t",
        "displacement",
        "c",
        "dx",
        "dy",
        "dt",
        "lambda_x",
        "lambda_y",
        "rx",
        "ry",
        "stability_sum",
        "boundary_type",
        "boundary_values",
        "store_every",
    }

    assert expected_keys.issubset(result.keys())


def test_displacement_array_has_expected_stored_shape():
    """Stored displacement should have shape (stored_times, ny, nx)."""
    nx = 11
    ny = 9

    result = simulate_wave_equation_2d(t_final=0.01, nx=nx, ny=ny)

    assert result["displacement"].shape == (len(result["t"]), ny, nx)


def test_fixed_boundaries_remain_fixed():
    """Fixed-displacement boundaries should remain fixed for all stored times."""
    result = simulate_wave_equation_2d(
        t_final=0.01,
        nx=11,
        ny=9,
        boundary_values=(1.0, 2.0, 3.0, 4.0),
    )

    displacement = result["displacement"]

    assert np.allclose(displacement[:, 1:-1, 0], 1.0)
    assert np.allclose(displacement[:, 1:-1, -1], 2.0)
    assert np.allclose(displacement[:, 0, :], 3.0)
    assert np.allclose(displacement[:, -1, :], 4.0)


def test_solution_contains_only_finite_values_for_stable_cfl():
    """The stable solver should produce finite displacement values."""
    result = simulate_wave_equation_2d(t_final=0.2, nx=31, ny=31)

    assert result["stability_sum"] <= 1.0
    assert result["rx"] + result["ry"] == pytest.approx(result["stability_sum"])
    assert np.all(np.isfinite(result["displacement"]))


def test_unstable_dt_raises_value_error_when_enforced():
    """The solver should reject unstable explicit time steps by default."""
    with pytest.raises(ValueError):
        simulate_wave_equation_2d(
            c=1.0,
            t_final=0.1,
            nx=11,
            ny=11,
            dt=0.1,
            enforce_stability=True,
        )


def test_invalid_width_raises_value_error():
    """Membrane width must be positive."""
    with pytest.raises(ValueError):
        simulate_wave_equation_2d(width=0.0)


def test_invalid_height_raises_value_error():
    """Membrane height must be positive."""
    with pytest.raises(ValueError):
        simulate_wave_equation_2d(height=0.0)


def test_invalid_wave_speed_raises_value_error():
    """Wave speed must be positive."""
    with pytest.raises(ValueError):
        simulate_wave_equation_2d(c=0.0)


def test_invalid_nx_raises_value_error():
    """The x grid needs two boundaries and at least one interior point."""
    with pytest.raises(ValueError):
        simulate_wave_equation_2d(nx=2)


def test_invalid_ny_raises_value_error():
    """The y grid needs two boundaries and at least one interior point."""
    with pytest.raises(ValueError):
        simulate_wave_equation_2d(ny=2)


def test_invalid_t_final_raises_value_error():
    """Final simulation time must be positive."""
    with pytest.raises(ValueError):
        simulate_wave_equation_2d(t_final=0.0)


def test_invalid_dt_raises_value_error():
    """Provided time step must be positive."""
    with pytest.raises(ValueError):
        simulate_wave_equation_2d(dt=0.0)


def test_invalid_store_every_raises_value_error():
    """store_every must be at least one."""
    with pytest.raises(ValueError):
        simulate_wave_equation_2d(store_every=0)


def test_invalid_boundary_type_raises_value_error():
    """Only fixed Dirichlet boundaries should be accepted in this first scope."""
    with pytest.raises(ValueError):
        simulate_wave_equation_2d(boundary_type="neumann")


def test_initial_displacement_wrong_shape_raises_value_error():
    """Initial displacement must provide one value per grid point."""
    with pytest.raises(ValueError):
        simulate_wave_equation_2d(
            t_final=0.01,
            nx=11,
            ny=9,
            initial_displacement=lambda X, Y: np.ones((8, 11)),
        )


def test_initial_velocity_wrong_shape_raises_value_error():
    """Initial velocity must provide one value per grid point."""
    with pytest.raises(ValueError):
        simulate_wave_equation_2d(
            t_final=0.01,
            nx=11,
            ny=9,
            initial_velocity=lambda X, Y: np.ones((8, 11)),
        )


def test_sine_mode_stays_bounded_for_stable_cfl():
    """A stable fixed-boundary sine mode should remain bounded."""
    result = simulate_wave_equation_2d(
        t_final=0.2,
        nx=31,
        ny=31,
        initial_displacement=lambda X, Y: sine_displacement_2d(
            X,
            Y,
            amplitude=1.0,
            mode_x=1,
            mode_y=1,
            width=1.0,
            height=1.0,
        ),
        initial_velocity=zero_initial_velocity_2d,
    )

    displacement = result["displacement"]

    assert np.max(np.abs(displacement)) <= 1.1


def test_default_gaussian_evolves_over_time():
    """A default displacement pulse should propagate, not remain static."""
    result = simulate_wave_equation_2d(t_final=0.2, nx=31, ny=31)

    displacement = result["displacement"]

    assert not np.allclose(displacement[0], displacement[-1])


def test_nonzero_initial_velocity_changes_first_step_in_expected_direction():
    """Positive initial velocity should move the first interior step upward."""
    result = simulate_wave_equation_2d(
        t_final=0.05,
        nx=21,
        ny=19,
        initial_displacement=lambda X, Y: np.zeros_like(X),
        initial_velocity=lambda X, Y: np.ones_like(X),
        boundary_values=0.0,
    )

    displacement = result["displacement"]

    assert np.all(displacement[1, 1:-1, 1:-1] > displacement[0, 1:-1, 1:-1])
