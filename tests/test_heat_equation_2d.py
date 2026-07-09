import numpy as np
import pytest

from models.heat_equation_2d import (
    apply_2d_boundary_conditions,
    check_heat_stability_2d,
    create_2d_grid,
    gaussian_hotspot_2d,
    heat_stability_numbers_2d,
    rectangular_hot_region_2d,
    simulate_heat_equation_2d,
    sine_initial_condition_2d,
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


def test_heat_stability_numbers_2d_match_formula():
    """The stability numbers should be alpha*dt/dx^2 and alpha*dt/dy^2."""
    alpha = 0.01
    dt = 0.002
    dx = 0.02
    dy = 0.04

    rx, ry, stability_sum = heat_stability_numbers_2d(alpha, dt, dx, dy)

    assert rx == pytest.approx(alpha * dt / dx**2)
    assert ry == pytest.approx(alpha * dt / dy**2)
    assert stability_sum == pytest.approx(rx + ry)


def test_stable_parameters_pass_stability_check():
    """Parameters with rx + ry <= 0.5 should pass the stability check."""
    assert check_heat_stability_2d(alpha=0.01, dt=0.001, dx=0.01, dy=0.01)


def test_unstable_parameters_fail_stability_check():
    """Parameters with rx + ry > 0.5 should fail when not raising."""
    assert not check_heat_stability_2d(alpha=1.0, dt=0.01, dx=0.1, dy=0.1)


def test_unstable_parameters_raise_when_requested():
    """The helper should raise for unstable parameters when configured."""
    with pytest.raises(ValueError):
        check_heat_stability_2d(
            alpha=1.0,
            dt=0.01,
            dx=0.1,
            dy=0.1,
            raise_error=True,
        )


def test_gaussian_hotspot_2d_returns_shape_ny_by_nx():
    """Gaussian helper should return one temperature per grid point."""
    _, _, X, Y, _, _ = create_2d_grid(nx=7, ny=5)

    temperature = gaussian_hotspot_2d(X, Y)

    assert temperature.shape == (5, 7)
    assert np.all(np.isfinite(temperature))


def test_rectangular_hot_region_2d_returns_shape_ny_by_nx():
    """Rectangular region helper should return one temperature per grid point."""
    _, _, X, Y, _, _ = create_2d_grid(nx=7, ny=5)

    temperature = rectangular_hot_region_2d(X, Y)

    assert temperature.shape == (5, 7)
    assert np.max(temperature) == pytest.approx(1.0)
    assert np.min(temperature) == pytest.approx(0.0)


def test_sine_initial_condition_2d_returns_shape_ny_by_nx():
    """Sine helper should return one temperature per grid point."""
    _, _, X, Y, _, _ = create_2d_grid(nx=7, ny=5)

    temperature = sine_initial_condition_2d(X, Y)

    assert temperature.shape == (5, 7)
    assert np.all(np.isfinite(temperature))


def test_apply_2d_boundary_conditions_fixes_dirichlet_boundaries():
    """Dirichlet boundary values should be written to all four edges."""
    temperature = np.ones((5, 7))

    apply_2d_boundary_conditions(
        temperature,
        boundary_type="dirichlet",
        boundary_values=(1.0, 2.0, 3.0, 4.0),
    )

    assert np.allclose(temperature[1:-1, 0], 1.0)
    assert np.allclose(temperature[1:-1, -1], 2.0)
    assert np.allclose(temperature[0, :], 3.0)
    assert np.allclose(temperature[-1, :], 4.0)


def test_simulate_heat_equation_2d_returns_expected_keys():
    """The solver should return the documented result dictionary fields."""
    result = simulate_heat_equation_2d(t_final=0.01, nx=11, ny=9)

    expected_keys = {
        "x",
        "y",
        "X",
        "Y",
        "t",
        "temperature",
        "alpha",
        "dx",
        "dy",
        "dt",
        "rx",
        "ry",
        "stability_sum",
        "boundary_type",
        "boundary_values",
        "store_every",
    }

    assert expected_keys.issubset(result.keys())


def test_temperature_array_has_expected_stored_shape():
    """Stored temperature should have shape (stored_times, ny, nx)."""
    nx = 11
    ny = 9

    result = simulate_heat_equation_2d(t_final=0.01, nx=nx, ny=ny)

    assert result["temperature"].shape == (len(result["t"]), ny, nx)


def test_fixed_boundaries_remain_fixed():
    """Fixed-temperature boundaries should remain fixed for all stored times."""
    result = simulate_heat_equation_2d(
        t_final=0.01,
        nx=11,
        ny=9,
        boundary_values=(1.0, 2.0, 3.0, 4.0),
    )

    temperature = result["temperature"]

    assert np.allclose(temperature[:, 1:-1, 0], 1.0)
    assert np.allclose(temperature[:, 1:-1, -1], 2.0)
    assert np.allclose(temperature[:, 0, :], 3.0)
    assert np.allclose(temperature[:, -1, :], 4.0)


def test_gaussian_hotspot_peak_decreases_over_time():
    """A hot spot should diffuse and lose peak temperature over time."""
    result = simulate_heat_equation_2d(
        t_final=0.2,
        nx=31,
        ny=31,
        initial_condition=lambda X, Y: gaussian_hotspot_2d(
            X,
            Y,
            center=(0.5, 0.5),
            width=0.08,
            amplitude=1.0,
        ),
    )

    temperature = result["temperature"]

    assert np.max(temperature[-1]) < np.max(temperature[0])


def test_stable_solution_remains_finite_and_respects_rx_plus_ry_limit():
    """Stable 2D diffusion should remain finite with rx + ry <= 0.5."""
    result = simulate_heat_equation_2d(t_final=0.05, nx=21, ny=19)

    assert result["stability_sum"] <= 0.5
    assert result["rx"] + result["ry"] == pytest.approx(result["stability_sum"])
    assert np.all(np.isfinite(result["temperature"]))


def test_stable_2d_heat_solution_satisfies_discrete_maximum_principle():
    """Stable 2D diffusion with fixed boundaries should not create new extrema."""
    result = simulate_heat_equation_2d(
        t_final=0.05,
        nx=21,
        ny=19,
        initial_condition=lambda X, Y: sine_initial_condition_2d(
            X,
            Y,
            amplitude=1.0,
            mode_x=1,
            mode_y=1,
        ),
        boundary_values=0.0,
    )
    temperature = result["temperature"]

    assert np.max(temperature) <= np.max(temperature[0]) + 1e-12
    assert np.min(temperature) >= np.min(temperature[0]) - 1e-12


def test_unstable_dt_raises_value_error_when_enforced():
    """The solver should reject unstable explicit time steps by default."""
    with pytest.raises(ValueError):
        simulate_heat_equation_2d(
            alpha=1.0,
            t_final=0.1,
            nx=11,
            ny=11,
            dt=0.01,
            enforce_stability=True,
        )


def test_invalid_width_raises_value_error():
    """Plate width must be positive."""
    with pytest.raises(ValueError):
        simulate_heat_equation_2d(width=0.0)


def test_invalid_height_raises_value_error():
    """Plate height must be positive."""
    with pytest.raises(ValueError):
        simulate_heat_equation_2d(height=0.0)


def test_invalid_alpha_raises_value_error():
    """Thermal diffusivity must be positive."""
    with pytest.raises(ValueError):
        simulate_heat_equation_2d(alpha=0.0)


def test_invalid_nx_raises_value_error():
    """The x grid needs two boundaries and at least one interior point."""
    with pytest.raises(ValueError):
        simulate_heat_equation_2d(nx=2)


def test_invalid_ny_raises_value_error():
    """The y grid needs two boundaries and at least one interior point."""
    with pytest.raises(ValueError):
        simulate_heat_equation_2d(ny=2)


def test_invalid_t_final_raises_value_error():
    """Final simulation time must be positive."""
    with pytest.raises(ValueError):
        simulate_heat_equation_2d(t_final=0.0)


def test_invalid_dt_raises_value_error():
    """Provided time step must be positive."""
    with pytest.raises(ValueError):
        simulate_heat_equation_2d(dt=0.0)


def test_invalid_store_every_raises_value_error():
    """store_every must be at least one."""
    with pytest.raises(ValueError):
        simulate_heat_equation_2d(store_every=0)


def test_invalid_boundary_type_raises_value_error():
    """Only supported boundary condition types should be accepted."""
    with pytest.raises(ValueError):
        simulate_heat_equation_2d(boundary_type="periodic")


def test_initial_condition_wrong_shape_raises_value_error():
    """Initial condition must provide one temperature per grid point."""
    with pytest.raises(ValueError):
        simulate_heat_equation_2d(
            t_final=0.01,
            nx=11,
            ny=9,
            initial_condition=lambda X, Y: np.ones((8, 11)),
        )


def test_neumann_boundaries_copy_nearest_interior_values():
    """Optional Neumann boundaries should impose zero-gradient edges."""
    result = simulate_heat_equation_2d(
        t_final=0.01,
        nx=11,
        ny=9,
        boundary_type="neumann",
    )

    temperature = result["temperature"]

    assert np.allclose(temperature[:, :, 0], temperature[:, :, 1])
    assert np.allclose(temperature[:, :, -1], temperature[:, :, -2])
    assert np.allclose(temperature[:, 0, :], temperature[:, 1, :])
    assert np.allclose(temperature[:, -1, :], temperature[:, -2, :])
