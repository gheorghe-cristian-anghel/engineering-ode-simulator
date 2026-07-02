import numpy as np
import pytest

from models.heat_equation_1d import (
    check_heat_stability,
    gaussian_initial_condition,
    heat_stability_number,
    simulate_heat_equation_1d,
    sine_initial_condition,
    step_initial_condition,
)


def test_heat_stability_number_matches_formula():
    """The stability number should be alpha*dt/dx^2."""
    alpha = 0.01
    dt = 0.002
    dx = 0.01

    assert heat_stability_number(alpha, dt, dx) == pytest.approx(
        alpha * dt / dx**2
    )


def test_stable_parameters_pass_stability_check():
    """Parameters with r <= 0.5 should pass the stability check."""
    assert check_heat_stability(alpha=0.01, dt=0.001, dx=0.01)


def test_unstable_parameters_fail_stability_check():
    """Parameters with r > 0.5 should fail when not raising."""
    assert not check_heat_stability(alpha=1.0, dt=0.01, dx=0.1)


def test_unstable_parameters_raise_when_requested():
    """The helper should raise for unstable parameters when configured."""
    with pytest.raises(ValueError):
        check_heat_stability(
            alpha=1.0,
            dt=0.01,
            dx=0.1,
            raise_error=True,
        )


def test_gaussian_initial_condition_returns_correct_shape():
    """The Gaussian helper should return one value per x point."""
    x = np.linspace(0.0, 1.0, 21)

    temperature = gaussian_initial_condition(x)

    assert temperature.shape == x.shape
    assert np.all(np.isfinite(temperature))


def test_step_initial_condition_returns_correct_shape():
    """The step helper should return one value per x point."""
    x = np.linspace(0.0, 1.0, 21)

    temperature = step_initial_condition(x)

    assert temperature.shape == x.shape
    assert np.all(np.isfinite(temperature))


def test_sine_initial_condition_returns_correct_shape():
    """The sine helper should return one value per x point."""
    x = np.linspace(0.0, 1.0, 21)

    temperature = sine_initial_condition(x)

    assert temperature.shape == x.shape
    assert np.all(np.isfinite(temperature))


def test_simulate_heat_equation_returns_expected_keys():
    """The solver should return the documented result dictionary fields."""
    result = simulate_heat_equation_1d(t_final=0.1, num_points=21)

    expected_keys = {
        "x",
        "t",
        "temperature",
        "alpha",
        "dx",
        "dt",
        "stability_number",
        "boundary_type",
        "boundary_values",
    }

    assert expected_keys.issubset(result.keys())


def test_temperature_array_has_expected_shape():
    """Temperature should have shape (num_time_steps, num_points)."""
    num_points = 21

    result = simulate_heat_equation_1d(t_final=0.1, num_points=num_points)

    assert result["temperature"].shape == (len(result["t"]), num_points)


def test_x_and_t_arrays_are_one_dimensional():
    """The spatial and time grids should be one-dimensional arrays."""
    result = simulate_heat_equation_1d(t_final=0.1, num_points=21)

    assert result["x"].ndim == 1
    assert result["t"].ndim == 1


def test_dirichlet_boundaries_remain_fixed():
    """Fixed-temperature boundaries should remain fixed for all time."""
    result = simulate_heat_equation_1d(
        t_final=0.1,
        num_points=21,
        boundary_values=(2.0, -1.0),
    )

    temperature = result["temperature"]

    assert np.allclose(temperature[:, 0], 2.0)
    assert np.allclose(temperature[:, -1], -1.0)


def test_neumann_boundaries_have_zero_gradient():
    """Insulated boundaries should copy their nearest interior neighbors."""
    result = simulate_heat_equation_1d(
        t_final=0.1,
        num_points=21,
        boundary_type="neumann",
    )

    temperature = result["temperature"]

    assert np.allclose(temperature[:, 0], temperature[:, 1])
    assert np.allclose(temperature[:, -1], temperature[:, -2])


def test_gaussian_pulse_peak_decreases_over_time():
    """A hot Gaussian pulse should diffuse and lose peak temperature."""
    result = simulate_heat_equation_1d(
        t_final=1.0,
        num_points=101,
        initial_condition=lambda x: gaussian_initial_condition(
            x,
            center=0.5,
            width=0.08,
            amplitude=1.0,
        ),
        boundary_values=(0.0, 0.0),
    )

    temperature = result["temperature"]

    assert np.max(temperature[-1]) < np.max(temperature[0])


def test_invalid_length_raises_value_error():
    """Rod length must be positive."""
    with pytest.raises(ValueError):
        simulate_heat_equation_1d(length=0.0)


def test_invalid_alpha_raises_value_error():
    """Thermal diffusivity must be positive."""
    with pytest.raises(ValueError):
        simulate_heat_equation_1d(alpha=0.0)


def test_invalid_t_final_raises_value_error():
    """Final simulation time must be positive."""
    with pytest.raises(ValueError):
        simulate_heat_equation_1d(t_final=0.0)


def test_invalid_num_points_raises_value_error():
    """The grid needs at least two boundaries and one interior point."""
    with pytest.raises(ValueError):
        simulate_heat_equation_1d(num_points=2)


def test_invalid_dt_raises_value_error():
    """Provided time step must be positive."""
    with pytest.raises(ValueError):
        simulate_heat_equation_1d(dt=0.0)


def test_invalid_boundary_type_raises_value_error():
    """Only supported boundary condition types should be accepted."""
    with pytest.raises(ValueError):
        simulate_heat_equation_1d(boundary_type="periodic")


def test_unstable_dt_raises_value_error_when_enforced():
    """The simulator should reject unstable explicit time steps by default."""
    with pytest.raises(ValueError):
        simulate_heat_equation_1d(
            length=1.0,
            alpha=1.0,
            t_final=0.1,
            num_points=11,
            dt=0.01,
            enforce_stability=True,
        )


def test_initial_condition_wrong_shape_raises_value_error():
    """Initial condition must provide one temperature per grid point."""
    with pytest.raises(ValueError):
        simulate_heat_equation_1d(
            t_final=0.1,
            num_points=21,
            initial_condition=lambda x: np.ones(len(x) - 1),
        )
