import pytest

from models.mass_spring_damper import (
    damping_ratio,
    mechanical_energy,
    natural_frequency,
    simulate_mass_spring_damper,
)


def test_initial_displacement_and_velocity_match_inputs():
    """The simulated state should start at x0 and v0."""
    m = 1
    c = 0.4
    k = 4
    x0 = 1
    v0 = 0

    t, displacement, velocity = simulate_mass_spring_damper(
        m,
        c,
        k,
        x0,
        v0,
        (0, 20),
        200,
    )

    assert t[0] == pytest.approx(0)
    assert displacement[0] == pytest.approx(x0)
    assert velocity[0] == pytest.approx(v0)


def test_natural_frequency():
    """For m = 1 kg and k = 4 N/m, natural frequency should be 2 rad/s."""
    assert natural_frequency(1, 4) == pytest.approx(2)


def test_damping_ratio():
    """For m = 1, c = 0.4, and k = 4, damping ratio should be 0.1."""
    assert damping_ratio(1, 0.4, 4) == pytest.approx(0.1)


def test_energy_is_conserved_without_damping_or_external_force():
    """With c = 0 and F(t) = 0, total mechanical energy should stay constant."""
    m = 1
    c = 0
    k = 4
    x0 = 1
    v0 = 0

    _, displacement, velocity = simulate_mass_spring_damper(
        m,
        c,
        k,
        x0,
        v0,
        (0, 20),
        500,
    )

    energy = mechanical_energy(displacement, velocity, m, k)

    assert energy[-1] == pytest.approx(energy[0], rel=0.01)


def test_energy_decreases_with_damping():
    """With damping, final mechanical energy should be less than initial energy."""
    m = 1
    c = 0.4
    k = 4
    x0 = 1
    v0 = 0

    _, displacement, velocity = simulate_mass_spring_damper(
        m,
        c,
        k,
        x0,
        v0,
        (0, 20),
        500,
    )

    energy = mechanical_energy(displacement, velocity, m, k)

    assert energy[-1] < energy[0]


def test_invalid_physical_parameters_raise_value_error():
    """Mass, damping, and stiffness should reject unphysical values."""
    with pytest.raises(ValueError):
        simulate_mass_spring_damper(0, 0.1, 4, 1, 0, (0, 1), 10)

    with pytest.raises(ValueError):
        simulate_mass_spring_damper(1, -0.1, 4, 1, 0, (0, 1), 10)

    with pytest.raises(ValueError):
        simulate_mass_spring_damper(1, 0.1, 0, 1, 0, (0, 1), 10)
