import numpy as np
import pytest
from matplotlib import animation

from visualization.inverted_pendulum_animation import animate_inverted_pendulum


def test_invalid_state_shape_raises_value_error():
    """Animation helper should require four cart-pole state columns."""
    t = np.linspace(0.0, 1.0, 5)
    states = np.zeros((5, 3))

    with pytest.raises(ValueError):
        animate_inverted_pendulum(t, states, show=False)


def test_invalid_time_and_state_lengths_raise_value_error():
    """Time and state arrays should have the same number of samples."""
    t = np.linspace(0.0, 1.0, 5)
    states = np.zeros((4, 4))

    with pytest.raises(ValueError):
        animate_inverted_pendulum(t, states, show=False)


def test_returns_func_animation_when_show_is_false():
    """A valid trajectory should create a Matplotlib FuncAnimation object."""
    t = np.linspace(0.0, 1.0, 5)
    states = np.zeros((5, 4))
    states[:, 0] = np.linspace(0.0, 0.2, 5)

    anim = animate_inverted_pendulum(t, states, show=False)

    assert isinstance(anim, animation.FuncAnimation)


def test_invalid_save_extension_raises_value_error():
    """Only GIF and MP4 animation outputs should be accepted."""
    t = np.linspace(0.0, 1.0, 5)
    states = np.zeros((5, 4))

    with pytest.raises(ValueError):
        animate_inverted_pendulum(t, states, save_path="animation.txt", show=False)
