"""Runtime regression tests for Streamlit plotting helper imports."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import visualization.plot_style as ps


def test_plot_style_exports_streamlit_runtime_surface():
    """Plot helpers used through `visualization.plot_style as ps` are present."""
    required_names = (
        "THERMAL_COLORMAP",
        "WAVE_COLORMAP",
        "DIVERGING_COLORMAP",
        "place_legend_outside",
        "place_legends_outside",
    )

    missing_names = [name for name in required_names if not hasattr(ps, name)]

    assert missing_names == []
    assert ps.THERMAL_COLORMAP == "inferno"
    assert ps.WAVE_COLORMAP == "coolwarm"
    assert ps.DIVERGING_COLORMAP == "coolwarm"


def test_place_legends_outside_accepts_single_axis():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], label="Response")

    legends = ps.place_legends_outside(ax, location="right")

    assert len(legends) == 1

    plt.close(fig)


def test_place_legends_outside_accepts_axes_array_and_skips_empty_axes():
    fig, axes = plt.subplots(2, 1)
    axes[0].plot([0, 1], [0, 1], label="Response")

    legends = ps.place_legends_outside(axes, location="right")

    assert len(legends) == 1
    assert axes[1].get_legend() is None

    plt.close(fig)


def test_place_legends_outside_ignores_none():
    assert ps.place_legends_outside(None) == []


def test_place_legend_outside_supports_best_fallback():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [1, 0], label="Response")

    legend = ps.place_legend_outside(ax, location="best")

    assert legend is not None
    assert legend._loc == 0

    plt.close(fig)
