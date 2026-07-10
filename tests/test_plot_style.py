"""Tests for shared plotting style helpers."""

import matplotlib
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from visualization.plot_style import (
    DEFAULT_DPI,
    DEFAULT_LINE_WIDTH,
    DIVERGING_COLORMAP,
    ENGINEERING_COLOR_CYCLE,
    add_clean_colorbar,
    add_colorbar,
    apply_engineering_plot_style,
    apply_plot_style,
    create_streamlit_figure,
    create_streamlit_subplots,
    finalize_streamlit_figure,
    format_axes,
    format_engineering_axes,
    format_heatmap_axes,
    place_legends_outside,
    place_legend_outside,
    save_figure,
    set_equal_2d_axes,
    set_xy_plot_limits_with_margin,
    symmetric_color_limits,
    THERMAL_COLORMAP,
    WAVE_COLORMAP,
    WAVE_DIVERGING_COLORMAP,
)


def test_apply_plot_style_updates_rcparams_without_backend_change():
    backend_before = matplotlib.get_backend()

    apply_plot_style()

    assert matplotlib.get_backend() == backend_before
    assert plt.rcParams["savefig.dpi"] == 200
    assert plt.rcParams["axes.titlesize"] == 13
    assert plt.rcParams["axes.prop_cycle"].by_key()["color"] == list(
        ENGINEERING_COLOR_CYCLE
    )


def test_apply_engineering_plot_style_preserves_backend():
    backend_before = matplotlib.get_backend()

    apply_engineering_plot_style()

    assert matplotlib.get_backend() == backend_before
    assert plt.rcParams["figure.facecolor"] == "white"
    assert plt.rcParams["lines.linewidth"] == 2.0


def test_colormap_constants_are_importable_and_consistent():
    assert DEFAULT_DPI == 200
    assert DEFAULT_LINE_WIDTH == 2.0
    assert THERMAL_COLORMAP == "inferno"
    assert WAVE_COLORMAP == "coolwarm"
    assert DIVERGING_COLORMAP == "coolwarm"
    assert WAVE_DIVERGING_COLORMAP == DIVERGING_COLORMAP


def test_format_axes_sets_labels_grid_and_legend():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], label="Response")

    format_axes(ax, title="Title", xlabel="x", ylabel="y")

    assert ax.get_title() == "Title"
    assert ax.get_xlabel() == "x"
    assert ax.get_ylabel() == "y"
    assert ax.get_legend() is not None
    assert any(line.get_visible() for line in ax.get_xgridlines())

    plt.close(fig)


def test_format_engineering_axes_sets_subtle_spine_style():
    fig, ax = plt.subplots()

    format_engineering_axes(ax, title="Response", xlabel="t", ylabel="y")

    assert ax.get_title() == "Response"
    assert ax.spines["left"].get_linewidth() == 0.8
    assert any(line.get_visible() for line in ax.get_xgridlines())

    plt.close(fig)


def test_format_heatmap_axes_disables_grid():
    fig, ax = plt.subplots()

    format_heatmap_axes(ax, title="Heatmap")

    assert ax.get_title() == "Heatmap"
    assert ax.get_xlabel() == "x (m)"
    assert ax.get_ylabel() == "y (m)"
    assert not any(line.get_visible() for line in ax.get_xgridlines())

    plt.close(fig)


def test_save_figure_writes_file(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [1, 0])
    output_path = tmp_path / "plots" / "figure.png"

    saved_path = save_figure(fig, output_path)

    assert saved_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0

    plt.close(fig)


def test_set_equal_2d_axes_uses_matching_ranges():
    fig, ax = plt.subplots()
    ax.plot([0, 2], [0, 1])

    set_equal_2d_axes(ax)

    x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    assert x_range == pytest.approx(y_range)

    plt.close(fig)


def test_add_colorbar_returns_labeled_colorbar():
    fig, ax = plt.subplots()
    image = ax.imshow([[0, 1], [1, 0]])

    colorbar = add_colorbar(fig, image, ax, label="Value")

    assert colorbar.ax.get_ylabel() == "Value"

    plt.close(fig)


def test_add_clean_colorbar_returns_labeled_colorbar():
    fig, ax = plt.subplots()
    image = ax.imshow([[0, 1], [1, 0]])

    colorbar = add_clean_colorbar(fig, image, ax, label="Temperature")

    assert colorbar.ax.get_ylabel() == "Temperature"
    assert colorbar.outline.get_linewidth() == 0.8

    plt.close(fig)


def test_create_streamlit_figure_uses_requested_size():
    fig = create_streamlit_figure(width=9, height=5)

    assert tuple(fig.get_size_inches()) == (9.0, 5.0)
    assert fig.get_constrained_layout()

    plt.close(fig)


def test_create_streamlit_subplots_uses_streamlit_defaults():
    fig, axes = create_streamlit_subplots(2, 1, width=10, height=7)

    assert tuple(fig.get_size_inches()) == (10.0, 7.0)
    assert len(axes) == 2
    assert fig.get_constrained_layout()

    plt.close(fig)


def test_place_legend_outside_reserves_right_space():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], label="Response")

    legend = place_legend_outside(ax, location="right", frameon=False)

    assert legend is not None
    assert not legend.get_frame_on()
    assert ax.get_position().x1 <= 0.78

    plt.close(fig)


def test_place_legend_outside_supports_bottom_location():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], label="Response")

    legend = place_legend_outside(ax, location="bottom", ncol=1)

    assert legend is not None
    assert ax.get_position().y0 >= 0.24

    plt.close(fig)


def test_place_legends_outside_accepts_multiple_axes():
    fig, axes = plt.subplots(2, 1)
    axes[0].plot([0, 1], [0, 1], label="Response")
    axes[1].plot([0, 1], [1, 0], label="Error")

    legends = place_legends_outside(axes, location="right")

    assert len(legends) == 2

    plt.close(fig)


def test_place_legends_outside_accepts_single_axis_and_skips_empty_axes():
    fig, axes = plt.subplots(2, 1)
    axes[0].plot([0, 1], [0, 1], label="Response")

    single_axis_legends = place_legends_outside(axes[0], location="right")
    mixed_axis_legends = place_legends_outside(axes, location="right")

    assert len(single_axis_legends) == 1
    assert len(mixed_axis_legends) == 1
    assert axes[1].get_legend() is None

    plt.close(fig)


def test_set_xy_plot_limits_with_margin_keeps_equal_span():
    fig, ax = plt.subplots()

    set_xy_plot_limits_with_margin(ax, [0, 2], [0, 1], margin_fraction=0.1)

    x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    assert x_range == pytest.approx(y_range)
    assert ax.get_xlim()[0] < 0
    assert ax.get_xlim()[1] > 2

    plt.close(fig)


def test_symmetric_color_limits_centers_around_zero():
    lower, upper = symmetric_color_limits([-1.0, 0.5, 2.0])

    assert lower == pytest.approx(-2.0)
    assert upper == pytest.approx(2.0)


def test_finalize_streamlit_figure_returns_same_figure():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    finalized = finalize_streamlit_figure(fig)

    assert finalized is fig
    assert fig.get_facecolor() == (1.0, 1.0, 1.0, 1.0)

    plt.close(fig)
