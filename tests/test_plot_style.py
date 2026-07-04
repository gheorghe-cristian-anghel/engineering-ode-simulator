"""Tests for shared plotting style helpers."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from visualization.plot_style import (
    add_colorbar,
    add_clean_colorbar,
    apply_engineering_plot_style,
    apply_plot_style,
    format_axes,
    format_engineering_axes,
    format_heatmap_axes,
    finalize_streamlit_figure,
    save_figure,
    set_equal_2d_axes,
)


def test_apply_plot_style_updates_rcparams_without_backend_change():
    backend_before = matplotlib.get_backend()

    apply_plot_style()

    assert matplotlib.get_backend() == backend_before
    assert plt.rcParams["savefig.dpi"] == 200
    assert plt.rcParams["axes.titlesize"] == 13


def test_apply_engineering_plot_style_preserves_backend():
    backend_before = matplotlib.get_backend()

    apply_engineering_plot_style()

    assert matplotlib.get_backend() == backend_before
    assert plt.rcParams["figure.facecolor"] == "white"
    assert plt.rcParams["lines.linewidth"] == 2.0


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
    assert x_range == y_range

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


def test_finalize_streamlit_figure_returns_same_figure():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    finalized = finalize_streamlit_figure(fig)

    assert finalized is fig
    assert fig.get_facecolor() == (1.0, 1.0, 1.0, 1.0)

    plt.close(fig)
