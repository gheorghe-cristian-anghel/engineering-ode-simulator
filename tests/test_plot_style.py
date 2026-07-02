"""Tests for shared plotting style helpers."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from visualization.plot_style import (
    add_colorbar,
    apply_plot_style,
    format_axes,
    save_figure,
    set_equal_2d_axes,
)


def test_apply_plot_style_updates_rcparams_without_backend_change():
    backend_before = matplotlib.get_backend()

    apply_plot_style()

    assert matplotlib.get_backend() == backend_before
    assert plt.rcParams["savefig.dpi"] == 200
    assert plt.rcParams["axes.titlesize"] == 13


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
