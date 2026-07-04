"""Shared Matplotlib styling helpers for examples and visualization code."""

from pathlib import Path

import matplotlib.pyplot as plt


def apply_plot_style():
    """Apply consistent Matplotlib defaults without changing the backend."""
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.titleweight": "bold",
            "axes.grid": False,
            "grid.color": "0.82",
            "grid.linestyle": "--",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.8,
            "legend.fontsize": 9,
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "legend.edgecolor": "0.85",
            "lines.linewidth": 1.8,
            "lines.markersize": 5,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


def apply_engineering_plot_style():
    """Apply polished engineering plot defaults without changing the backend."""
    apply_plot_style()
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "0.35",
            "axes.linewidth": 0.8,
            "axes.titlepad": 10,
            "axes.labelpad": 7,
            "grid.color": "0.86",
            "grid.linestyle": "--",
            "grid.linewidth": 0.65,
            "grid.alpha": 0.85,
            "legend.borderpad": 0.6,
            "legend.handlelength": 2.2,
            "legend.labelspacing": 0.45,
            "lines.linewidth": 2.0,
        }
    )


def format_axes(ax, title=None, xlabel=None, ylabel=None, grid=True):
    """Format one Matplotlib axes object with shared labels, grid, and legend."""
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if grid:
        ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.8)
    else:
        ax.grid(False)

    handles, labels = ax.get_legend_handles_labels()
    visible_labels = [
        label for label in labels if label and not label.startswith("_")
    ]
    if handles and visible_labels:
        ax.legend()

    return ax


def format_engineering_axes(ax, title=None, xlabel=None, ylabel=None, grid=True):
    """Format one axes object for clean engineering dashboard plots."""
    format_axes(ax, title=title, xlabel=xlabel, ylabel=ylabel, grid=grid)
    ax.tick_params(axis="both", which="major", colors="0.25", length=4, width=0.8)
    ax.tick_params(axis="both", which="minor", colors="0.35", length=2, width=0.6)
    for spine in ax.spines.values():
        spine.set_color("0.45")
        spine.set_linewidth(0.8)
    return ax


def format_heatmap_axes(ax, title=None, xlabel="x (m)", ylabel="y (m)"):
    """Format image-style axes without overlaying grid lines."""
    return format_engineering_axes(
        ax,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        grid=False,
    )


def save_figure(fig, path, dpi=200):
    """Save a figure at high DPI with tight layout and parent directories."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    return output_path


def set_equal_2d_axes(ax):
    """Set equal data scaling around the current 2D x/y limits."""
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)
    radius = 0.5 * max(x_max - x_min, y_max - y_min)

    ax.set_xlim(x_mid - radius, x_mid + radius)
    ax.set_ylim(y_mid - radius, y_mid + radius)
    ax.set_aspect("equal", adjustable="box")
    return ax


def add_colorbar(fig, image, ax, label=None):
    """Add a consistently labeled colorbar for an image-like plot."""
    colorbar = fig.colorbar(image, ax=ax)
    if label is not None:
        colorbar.set_label(label)
    return colorbar


def add_clean_colorbar(fig, image, ax, label=None):
    """Add a compact, consistently styled colorbar for Streamlit figures."""
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    if label is not None:
        colorbar.set_label(label)
    colorbar.ax.tick_params(labelsize=9, colors="0.25")
    colorbar.outline.set_edgecolor("0.65")
    colorbar.outline.set_linewidth(0.8)
    return colorbar


def finalize_streamlit_figure(fig):
    """Apply final layout cleanup before showing a figure in Streamlit."""
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    return fig
