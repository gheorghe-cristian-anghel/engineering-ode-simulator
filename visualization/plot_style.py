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
