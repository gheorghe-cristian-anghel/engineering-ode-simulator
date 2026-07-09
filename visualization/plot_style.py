"""Shared Matplotlib styling helpers for examples and visualization code."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

STREAMLIT_LINE_FIGSIZE = (9, 5)
STREAMLIT_HEATMAP_FIGSIZE = (8, 6)
STREAMLIT_XY_FIGSIZE = (9, 6)
STREAMLIT_MULTIPANEL_FIGSIZE = (10, 7)


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


def create_streamlit_figure(width=9, height=5, constrained=True):
    """Create a Streamlit-sized Matplotlib figure."""
    return plt.figure(figsize=(width, height), constrained_layout=constrained)


def create_streamlit_subplots(
    *args,
    width=9,
    height=5,
    constrained=True,
    **kwargs,
):
    """Create Streamlit-sized subplots with explicit figure dimensions."""
    kwargs.setdefault("figsize", (width, height))
    kwargs.setdefault("constrained_layout", constrained)
    return plt.subplots(*args, **kwargs)


def _finite_values(values):
    """Return finite flattened numeric values for layout limit helpers."""
    array = np.asarray(values, dtype=float).ravel()
    return array[np.isfinite(array)]


def set_xy_plot_limits_with_margin(
    ax,
    x_values,
    y_values,
    margin_fraction=0.08,
    equal_aspect=True,
):
    """Set readable 2D limits around path-like data with optional equal aspect."""
    x = _finite_values(x_values)
    y = _finite_values(y_values)
    if x.size == 0 or y.size == 0:
        return ax

    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    x_range = max(x_max - x_min, 1e-9)
    y_range = max(y_max - y_min, 1e-9)

    if equal_aspect:
        span = max(x_range, y_range)
        x_mid = 0.5 * (x_min + x_max)
        y_mid = 0.5 * (y_min + y_max)
        padding = span * margin_fraction
        ax.set_xlim(x_mid - 0.5 * span - padding, x_mid + 0.5 * span + padding)
        ax.set_ylim(y_mid - 0.5 * span - padding, y_mid + 0.5 * span + padding)
        ax.set_aspect("equal", adjustable="box")
        return ax

    ax.set_xlim(x_min - x_range * margin_fraction, x_max + x_range * margin_fraction)
    ax.set_ylim(y_min - y_range * margin_fraction, y_max + y_range * margin_fraction)
    return ax


def place_legend_outside(ax, location="right", ncol=1):
    """Place a legend outside an axes while reserving predictable figure space."""
    handles, labels = ax.get_legend_handles_labels()
    visible = [
        (handle, label)
        for handle, label in zip(handles, labels, strict=True)
        if label and not label.startswith("_")
    ]
    if not visible:
        return None

    handles, labels = zip(*visible, strict=True)
    fig = ax.figure
    if fig.get_constrained_layout():
        fig.set_constrained_layout(False)
    if hasattr(fig, "set_layout_engine"):
        fig.set_layout_engine(None)

    if location == "bottom":
        legend = ax.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.16),
            ncol=ncol,
            frameon=True,
        )
        fig.subplots_adjust(bottom=0.24)
        return legend

    legend = ax.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        ncol=ncol,
        frameon=True,
    )
    fig.subplots_adjust(right=0.78)
    return legend


def ensure_readable_axes(ax):
    """Apply final axis readability settings without changing plotted data."""
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.tick_params(axis="both", which="minor", labelsize=9)
    title = ax.title
    title.set_wrap(True)
    ax.xaxis.label.set_wrap(True)
    ax.yaxis.label.set_wrap(True)
    return ax


def finalize_streamlit_figure(fig):
    """Apply final cleanup before showing a figure in Streamlit."""
    fig.patch.set_facecolor("white")
    for ax in fig.axes:
        ensure_readable_axes(ax)
    return fig


def display_streamlit_figure(fig, caption=None):
    """Display a Matplotlib figure in Streamlit and close it afterwards."""
    import streamlit as st

    finalize_streamlit_figure(fig)
    st.pyplot(fig, width="stretch")
    if caption is not None:
        st.caption(caption)
    plt.close(fig)
