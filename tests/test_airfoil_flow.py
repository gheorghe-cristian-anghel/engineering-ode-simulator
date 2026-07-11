"""Tests for the qualitative airfoil-flow visualization."""

from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from visualization.airfoil_flow import naca_airfoil_outline, plot_airfoil_flow_field


def test_plot_airfoil_flow_field_returns_readable_figure_and_metrics():
    """The default visualization provides a speed field and summary metrics."""
    figure, metrics = plot_airfoil_flow_field(grid_shape=(80, 160))

    assert metrics.grid_shape == (80, 160)
    assert metrics.max_speed >= metrics.min_speed >= 0.0
    assert len(figure.axes) >= 2

    plt.close(figure)


def test_streamlit_app_includes_airfoil_wind_tunnel_navigation():
    """The qualitative flow page is reachable from the existing sidebar."""
    source = Path("streamlit_app.py").read_text(encoding="utf-8")

    assert '"Fluid Flow / Wind Tunnel"' in source
    assert "render_airfoil_wind_tunnel" in source


def test_naca_airfoil_outline_supports_symmetric_and_cambered_profiles():
    """Both selectable NACA profiles produce closed, unit-chord geometry."""
    symmetric_x, symmetric_y = naca_airfoil_outline("NACA 0012")
    cambered_x, cambered_y = naca_airfoil_outline("2412")

    assert symmetric_x[0] == pytest.approx(symmetric_x[-1])
    assert symmetric_y[0] == pytest.approx(symmetric_y[-1])
    assert cambered_y.max() > abs(cambered_y.min())


def test_plot_airfoil_flow_field_rejects_unsafe_grid_resolution():
    """The visualization caps grids to keep interactive redraws responsive."""
    with pytest.raises(ValueError, match="interactive limit"):
        plot_airfoil_flow_field(grid_shape=(200, 400))
