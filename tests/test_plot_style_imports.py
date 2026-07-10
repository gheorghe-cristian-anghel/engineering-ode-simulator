"""Regression tests for the Streamlit plot-style import surface."""

from visualization import plot_style as ps


def test_streamlit_plot_style_api_is_available():
    """Every plot-style name used by streamlit_app should exist on the module."""
    required_names = (
        "THERMAL_COLORMAP",
        "DIVERGING_COLORMAP",
        "WAVE_COLORMAP",
        "DEFAULT_LINE_WIDTH",
        "DEFAULT_DPI",
        "add_clean_colorbar",
        "apply_engineering_plot_style",
        "create_streamlit_subplots",
        "display_streamlit_figure",
        "format_engineering_axes",
        "format_heatmap_axes",
        "place_legends_outside",
        "place_legend_outside",
        "set_equal_2d_axes",
        "set_xy_plot_limits_with_margin",
        "symmetric_color_limits",
    )

    missing_names = [name for name in required_names if not hasattr(ps, name)]

    assert missing_names == []
    assert ps.THERMAL_COLORMAP == "inferno"
    assert ps.WAVE_COLORMAP == "coolwarm"
    assert ps.DIVERGING_COLORMAP == "coolwarm"
    assert ps.DEFAULT_LINE_WIDTH == 2.0
    assert ps.DEFAULT_DPI == 200
