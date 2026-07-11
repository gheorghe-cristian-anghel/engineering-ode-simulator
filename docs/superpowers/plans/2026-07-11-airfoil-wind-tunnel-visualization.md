# Airfoil Wind-Tunnel Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a responsive Streamlit wind-tunnel page backed by a clearly labelled qualitative airfoil-flow plot.

**Architecture:** A new `visualization.airfoil_flow` module will generate NACA 0012/2412 outlines and a synthetic potential-flow/wake field, returning a Matplotlib figure with summary metrics. `streamlit_app.py` will only render controls and results, retaining the current sidebar domain/demo navigation. No existing model or numerical solver will change.

**Tech Stack:** Python 3.10+, NumPy, Matplotlib, Streamlit, pytest, Ruff.

## Global Constraints

- Do not modify existing numerical solvers.
- Describe the view as CFD-inspired, qualitative, and not a full Navier-Stokes or industrial-CFD solution.
- Do not add animation or commit changes.
- Use a safe default grid of 160 x 80 and cap the selectable maximum size.
- Use `width="stretch"` for current Streamlit figure sizing.

---

### Task 1: Qualitative airfoil-flow plotting API

**Files:**
- Create: `visualization/airfoil_flow.py`
- Test: `tests/test_airfoil_flow.py`

**Interfaces:**
- Produces: `plot_airfoil_flow_field(airfoil_code, angle_of_attack_deg, free_stream_velocity, circulation_strength, wake_strength, grid_shape, streamline_density) -> tuple[Figure, AirfoilFlowMetrics]`.
- Produces: immutable `AirfoilFlowMetrics` with speed extrema, qualitative wake strength, and grid shape.

- [ ] **Step 1: Write the failing test**

```python
def test_plot_airfoil_flow_field_returns_readable_figure_and_metrics():
    figure, metrics = plot_airfoil_flow_field(grid_shape=(80, 160))
    assert metrics.grid_shape == (80, 160)
    assert metrics.max_speed >= metrics.min_speed >= 0.0
    assert len(figure.axes) >= 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_airfoil_flow.py -v`

Expected: import failure because `visualization.airfoil_flow` does not exist.

- [ ] **Step 3: Write minimal implementation**

Create the module with NACA 0012/2412 geometry, a bounded synthetic potential-flow/circulation/wake field, explicit validation, a colour-mapped speed plot, streamlines, and the returned metrics.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_airfoil_flow.py -v`

Expected: all airfoil visualization tests pass.

### Task 2: Streamlit page integration

**Files:**
- Modify: `streamlit_app.py`
- Test: `tests/test_airfoil_flow.py`

**Interfaces:**
- Consumes: `plot_airfoil_flow_field` and `AirfoilFlowMetrics` from Task 1.
- Produces: `render_airfoil_wind_tunnel()` reachable via the `Fluid Flow / Wind Tunnel` sidebar domain.

- [ ] **Step 1: Write the failing integration-presence test**

```python
def test_streamlit_app_includes_airfoil_wind_tunnel_navigation():
    source = Path("streamlit_app.py").read_text(encoding="utf-8")
    assert '"Fluid Flow / Wind Tunnel"' in source
    assert "render_airfoil_wind_tunnel" in source
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_airfoil_flow.py::test_streamlit_app_includes_airfoil_wind_tunnel_navigation -v`

Expected: assertion failure because the navigation/page is absent.

- [ ] **Step 3: Write minimal integration**

Import the new plotter, add one domain/demo pair, add controls and metric cards, render the figure with `st.pyplot(fig, width="stretch")`, close the figure, and show an assumptions expander. Keep all computation on the selected page only.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_airfoil_flow.py -v`

Expected: all airfoil visualization and navigation-presence tests pass.

### Task 3: Documentation and verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add one concise README bullet**

Mention the qualitative Streamlit airfoil wind-tunnel demo and its educational/non-CFD scope.

- [ ] **Step 2: Check scoped lint and formatting**

Run: `python -m ruff check streamlit_app.py visualization/airfoil_flow.py tests/test_airfoil_flow.py` and `python -m ruff format --check streamlit_app.py visualization/airfoil_flow.py tests/test_airfoil_flow.py`

- [ ] **Step 3: Run full verification**

Run: `python -m pytest -v --basetemp=".pytest_tmp_$([DateTimeOffset]::UtcNow.ToUnixTimeSeconds())" -p no:cacheprovider`

Run: `python -m streamlit run streamlit_app.py`

Expected: tests pass and Streamlit starts without import/runtime errors. Do not commit.
