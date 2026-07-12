# Scientific Visualization Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reusable, browser-native scientific plots and integrate the existing Heat, UAV, and Kalman simulations in the React frontend.

**Architecture:** Canvas owns regular scalar-field rendering; SVG owns time and geometric plots. Small plot-specific components share focused types, validation, bounds, decimation, and panel presentation helpers. Existing API contracts and numerical backends remain unchanged.

**Tech Stack:** React 19, TypeScript, Canvas 2D, SVG, Vitest, Testing Library, FastAPI/Python test suite.

## Global Constraints

- Do not alter numerical equations or existing API fields; any API extension must be additive and optional.
- Do not add animations, C++, or a visualization dependency.
- Treat fields as `field[y][x]`; render y increasing upward.
- Use response-supplied units and safely exclude invalid samples.
- Heat grids are bounded at 65,536 cells; paths/series must avoid copying or rendering excessive vertices.
- Do not create a git commit.

---

### Task 1: Plot contracts and pure scientific transforms

**Files:**
- Create: `frontend/src/components/plots/plotTypes.ts`
- Create: `frontend/src/components/plots/plotTransforms.ts`
- Test: `frontend/src/components/plots/plotTransforms.test.ts`

**Interfaces:**
- Produces `resolveDomain`, `buildPolyline`, `getEqualAspectBounds`, and shared plot types used by Tasks 2–4.

- [ ] Write failing tests for fixed/symmetric domains, non-finite filtering, bounded line decimation, and equal-scale geometry.
- [ ] Run `npm run test -- plotTransforms.test.ts` and confirm the feature imports fail.
- [ ] Implement immutable transform helpers that retain endpoints and avoid duplicate large arrays.
- [ ] Re-run the focused test file and confirm it passes.

### Task 2: Heatmap canvas with scientific annotation

**Files:**
- Create: `frontend/src/components/plots/PlotPanel.tsx`
- Create: `frontend/src/components/plots/HeatmapPlot.tsx`
- Test: `frontend/src/components/plots/HeatmapPlot.test.tsx`
- Modify: `frontend/src/features/heat2d/Heat2DPage.tsx`
- Modify: `frontend/src/components/HeatmapCanvas.tsx`

**Interfaces:**
- Consumes `HeatmapPlot` field coordinates, units, title, and color domain.
- Produces an accessible canvas plot with visible x/y axes, colorbar, and point probe.

- [ ] Write failing tests that assert the field dimensions/units are announced and the first y row is associated with the lower y bound.
- [ ] Run focused tests and observe failure before implementation.
- [ ] Implement the Canvas/SVG composite, validate rectangular finite matrices, invert rows only at the raster write, and expose automatic/fixed/symmetric ranges.
- [ ] Replace `HeatmapCanvas` usage with `HeatmapPlot`, retaining a compatibility wrapper if existing imports require it.
- [ ] Re-run focused tests.

### Task 3: Reusable SVG time-series and estimation composition

**Files:**
- Create: `frontend/src/components/plots/TimeSeriesPlot.tsx`
- Create: `frontend/src/features/estimation/StateEstimationView.tsx`
- Create: `frontend/src/features/estimation/KalmanPage.tsx`
- Create: `frontend/src/api/kalman.ts`
- Create: `frontend/src/features/estimation/useKalmanSimulation.ts`
- Test: `frontend/src/components/plots/TimeSeriesPlot.test.tsx`
- Test: `frontend/src/features/estimation/StateEstimationView.test.tsx`

**Interfaces:**
- `TimeSeriesPlot` consumes `TimeSeries` with names, values, units, and reference conventions.
- `StateEstimationView` consumes the existing `KalmanResponse` contract.

- [ ] Write failing tests for legends, dashed reference appearance, unit-aware axes, finite sample filtering, and no-data behavior.
- [ ] Verify focused tests fail.
- [ ] Implement SVG paths with hover readout, external legend, memoized transformed data, and index-based vertex budget.
- [ ] Implement Kalman API/hook/page and compose speed/current/error panels; only render uncertainty if supplied by a backward-compatible future response.
- [ ] Re-run the focused test files.

### Task 4: UAV path visual and page integration

**Files:**
- Create: `frontend/src/components/plots/XYPathPlot.tsx`
- Create: `frontend/src/features/uav/UavPage.tsx`
- Create: `frontend/src/api/uav.ts`
- Create: `frontend/src/features/uav/useUavSimulation.ts`
- Modify: `backend/schemas/simulations.py`
- Modify: `backend/services/simulations.py`
- Test: `frontend/src/components/plots/XYPathPlot.test.tsx`
- Test: `tests/test_backend_api.py`

**Interfaces:**
- `XYPathPlot` consumes reference/actual XY paths, waypoint list, obstacle and units.
- `UavPage` requests optional aligned series from the additively extended `UavResponse` API contract.

- [ ] Write failing tests for start/end markers, waypoint/obstacle labels, external legend, and SVG equal-aspect behavior.
- [ ] Verify the test fails before code exists.
- [ ] Write a failing backend API test that requests `include_series` and asserts the aligned, finite response series while existing requests remain valid.
- [ ] Add an opt-in `include_series` request flag and additive response payload sourced from the simulation's existing altitude, tracking-error, clearance, thrust, and torque arrays; do not modify numerical calculations.
- [ ] Implement path projection, automatic geometry margins, reference versus actual styles, obstacle radius and influence circle.
- [ ] Build the controlled UAV scenario and show altitude, tracking error, clearance, and control effort as time-series alongside the XY path.
- [ ] Re-run the focused test file.

### Task 5: Application shell, styling, and validation

**Files:**
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/components/AppShell.tsx`
- Modify: `frontend/src/styles.css`
- Modify: `frontend/README.md`
- Test: `frontend/src/App.test.tsx`

**Interfaces:**
- Adds routes and navigation for `/simulations/uav-obstacle-avoidance` and `/simulations/kalman-filter`.

- [ ] Write failing route/navigation tests.
- [ ] Run the test and verify it fails.
- [ ] Add responsive plot layout, accessible focus/contrast treatment, and documented payload limits without modifying numerical code.
- [ ] Run frontend tests, coverage, and production build.
- [ ] Run `pytest` from the repository root; run the backend and browser-inspect the three pages at desktop and narrow widths.
