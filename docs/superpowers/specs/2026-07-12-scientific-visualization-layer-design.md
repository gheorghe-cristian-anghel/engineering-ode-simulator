# Scientific Visualization Layer Design

## Goal

Replace the React frontend's single static heat canvas with reusable, browser-native scientific visualizations for heat fields, engineering time series, UAV paths, and state estimation, without changing numerical equations or breaking existing API contracts.

## Technology choice

Use React with native Canvas and SVG. Canvas is used for raster scalar fields, where a 256 by 256 grid remains inexpensive and direct pixel mapping guarantees control of matrix orientation. SVG is used for line and geometric plots, where axes, accessible labels, hover inspection, and equal coordinate scales are clearer than a general-purpose chart dependency. No visualization dependency is added.

## Components and data flow

- `PlotPanel` provides a titled, labelled responsive container and a consistent empty-data treatment.
- `HeatmapPlot` consumes x/y coordinates and a `field[y][x]` matrix. It validates rectangular finite data, maps the first matrix row to the minimum y coordinate, renders y upward on the screen, and supports automatic or fixed value domains. It provides labelled axes, a colorbar, and a pointer probe.
- `TimeSeriesPlot` consumes focused series definitions. It renders finite samples only, uses reference lines with a dashed convention, computes domains with padding, and limits rendered vertices for dense series without modifying input arrays.
- `XYPathPlot` consumes x/y projections of path data. It calculates an equal-scale view box with margins, renders reference/actual paths, waypoints, start/end markers, and a circular obstacle plus influence radius. Its legend is outside the plotting area.
- `StateEstimationView` composes time-series plots for true state, noisy measurements, estimates, and estimation error. It accepts covariance-derived bands only as an optional extension; the current API has no covariance.
- API modules and feature hooks remain focused on their endpoint contracts. New UAV and Kalman pages consume the existing endpoints unchanged.

## Scientific conventions

- Heat fields are treated as `field[yIndex][xIndex]`; no transpose occurs. Canvas rows are vertically inverted so increasing y is displayed upward.
- Coordinate, time, temperature, position, current, speed, and error units come from response metadata.
- Heatmap configuration supports explicit symmetric domains for future signed wave fields.
- XY geometry has `preserveAspectRatio="xMidYMid meet"`, based on a shared world-unit scale rather than screen aspect ratio.
- Non-finite samples and malformed rows are ignored safely; plots explain when no usable data remains.

## Performance and payload limits

The heat API already caps fields at 65,536 cells; this remains the practical client-side limit for one canvas render. The UAV and Kalman APIs cap their time samples at 1,001 and 2,001 respectively. Transformations are memoized, canvas pixel buffers are generated without duplicating grid-sized structures, and line paths are bounded through index-based decimation when samples exceed the display budget. Animation and client-side storage of snapshots are out of scope.

## Validation

Add focused component/helper tests for orientation, domain selection, finite-value handling, decimation, equal-scale path bounds, and scientific labels. Run frontend tests/build, existing Python tests, and inspect the responsive frontend in a browser with the API running. No commit is created automatically.
