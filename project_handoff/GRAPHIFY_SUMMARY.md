# Graphify Analysis Summary

## Run status

Graphify completed successfully on 2026-07-12 using code-only structural extraction. It analyzed 181 code files (83 non-code files were intentionally skipped), producing a graph with 3,292 nodes, 6,157 relationships, and 170 communities. The post-build integrity diagnostic found no missing or dangling endpoints, self-loops, duplicate edges, or same-endpoint edge collapse.

## Major relationships

- `streamlit_app.py` is the UI integration hub. It imports 23 internal modules, spanning numerical-analysis helpers, simulation models, and visualization utilities.
- `backend/main.py` routes into `backend/api/routes.py`; routes depend on request/response schemas and `backend/services/simulations.py`; the service layer invokes analysis/model modules and serializes results through `backend/core/serialization.py`.
- `visualization/plot_style.py` is a shared presentation dependency, imported by `visualization/__init__.py`, Streamlit, examples, and tests.
- Analysis/control modules form an orchestration layer over model simulations; tests and examples are the main consumers of individual model and analysis modules.

## Highly coupled modules

- `streamlit_app.py`: 23 internal module dependencies; the highest-level UI coupling point.
- `visualization/plot_style.py`: 21 incoming module relationships; shared plotting policy.
- `models/discrete_pid.py`: 9 incoming and 1 outgoing module relationships.
- `backend/services/simulations.py`: 1 incoming and 7 outgoing module relationships; primary backend orchestration point.
- Other frequent consumers include `models/dc_motor.py`, `analysis/state_space.py`, and the quadcopter trajectory modules.

## Streamlit dependencies

`streamlit_app.py` depends on finite-difference/FEM utilities; Kalman, particle, and unscented filters; quadcopter control and trajectory workflows; circuit, heat, and wave models; particle-filter/UKF example helpers; and airfoil/fluid/plot-style visualization modules. Its broad reach makes it the clearest candidate for feature-level UI composition.

## Backend dependencies

The backend follows a clear entrypoint-to-route-to-service path. `backend/services/simulations.py` depends on schemas and serialization plus state-space, Kalman, quadcopter obstacle/waypoint, and 2D heat-equation functionality. `backend/api/routes.py` depends on both the schemas and service layer.

## Refactor opportunities

- Split `streamlit_app.py` into feature-oriented UI panels/controllers so each simulation family owns a smaller import surface.
- Keep `backend/services/simulations.py` as an orchestration layer; move family-specific dispatch into focused service modules if it continues to grow.
- Treat `visualization/plot_style.py` as a stable shared boundary and avoid adding simulation-specific logic there.
- Where examples provide reusable simulation helpers to Streamlit, promote those helpers into an analysis/model module rather than importing from `examples/`.

## Circular-dependency review

No circular internal-import groups were found in the directed import projection. The exported Graphify graph is undirected, so this result is based specifically on `imports` and `imports_from` edges, not a general call-graph cycle claim.

## Generated output

All generated artifacts are under the Git-ignored `graphify-out/` directory:

- `graphify-out/graph.json` — dependency/module relationship graph
- `graphify-out/manifest.json` — scan manifest
- `graphify-out/.graphify_analysis.json` — analysis metadata
- `graphify-out/.graphify_labels.json` — community labels
- `graphify-out/GRAPH_REPORT.md` — human-readable Graphify report
- `graphify-out/cache/ast/` — AST extraction cache

Graphify commands used:

```powershell
graphify extract . --code-only --out .
graphify cluster-only . --no-viz --no-label
```
