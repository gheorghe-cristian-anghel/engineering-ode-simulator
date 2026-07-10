# Roadmap Professionalization

This roadmap keeps the project framed as an educational/prototype/portfolio toolkit moving toward professional quality.

## 1. Fix Current Bugs

- Re-run the full pytest suite before and after changes.
- Regression-test Streamlit after plot helper or layout edits.
- Fix only confirmed issues; avoid speculative solver rewrites.
- Keep numerical behavior unchanged unless a bug is reproduced and tested.

## 2. Scientific Robustness

- Audit assumptions in each model and example.
- Add units and parameter range notes where helpful.
- Keep simplified-model disclaimers visible.
- Check stability constraints for explicit PDE methods.
- Verify controller examples do not imply production/safety-critical use.

## 3. Validation Tests

- Expand analytical comparisons where closed-form references exist.
- Add convergence checks for numerical methods when appropriate.
- Test failure modes for invalid inputs.
- Keep tests behavior-focused and deterministic.
- Avoid brittle visual snapshot tests unless there is a strong reason.

## 4. UI / UX

- Preserve the current Streamlit app structure.
- Improve readability, grouping, default parameters, and explanatory captions.
- Keep controls ergonomic and prevent expensive default simulations.
- Avoid redesigns unless explicitly requested.

## 5. Graph Polish

- Standardize figure sizes, labels, legends, captions, and colorbars.
- Use colorblind-aware palettes and perceptually reasonable colormaps.
- Avoid misleading scales or decorative styling.
- Keep plots understandable in screenshots and Streamlit.

## 6. Animations

- Keep animations optional and lightweight.
- Use existing model outputs and visualization helpers.
- Store only curated portfolio assets.
- Avoid making animation generation part of normal tests unless fast and stable.

## 7. Exports

- Continue CSV export support for selected simulations.
- Add metadata only where it helps reproducibility.
- Keep output paths predictable under `outputs/`.

## 8. Reports

- Consider generated Markdown or notebook-style reports for selected demos.
- Include assumptions, parameter values, plots, and key metrics.
- Avoid claiming certification, high-fidelity validation, or production readiness.

## 9. Final QA

- Run full pytest.
- Smoke-test Streamlit.
- Run representative examples.
- Check README screenshots and docs links.
- Review claims for accuracy and educational scope.

## 10. Future Advanced Features

- 2D truss FEM solver.
- Beam bending FEM.
- More curated Streamlit PDE demos.
- Packaging and release polish.
- Optional docs site.
- More formal benchmark/validation references for selected models.
