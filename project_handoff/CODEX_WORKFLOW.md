# Codex Workflow

## Start Git Bash

Open Git Bash, then:

```bash
cd /d/engineering-ode-simulator
```

## Activate Virtual Environment

```bash
source .venv/Scripts/activate
```

If dependencies need refreshing:

```bash
python -m pip install -e ".[dev]"
```

## Run Tests

Preferred full verification:

```bash
python -m pytest -v --basetemp=".pytest_tmp_$(date +%s)" -p no:cacheprovider
```

Basic command:

```bash
python -m pytest
```

## Run Streamlit

```bash
streamlit run streamlit_app.py
```

Default local URL is usually:

```text
http://localhost:8501
```

## Cleanup Commands

Use these only for generated local artifacts:

```bash
find . -maxdepth 1 -type d -name ".pytest_tmp_*" -print
find . -maxdepth 1 -type d -name ".pytest_tmp_*" -exec rm -rf {} +
```

On PowerShell:

```powershell
Get-ChildItem -Directory -Filter ".pytest_tmp_*"
Remove-Item -Recurse -Force .pytest_tmp_*
```

Do not delete source, docs, examples, screenshots, or outputs unless explicitly asked.

## Commit Workflow

Do not commit automatically.

When the user asks for a commit:

```bash
git status --short
git diff
python -m pytest -v --basetemp=".pytest_tmp_$(date +%s)" -p no:cacheprovider
git add <specific files>
git commit -m "docs: add project handoff pack"
```

Use conventional commit messages such as `docs: ...`, `fix: ...`, `test: ...`, `refactor: ...`.

## One-Codex-Chat-Per-Domain Rule

For large follow-up work, keep one chat per domain:

- documentation and handoff
- Streamlit UI
- scientific validation/tests
- plot styling
- numerical solver changes
- packaging/release

This avoids mixing context and reduces accidental cross-domain edits.

## Recommended Codex Skills By Task

- `strategic-compact`: use at phase boundaries and before moving context into a new chat.
- `codebase-onboarding`: use when a new chat needs to quickly understand repo structure.
- `code-tour`: use only when creating `.tour` walkthrough artifacts, not ordinary Markdown docs.
- `python-patterns`: use for Python refactors, module design, and code review.
- `developing-with-streamlit`: use for all Streamlit edits, debugging, styling, layout, and app-running tasks.
- `scientific-visualization`: use for plot polish, publication-style graph cleanup, color choices, and export quality.
- `python-testing`: use for pytest strategy, validation tests, fixtures, and regression coverage.
- `delivery-gate`: use as a final quality reminder: verify, avoid rationalizing skipped checks, and keep session outputs documented.
- `ruff`: use for Python lint/format checks when requested.
- `ty`: use for Python type-checking if introduced.
- `uv`: use if the project later moves to uv-based environment/dependency workflows.
