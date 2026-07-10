---
title: Codex Workflow
tags:
  - engineering-simulation-toolkit
  - codex
  - workflow
---

# Codex Workflow

## Bash Commands

Use Git Bash when possible:

```bash
cd /d/engineering-ode-simulator
source .venv/Scripts/activate
python -m pip install -e ".[dev]"
```

## Testing Command

```bash
python -m pytest -v --basetemp=".pytest_tmp_$(date +%s)" -p no:cacheprovider
```

Use this after behavior changes and before claiming tests pass.

## Streamlit Command

```bash
streamlit run streamlit_app.py
```

Default local URL is usually `http://localhost:8501`.

## Commit Workflow

Do not commit automatically.

When the user asks for a commit:

```bash
git status --short
git diff
python -m pytest -v --basetemp=".pytest_tmp_$(date +%s)" -p no:cacheprovider
git add <specific files>
git commit -m "docs: add project notes"
```

## One Chat Per Domain Rule

Use separate Codex chats for:

- documentation and handoff
- Streamlit UI
- scientific validation/tests
- plot styling
- numerical solver changes
- packaging/release

This reduces accidental cross-domain edits.

## Installed Skills Usage

- `codebase-onboarding`: repo maps and new-chat context.
- `code-tour`: only for `.tour` walkthrough artifacts, not ordinary Markdown notes.
- `strategic-compact`: phase handoffs and context resets.
- `python-patterns`: Python readability and refactoring.
- `developing-with-streamlit`: all Streamlit work.
- `scientific-visualization`: plot polish, color, labels, export quality.
- `python-testing`: pytest and validation work.
- `ruff`: lint/format checks when requested.
- `ty`: type checking if introduced.
- `uv`: future environment workflow if adopted.

## Hard Boundaries

- Do not modify solvers unless explicitly requested.
- Do not redesign Streamlit unless asked.
- Do not add features during docs tasks.
- Do not commit unless asked.

Related: [[New Chat Starter]], [[Known Bugs and Fixes]].
