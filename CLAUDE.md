# CLAUDE.md

Project-specific rules for Claude Code. Keep this terse — if a rule isn't
load-bearing, it doesn't belong here.

## What this project is

Pure-NumPy reference implementation of AES (FIPS 197). Pet project,
**not production crypto** — no IV, no block mode support. The README's
"Caution" section is load-bearing; don't describe this as production-ready.

## Commands

Always go through `task` (Taskfile.yml) or `uv` — never `python` or `pip`
directly.

- `task check` — lint + typecheck + test (use before declaring "done")
- `task test` — pytest with coverage gate (fails under 75%)
- `task format` — apply ruff fixes + formatting
- `task build` — sdist + wheel via `uv build`
- `uv run <cmd>` — run a tool inside the project venv

## Non-obvious rules

- Mutable default args and `arange(...)` in parameter defaults are
  **intentional** per-function caches — `B006`/`B008` are ignored globally.
  Do not "fix" them.
- `N802`/`N803`/`N806` are ignored because variable/function names follow
  FIPS 197 (`Nk`, `Nr`, `SubBytes`, ...). Match the spec's casing.
- Array hot paths pass `out=` buffers and use `functools.partial` to
  pre-bind polynomial kwargs. Don't refactor these; benchmark first.
- PEP 695 `type` statements aren't used — `requires-python` is `>=3.10`
  and PEP 695 needs 3.12+. Use `TypeAlias` with `from __future__ import
  annotations` instead.
- Tests must exercise the FIPS 197 Appendix A/B/C vectors in
  `tests/test_npaes.py`. If you touch `encrypt_raw`/`decrypt_raw` or any
  transform (`sub_bytes`, `shift_rows`, `mix_columns`, ...), run the full
  test suite before reporting success.

## Never do

- Reintroduce Python 2 / 3.5–3.9 compat shims (`__future__` imports beyond
  `annotations`, `class Foo(object)`, `%` formatting).
- Swap `uv_build` for `hatchling` / `setuptools` / `poetry`.
- Re-add the author's email anywhere in source or metadata.
- Ship code that claims or implies production-grade cryptography.

## Release flow

Tags `v*` trigger `.github/workflows/publish.yml`:
build → PyPI (OIDC, gated by `pypi` environment with required reviewers)
→ Sigstore sign → GitHub Release with SBOM.
Bump `project.version` in `pyproject.toml` and `__version__` in
`src/npaes/__init__.py` in the same commit as the tag. A CI step
fails the release if they disagree.
