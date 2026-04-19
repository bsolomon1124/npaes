# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- License changed from Apache-2.0 to MIT.

## [0.4] - 2026-04-18

### Added

- `pyproject.toml` using the `uv_build` backend with a `src/` layout.
- Development tooling via `ruff` (lint + format) and `ty` (type checking).
- Type hints on the public API (`AES`, `encrypt_raw`, `decrypt_raw`, helpers)
  using `typing.Literal` for constrained key and round sizes.
- `.editorconfig`, `CLAUDE.md`, and a GitHub Actions CI workflow
  (lint, type check, test, build) matrixed across Python 3.10–3.14.
- Coverage reporting via `pytest-cov`.

### Changed

- Bumped supported Python range to `>=3.10, <3.15`.
- Bumped NumPy floor to `>=2.1`.
- Replaced `np.int(0xff)` (removed in NumPy 1.24) with a plain Python `int`.
- Modernized source: dropped Python 2 `__future__` imports and
  `class Foo(object)` style; switched percent-formatting to f-strings.
- License metadata uses the PEP 639 SPDX form (`license = "Apache-2.0"`).

### Removed

- Support for Python 2.7 and Python 3.5–3.9.
- `setup.py`, `requirements.txt`, and `tox.ini`.
- `flake8` and `black` (replaced by `ruff`).
- Author email address from source metadata.

## [0.3] - 2019-01-01

### Added

- Additional functionality and tests.

## [0.2] - 2019-01-01

### Added

- Expanded from raw encryption utility.

## [0.1] - 2019-01-01

### Added

- Initial upload; contains raw encryption utility only.

[Unreleased]: https://github.com/bsolomon1124/npaes/compare/v0.4...HEAD
[0.4]: https://github.com/bsolomon1124/npaes/compare/v0.3...v0.4
[0.3]: https://github.com/bsolomon1124/npaes/compare/v0.2...v0.3
[0.2]: https://github.com/bsolomon1124/npaes/compare/v0.1...v0.2
[0.1]: https://github.com/bsolomon1124/npaes/releases/tag/v0.1
