# ADR-002 — Third-party stub packages are rejected in favour of per-module mypy overrides

* **Status:** accepted
* **Date:** 2026-09-15
* **Task:** `INF-01`
* **Clause concerned:** `PROJECT_SPEC.md` 3.1 (float-critical exact pins), 5.2 (strict typing)

## Context

`pandas`, `scipy`, `statsmodels`, `scikit-learn` and `pyarrow` are installed without a PEP 561
marker, so mypy reports `import-untyped`. The standard remedy is to install stubs
(`pandas-stubs`, `scipy-stubs`).

While implementing `INF-01` this was attempted. Pip resolved the stubs against a newer
dependency graph and **upgraded `numpy` from the pinned `1.26.4` to `2.5.3`**, leaving `scipy`
1.13.1 in a broken combination (`scipy requires numpy<2.3`). The drift compiled the project but
silently changed the numerical environment.

## Decision

Do not install third-party stub packages. Use per-module mypy overrides instead:

```ini
[mypy-pandas.*]      ignore_missing_imports = True
[mypy-scipy.*]       ignore_missing_imports = True
[mypy-statsmodels.*] ignore_missing_imports = True
[mypy-sklearn.*]     ignore_missing_imports = True
[mypy-pyarrow.*]     ignore_missing_imports = True
```

`numpy`, `torch`, `cvxpy`, `lightgbm`, `matplotlib` and `mlflow` do ship markers and stay fully
typed. The global `--ignore-missing-imports` remains OFF, so `qresearch` itself stays under
`strict = True`.

## Rationale

* A dependency that can upgrade a float-critical pin is a reproducibility hazard, exactly the
  class of defect `PROJECT_SPEC.md` 3.1 exists to prevent, and it was observed rather than
  hypothesised.
* Blast radius is bounded: only third-party symbols become `Any`; every function we author is
  still fully annotated and checked.
* The alternative (pinning stub versions with `--no-deps`) does not remove the hazard: the next
  stub release re-introduces it, and stub/runtime version skew then produces *false* type errors
  inside statistical code.

## Consequences

* Type errors arising from a change in a `pandas`/`scipy` API are not caught statically; they
  are caught by the unit-test suite, which is the compensating control.
* `numpy` was restored to `1.26.4` and verified; the contract check (`scripts/check_env.py`)
  now passes and would have failed loudly had the drift been left in place.
* `tests/test_dependency_pinning.py::test_no_stub_package_can_drift_the_float_core` fails if the
  stubs are ever silently re-added, and requires the rejection to stay documented.
