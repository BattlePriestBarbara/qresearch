# ADR-001 — mypy runs inside the pinned environment (pre-commit `language: system`)

* **Status:** accepted
* **Date:** 2026-09-15
* **Task:** `INF-02`
* **Clause concerned:** `PROJECT_SPEC.md` 3.8 / 5.5 quality gates (mypy is a merge blocker)

## Context

The mandatory gate requires strict `mypy` over `src/qresearch`. The project's own code imports
`qlib`, `torch`, `pandas` and `numpy`. `qlib` ships no PEP 561 marker and no stubs, so mypy can
only resolve those imports when it can see the installed distribution.

Two implementations were possible:

1. A standard remote hook (`repo: https://github.com/pre-commit/mirrors-mypy`) which runs in an
   isolated hook virtualenv.
2. A `language: system` hook that runs the pinned interpreter - originally a literal path, now
   resolved at run time by `scripts/hook_runner.py` (see `ADR-005`) - with the tools pinned in
   `requirements-dev.txt`.

## Decision

Adopt option 2 for `mypy` (and for the other hooks, for consistency of tool versions).

## Rationale

* An isolated environment cannot import `qlib`/`torch`; the only ways to make gate 1 pass would
  be `--ignore-missing-imports` (globally weakening the one control that protects against tensor
  and Index shape errors) or `additional_dependencies` on the full runtime stack (re-downloading
  torch into a second environment, with a version that floats unless pinned twice).
* `requirements-dev.txt` already pins every tool at a full patch version, so the gate result is
  reproducible; the floating remote `rev:` of hook 1 would not be.
* This machine's default `python` on `PATH` is an MSYS2 build without the quant stack; pinning
  the interpreter path removes that ambiguity entirely.

## Consequences

* `pre-commit` must be invoked with the pinned environment available; the hooks contain the
  absolute interpreter path, so this is satisfied by construction.
* `pre-commit autoupdate` must NOT be used for the local hooks: versions change by editing
  `requirements-dev.txt`.
* `tests/test_repo_structure.py::test_precommit_gate_declares_required_hooks` asserts this
  configuration, so a future switch back to isolated hooks fails the suite and forces a new ADR.
