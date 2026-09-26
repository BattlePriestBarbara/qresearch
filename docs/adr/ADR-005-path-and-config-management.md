# ADR-005 — Path and configuration management (no literal machine path in a tracked file)

* **Status:** accepted
* **Date:** 2026-09-26
* **Tasks:** `INF-01` (environment contract), `INF-02` (gate), `INF-03` (data check), `INF-09` (IO/seed/logging), `INF-10` (typed configuration), `INF-11` (synthetic panel), `INF-12` (CI gate, partially)
* **Clauses concerned:** `PROJECT_SPEC.md` 3.1 (runtime environment), 3.2 (directory structure), 3.3 (data flow), 3.5 (configuration schema), 5.4 (logging and IO), 6.5 (prohibited practices)

## Context

The pre-publish audit (`docs/audits/PRE-PUBLISH-AUDIT.md`, revision `fe67fef`) found **50 blocking
findings, all of them local absolute paths**, and identified three of them as *functional*
hard-codings rather than documentation:

| Location | Nature | Consequence on a different machine |
|---|---|---|
| `src/qresearch/env.py` - `EXPECTED_INTERPRETER`, `DEFAULT_PROVIDER_URI`, `DEFAULT_CACHE_ROOT` | constants | `import qresearch` raises `EnvironmentContractError` and the process terminates, because the import-time verification runs before any user code |
| `.pre-commit-config.yaml` - 9 hook `entry` values | gate configuration | every hook fails with *Executable not found*, so the quality gate cannot run at all |
| `configs/qlib_init.yaml` - `provider_uri`, MLflow `uri` | declarative configuration | the configuration names a data directory and an artifact directory that belong to one developer's disk |

Three facts discovered while fixing this shaped the decision:

1. **Qlib does not read environment variables.** It calls `Path(uri).expanduser().resolve()`
   (`qlib/config.py`, `provider_uri` handling) and nothing else, so a `${VAR}` placeholder in a
   `qlib_init` document is taken *literally* unless something expands it first. Any solution must
   therefore materialize the document before `qrun` sees it.
2. **The contract contradicted its own environment recipe.** `environment.yml` creates a *named* conda
   environment (`qresearch`), whose interpreter is `<conda root>/envs/qresearch/python.exe`, while the
   contract pinned the *base* interpreter. A contract that is only correct on the machine where it was
   written is not a contract.
3. **A generated file is still a tracked file.** `scripts/lock_requirements.py::_header()` wrote
   `sys.executable` into the three lock files, so every regeneration reintroduced the very path the audit
   blocks - the gate could never have stayed green.

At the same time, the strictness `INF-01` exists for must not be traded away: the audit notes that
`tests/test_env_contract.py::test_no_bypass_environment_variable_exists` forbids any "convenience" flag,
and `ADR-001` requires mypy to run inside the pinned environment.

## Decision

Adopt one resolution policy, implemented once in `qresearch.config.paths` (`INF-10`) and mirrored by
`scripts/hook_runner.py`:

| Location | Environment override | Fallback |
|---|---|---|
| read-only market data | `QLIB_DATA_DIR` | `<home>/.qlib/qlib_data/cn_data` (Qlib's own documented default) |
| read-only market data | `QLIB_PROVIDER_URI` (legacy alias of 3.5) | - |
| generated artifacts, caches, MLflow store | `ARTIFACT_ROOT` | `<project_root>/artifacts` |
| pinned interpreter | `QRESEARCH_PYTHON` | `%CONDA_PREFIX%/python[.exe]`, else `sys.executable` |

with these seven rules:

1. **No literal machine path in a tracked file.** No drive letter, UNC path or home shortcut may appear
   in `src/`, `scripts/`, `tests/`, `configs/` or a root configuration file - not even in a comment,
   docstring or error message. `scripts/prepublish_audit.py` enforces it in the pre-push gate, and
   `tests/test_config_paths.py::test_tracked_sources_contain_no_local_absolute_path` enforces it in the
   fast gate with the same patterns.
2. **Paths are inputs, not constants.** `src/qresearch/env.py` keeps its public constants
   (`EXPECTED_INTERPRETER`, `DEFAULT_PROVIDER_URI`, `DEFAULT_CACHE_ROOT`) but computes them from
   `qresearch.config.paths`, so no call site changes while the literal disappears.
3. **The project root is discovered, never declared.** `paths.project_root()` walks up from the
   installed package to `pyproject.toml`; a non-editable install without `ARTIFACT_ROOT` raises
   `ConfigError` instead of writing artifacts into `site-packages`.
4. **Malformed overrides fail loudly.** A value that is empty, relative, a file, or (in strict mode) a
   missing directory raises `qresearch.config.errors.ConfigError` with the offending value and the
   remediation. A *missing* variable falls back to the documented default above; nothing is guessed.
   `ConfigError` is a `RuntimeError`, and `EnvironmentContractError` / `ReadOnlyViolationError` are its
   subclasses, so all configuration defects share one catchable family.
5. **Only paths are overridable, and every override is logged.** `3.5` limits overrides to paths;
   `qresearch.config.loader.expand_placeholders` accepts `${QLIB_DATA_DIR}`, `${QLIB_PROVIDER_URI}` and
   `${ARTIFACT_ROOT:-<fallback>}` and rejects every other name. Honoured overrides are recorded in
   `paths.applied_overrides()`, printed by the CLIs, and embedded in `resolved_config.yaml` under
   `environment_overrides`.
6. **Substituted paths are POSIX-form.** A substitution lands inside a YAML scalar, where a backslash
   starts an escape sequence (a separator before a letter stops the document from parsing at all).
   `as_posix()` is used for every substitution, and Qlib's own `Path(...).resolve()` normalizes it back on
   Windows.
7. **The gate resolves its own interpreter; it never substitutes one.** Every hook calls
   `python scripts/hook_runner.py -m <tool>` (bare `python` only has to reach a stdlib-only bootstrap),
   and the runner re-executes with `QRESEARCH_PYTHON` → `%CONDA_PREFIX%` → the running interpreter. An
   override that does not exist stops the gate with exit code 2 rather than silently running the tool on
   another interpreter.

### Strictness is preserved, not weakened

| `INF-01` guarantee | Status after ADR-005 |
|---|---|
| exact CPython patch version (`3.12.7`) | unchanged, still enforced at import |
| exact pins of every float-critical distribution | unchanged (`REQUIRED_DISTRIBUTIONS` untouched) |
| forbidden interpreter markers (MSYS2, WindowsApps) | unchanged |
| no bypass variable may disable the check | unchanged, re-asserted by its test |
| read-only provider, caches outside it, pollution detection | unchanged |
| interpreter *location* | now declarative - the only thing that changed |

`tests/test_env_contract.py` was extended accordingly: the two negative-control interpreters are now
**assembled from parts** (`Path("C:") / "Python310" / "python.exe"`) instead of being spelled out, so the
test that proves the diagnosis works no longer leaks a machine layout; two new tests pin that the
contract's constants come from the resolver and that the error family is shared.

## Evidence

* **Audit.** `python scripts/prepublish_audit.py` went from `50 blocker(s), 33 warning(s)` (exit 1) to
  `0 blocker(s)` (exit 0); `--selftest` still reports `17 cases`. The three functional hard-codings and
  the three lock-file headers were the last of the 50; the documentation warnings were cleaned in the
  same change, so the only remaining warnings are the two non-path ones (git identity, git housekeeping).
* **The failure mode is gone, verified in three configurations:**
  * nothing set → `qresearch.PROVIDER_URI == <home>/.qlib/qlib_data/cn_data`, import succeeds;
  * `QLIB_DATA_DIR` set → the override is honoured *and* printed
    (`QLIB_DATA_DIR (market data (read-only)) -> ... [environment]`);
  * `QLIB_DATA_DIR=relative/path` → `ConfigError: QLIB_DATA_DIR must be an absolute path, got
    'relative/path'.` with the remediation, exit code 1.
* **Regression risk of the new directory entry point.** The `qrun`-facing documents are now templates:
  `qresearch-config --expand configs/qlib_init.yaml --out <artifacts>/generated/qlib_init.yaml`
  materializes them, which is required because Qlib expands only `~` (fact 1 above).
* **Lock files.** After the fix `_header()` contains no path and the `Interpreter:` line records
  `Path(sys.executable).name`; the three tracked lock files were patched on exactly 7 lines (3+2+2) and
  their package pins are byte-identical.
* **Gate.** `pytest -q -m "not slow"` → 260 passed (was 173). `scripts/prepublish_audit.py` → exit 0.

## Deviations from the plan recorded in the audit report

| Audit report suggested | Adopted instead | Why |
|---|---|---|
| `%QRESEARCH_DATA_ROOT%`, and *error* when unset | `QLIB_DATA_DIR` (primary), `QLIB_PROVIDER_URI` (alias), fallback to `<home>/.qlib/qlib_data/cn_data` | The instruction for this change names `QLIB_DATA_DIR`; the fallback is Qlib's own documented convention, not a guess, and keeping `QLIB_PROVIDER_URI` honours the alias `3.5` already names. A variable that must be set would break a fresh clone before the data is downloaded |
| `configs/environment.yaml` | the same table in code (`paths.py`) plus `${...}` placeholders in the YAML templates | one source of truth; a YAML mirror of code can drift, and the loader had to expand placeholders anyway |
| "schema" validated by a framework | `dataclasses`, as `3.5` literally requires | `pydantic` is not in the frozen closure of `3.1`/`ADR-003`; adding it would be the dependency drift the lock files exist to prevent |
| hook wrapper named `scripts/hook_runner.py` | same name | no reason to deviate |

## Consequences

* A new machine needs exactly one thing to become productive: `QLIB_DATA_DIR` pointing at the store (or
  the Qlib default location). Nothing else is machine-specific, and the effective value is always logged.
* `configs/*.yaml` are **templates**. A `qrun` workflow MUST materialize them first
  (`qresearch-config --expand ...`); `qresearch.env.build_qlib_init_kwargs()` remains the canonical
  programmatic path and needs no materialization.
* The interpreter pin is now "the activated environment", so a study MUST activate the `qresearch`
  environment (or set `QRESEARCH_PYTHON`). `check_env.py` and every CLI verify this at import and abort
  with a remediation message.
* `docs/` MAY describe machine layout (the audit treats documentation as a warning, not a blocker); the
  strings used here are deliberately machine-independent anyway.
* **Not covered:** the *git history* still contains the earlier revisions of these files, including the
  paths they carried, so pushing `master` publishes them. Removing them requires rewriting history
  (`git filter-repo`) or publishing from a single clean commit - a destructive decision that is
  explicitly out of scope here and remains open (recorded in the audit report, section 4.2).
* Adding a path variable is now a two-line change (`paths.py` plus the loader's allow-list) and an ADR
  entry; adding a *non-path* environment override remains prohibited by `3.5`.

## Impact on previously reported results

None. No numerical code path changed: the distribution pins, the formula implementations, the
`ICMoments` / `NeweyWestResult` outputs and the data layer are untouched, and on the machine where prior
results were produced the resolved provider, cache root and interpreter are identical to the previous
constants. The change is therefore an amendment that bumps the specification's **minor** version
(v1.0.0 → v1.1.0: new normative obligations around paths and configuration), not its major version, and
no study is marked `SUPERSEDED`.
