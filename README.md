# qresearch — Alpha Research Platform on Qlib

A quantitative research project for **temporal-neural Alpha factor mining under strict
statistical inference**, built on Microsoft Qlib.

> **Governing document:** [`PROJECT_SPEC.md`](PROJECT_SPEC.md) is normative. It defines the
> formalism (Rank IC, Newey–West HAC inference, bootstrap monotonicity tests, Student-*t*
> negative log-likelihood, cost-aware mean–variance optimization), the interface contracts and
> the task graph. Code that contradicts it is defective, regardless of its numerical output.

| Field | Value |
|---|---|
| Project root | `D:\Qlib` |
| Interpreter | `D:\Anaconda3\python.exe` (CPython **3.12.7** — enforced at runtime) |
| Framework | `pyqlib==0.9.7` |
| Market data (read-only) | `D:\qlib_data\cn_data` |
| Spec version | `v1.0.0` |

---

## 1. Verified environment (task `INF-01`)

The runtime contract is **enforced in code**, not documented and hoped for:

```text
qresearch.env.verify_environment()  ->  runs automatically when `qresearch` is imported
```

* CPython must be exactly **3.12.7** on `D:\Anaconda3\python.exe`.
* Every float-critical distribution must match its full patch version.
* Any mismatch raises `EnvironmentContractError` (a `RuntimeError`) and **terminates the
  process**. There is deliberately **no** bypass flag: a silent downgrade is the exact failure
  mode this check prevents.

Why so strict: a patch drift in `numpy`/`scipy`/`torch` changes the low-order bits of the IC
series. That propagates into the Newey–West long-run variance and can move a `p`-value across a
significance threshold, silently invalidating a research conclusion.

### Dependency artifacts

| File | Role |
|---|---|
| `requirements.txt` | Direct runtime deps, **full patch pins only** (no ranges, no `~=`) |
| `requirements-dev.txt` | Formatter/linter/typing/test tools, same pin policy |
| `environment.yml` | **Hybrid env spec** (ADR-003): `python=3.12.7` + the conda-managed native extensions + a `pip:` block referencing the pip lock |
| `requirements.lock.txt` | **Exact transitive closure** (210 distributions) of the verified environment, offline, with per-package provenance |
| `requirements.lock.pip.txt` | The same closure **minus the conda-managed** distributions (207) — the file `environment.yml` installs |
| `requirements.lock.hashes.txt` | **sha256 digest lock** (207 distributions) usable with `pip install --require-hashes` |

### Hybrid environment strategy — 底层 C/C++ 扩展由 Conda 托管，纯 Python 科学计算栈由 Pip Hash 锁定

> *Low-level C/C++ extensions are managed by conda; the pure-Python scientific stack is
> hash-locked by pip.* — see [ADR-003](docs/adr/ADR-003-hybrid-conda-pip-environment.md)

A pure-pip closure is **impossible** for CPython 3.12 here, because some native extensions were
released before 3.12 and ship no compatible wheel. Compiling them with pip would drag an unpinned
MSVC toolchain and unpinned build dependencies into the environment — the exact non-determinism
`PROJECT_SPEC.md` 3.1 forbids. The split is therefore:

| Category | Rule (derived by `scripts/lock_requirements.py`) | Managed by | Current members |
|---|---|---|---|
| `wheel` | a wheel tagged `cp312`, `py3*`, or stable-ABI `cp3X-abi3` (`X ≤ 12`) exists | **pip**, digest-pinned | 206 |
| `sdist_pure_python` | the release ships **no wheel at all** → pure Python, no compiler needed | **pip**, sdist digest pinned | `gym==0.26.2` |
| `conda` | wheels exist but none compatible with this interpreter | **conda**, pinned in `environment.yml` incl. build string | `psutil==5.9.0`, `pywin32==305` (reports 305.1), `pywinpty==2.0.10` |

`gym` needs no compiler (pure Python) and conda-forge only reaches `0.26.1`, so delegating it to
conda would silently downgrade a dependency of the verified environment; it stays pip-managed with
its sdist digest in the lock.

```powershell
# reproduce the environment (two managers, in this order)
D:\Anaconda3\Scripts\conda.exe env create -f environment.yml   # python + 3 native extensions
conda activate qresearch
D:\Anaconda3\python.exe -m pip install --require-hashes -r requirements.lock.hashes.txt
D:\Anaconda3\python.exe -m pip install -e . --no-deps          # editable, src-layout

# verify the contract (exit 0 required)
D:\Anaconda3\python.exe scripts\check_env.py

# regenerate every lock artefact (offline closure; --hashes adds the PyPI classification)
D:\Anaconda3\python.exe scripts\lock_requirements.py --hashes
```

**Float-critical policy.** `numpy`, `scipy`, `pandas` and `torch` are installed from **PyPI
wheels**, never from a conda channel: conda and PyPI builds differ in BLAS/LAPACK linkage and
therefore in the last bits of every computed moment.

> **Fidelity nuance (recorded deliberately).** In the currently verified environment, `pandas`,
> `statsmodels` and `scikit-learn` happen to come from conda builds. The hybrid specification
> reproduces them from hash-verified PyPI wheels at **identical versions** (asserted at runtime by
> `scripts/check_env.py`). Adding them to the conda section is a one-line change if byte-level
> fidelity to that specific environment is required.

> **Field note (observed and fixed).** Installing `pandas-stubs` / `scipy-stubs` silently
> upgraded `numpy` to `2.5.3`, breaking the pin. The stubs were removed, `numpy` restored to
> `1.26.4`, and the rejection is now (a) documented in `requirements-dev.txt`, (b) encoded as a
> mypy per-module override instead, and (c) pinned by a regression test
> (`tests/test_dependency_pinning.py`). See `docs/adr/ADR-002`.

> **Field note (observed and fixed).** The base environment shipped `pylint 2.16.2`, which
> predates Python 3.12 support and aborted with `astroid F0002` on this codebase. Pylint is
> deliberately upgraded and pinned to `3.3.1` in `requirements-dev.txt`; the unrelated
> base-environment package `spyder 5.5.1` then reports a `pylint<3.1` conflict, which is
> recorded in that file rather than silently accepted.

### Read-only data source

`D:\qlib_data\cn_data` is **read-only**. The project enforces this three ways:

1. **Cache redirection** — `qresearch.env.build_qlib_init_kwargs()` is the only sanctioned way
   to build `qlib.init` arguments. It returns `expression_cache=None`, `dataset_cache=None` by
   default and routes every writable location (opt-in caches, the MLflow tracking store) to
   `D:\Qlib\artifacts\`, which is git-ignored.
2. **Path scan** — the generated kwargs are walked recursively; any path pointing inside the
   provider raises `ReadOnlyViolationError`.
3. **Write detection** — `ProviderWriteGuard` fingerprints the provider tree (file count,
   directory count, total bytes, newest mtime) and raises if a single byte changed:

```python
import qlib, qresearch
from qresearch.env import ProviderWriteGuard, build_qlib_init_kwargs

with ProviderWriteGuard(qresearch.PROVIDER_URI):
    qlib.init(**build_qlib_init_kwargs())
    # ... any research step: a write into the provider is detected and aborts the run
```

Optional OS-level hardening (manual, elevated shell) is documented in
[`configs/qlib_init.yaml`](configs/qlib_init.yaml):

```powershell
icacls "D:\qlib_data\cn_data" /deny "%USERNAME%":(W,D,DC)
```

---

## 2. Repository layout (task `INF-02`)

**src-layout is mandatory**: packages live under `src/qresearch/`, so `import qresearch` in tests
and scripts resolves through the *installed* distribution. A flat top-level package directory
does not exist — it is asserted by `tests/test_repo_structure.py`. This prevents the classic
failure where local tests import un-packaged source and hide a missing dependency.

```text
D:\Qlib\
├── PROJECT_SPEC.md            # normative specification
├── README.md                  # this file
├── pyproject.toml             # packaging (src-layout) + black/isort/ruff/pytest config
├── .mypy.ini                  # strict typing; per-module overrides for stub-less libraries
├── .pylintrc                  # slow semantic gate
├── .pre-commit-config.yaml    # the mandatory gate
├── .gitignore                 # data / weights / trackers / notebook outputs excluded
├── environment.yml · requirements.txt · requirements-dev.txt · requirements.lock.txt
├── configs/qlib_init.yaml     # canonical qlib.init values (mirrors build_qlib_init_kwargs)
├── src/qresearch/             # the package (6 architecture layers)
│   ├── env.py                 # INF-01: contract enforcement + read-only provider guard
│   ├── version.py · py.typed
│   ├── config/   data/   features/   stats/   models/   portfolio/   evaluation/   utils/
├── scripts/check_env.py       # INF-01 CLI
├── scripts/lock_requirements.py
├── tests/                     # env contract, pinning, read-only guard, repo structure
├── artifacts/                 # git-ignored: mlruns, caches, reports, figures
└── data/                      # git-ignored accessor directory
```

## 3. Quality gates

| Gate | Scope | Command |
|---|---|---|
| **Fast** (every commit) | hygiene, `black`, `isort`, `ruff`, `flake8`, `mypy`, `nbstripout` | `D:\Anaconda3\Scripts\pre-commit.exe install` then normal commits |
| **Slow** (CI / before reporting) | `pylint --fail-under=9.0`, `pytest -q -m "not slow"` | `pre-commit run --all-files --hook-stage manual` |
| Tests | unit + statistical validation | `D:\Anaconda3\python.exe -m pytest -q` |
| Types | strict, tensor/Index shape safety | `D:\Anaconda3\python.exe -m mypy --config-file .mypy.ini` |

`mypy` runs as a `language: system` hook on purpose: it needs the **pinned** environment to see
`qlib`/`torch`/`pandas`, so it cannot live in an isolated hook virtualenv without being weakened
to `ignore-missing-imports`. Hook versions are pinned in `requirements-dev.txt`, so gate results
are reproducible instead of floating with a remote hook revision.

**Notebook policy.** `nbstripout` runs on every `*.ipynb` change, so notebook outputs and
execution counts never reach a commit. `notebooks/` is for exploration only; any number that is
reported must be regenerated through a `scripts/` entry point (`PROJECT_SPEC.md` 3.2, 6.1).

**`.gitignore` policy.** Excluded: `__pycache__` and tool caches, `.qlib/` and any
`qlib_cache/`/`expression_cache/`/`dataset_cache/`, model weights (`*.pth`, `*.pt`, `*.ckpt`,
`*.pkl`, `*.safetensors`, …), experiment state (`mlruns/`, `wandb/`, `tensorboard/`,
`lightning_logs/`), notebook checkpoints, secrets and all of `artifacts/**`.

## 4. Delivered in this increment

| Task | Artefact | Evidence |
|---|---|---|
| `INF-01` exact locking | `requirements.txt`, `requirements-dev.txt`, `environment.yml`, `requirements.lock*.txt`, `scripts/lock_requirements.py` | pin-policy tests; hybrid-split consistency tests; lock format verified against `pip --require-hashes` |
| `INF-01` hybrid environment (`ADR-003`) | conda section pins the 3 wheel-less native extensions with build strings; `pip:` block installs `requirements.lock.pip.txt` | `tests/test_dependency_pinning.py::test_environment_yml_conda_set_matches_the_delegated_set` |
| `INF-01` environment assertion | `src/qresearch/env.py`, import-time hook in `src/qresearch/__init__.py`, `scripts/check_env.py`, console script `qresearch-check-env` | `tests/test_env_contract.py` (raises `RuntimeError`, aggregates all violations, no bypass variable) |
| `INF-01` read-only isolation | `fingerprint_tree`, `ProviderWriteGuard`, `assert_provider_isolation`, `build_qlib_init_kwargs`, `configs/qlib_init.yaml` | `tests/test_provider_readonly.py` |
| `INF-02` src-layout | `pyproject.toml`, `src/qresearch/**`, `py.typed` | `tests/test_repo_structure.py` |
| `INF-02` pre-commit gate | `.pre-commit-config.yaml`, `.mypy.ini`, `.pylintrc`, `pyproject.toml` tool config | `pre-commit run --all-files` |
| `INF-02` ignore policy | `.gitignore` | pattern assertions in `tests/test_repo_structure.py` |

## 5. Specification provenance

No formula, interface or naming rule may be changed in code without amending
`PROJECT_SPEC.md` and recording an Architecture Decision Record in `docs/adr/`. Deviations that
were already necessary are recorded in the relevant config comments:

* **ADR-worthy note 1** — `mypy` runs in-environment rather than in an isolated hook venv.
* **ADR-worthy note 2** — third-party stub packages are rejected in favour of per-module mypy
  overrides, because they caused a float-critical `numpy` upgrade.
* **ADR-003** — the hybrid conda/pip environment split.
* **ADR-004** — the label purge radius uses the inclusive price window (one extra period relative
  to the literal formula in `PROJECT_SPEC.md` 2.2.5, i.e. the conservative direction).
* **ADR-worthy note 3** (`ST-02`) — `PROJECT_SPEC.md` 3.4.6 declares
  `newey_west_se(x, bandwidth=None, prewhite=False, small_sample: TestType)`, but `TestType` is
  defined nowhere in the specification. `small_sample` is bound to the boolean `sqrt(T/(T - k_reg))`
  correction factor of 2.2.2 option 2, because option 1 (the Student-`t` reference) changes the
  *p*-value and not the standard error this function returns; option 1 stays reachable through
  `newey_west(..., reference="student_t")`.

## 6. Data snapshot — what the store actually contains (`INF-03`)

`scripts/data_check.py` reports the inventory rather than assuming it:

```text
provider   : D:\qlib_data\cn_data (read-only)
calendar   : 1999-11-10 .. 2020-09-25 (4943 trading days)
universe   : csi300 -> 690 instruments (membership intervals, not a point-in-time count)
sample     : SH600000 cols=['$open', '$close', '$volume', '$factor'], rows=424, missing_rate=0.0047
```

Three consequences are recorded here so no study walks into them:

1. **The snapshot ends on 2020-09-25.** Every example date range in `PROJECT_SPEC.md` (2008–2024,
   test 2019–2024) is therefore illustrative only; `data_check.py` exits `5` when a requested window
   reaches beyond the store, because that is a data-domain failure rather than a code failure.
2. **The store carries no fundamental fields** — only price/volume/factor. The announcement-lag
   machinery of `INF-04` and its stress test are therefore exercised on synthetic fundamentals
   (`tests/test_pit_announcement_lag.py`), and wiring a real PIT fundamentals source is a
   data-acquisition task, not a code task.
3. **`csi300` membership intervals are not a point-in-time count.** `D.list_instruments` returns the
   union of all memberships (690 here); the tradable set on a given date comes from
   `qresearch.data.universe.membership_mask`, which reads the recorded intervals.

## 7. Phase 1 — data pipeline and leakage audit (`INF-04` … `INF-08`)

| Task | Artefact | Invariant proven by tests |
|---|---|---|
| `INF-04` | `qresearch/data/universe.py` — announcement-lag conversion, `assert_point_in_time`, tradability masks | a Q1 value produced 2024-03-31 but announced 2024-04-25 is **invisible through 2024-04-24** and present from 2024-04-25, at *feature* and at *model-prediction* level; the deliberately leaky report-date variant is caught (`tests/test_pit_announcement_lag.py`) |
| `INF-05` | `qresearch/data/labels.py` — `LabelSpec`, forward returns, rank/z targets, availability mask, purge radius | no row is dropped for being unavailable; a suspension on the **execution** date invalidates the label |
| `INF-06` | `qresearch/data/processors.py` — rolling/expanding/CS normalizers + `assert_causal` | an extreme outlier at `t+1` leaves the standardization at `t` **bit-identical**; the leaky global z-score is refused unless explicitly flagged and is caught by both probes (`tests/test_rolling_normalization.py`) |
| `INF-07` | `qresearch/data/handlers.py` — `PanelBundle`, `build_sequence_bundle`, `masked_mean` | **no global dropna**: invalid rows stay in the tensor with `mask=False`, a neutral fill and a per-step `step_mask`; `masked_mean` excludes them and returns `nan` (not `0.0`) when nothing is valid (`tests/test_handler_mask.py`) |
| `INF-08` | `qresearch/data/audit.py` — `LeakageAuditor`, `DataLeakageDetectedError`, `audit_prefix_invariance` | time-permuted labels are unpredictable under a purged split; a contaminated split (duplicates across the boundary) **is** rejected; the audit is seed-reproducible; prefix replay catches a full-sample feature pipeline (`tests/test_leakage_auditor.py`) |
| `ST-01` | `qresearch/stats/ic.py` — `rank_ic`, `pearson_ic`, `ic_moments` (pulled forward: the audit needs it) | per-date statistics match `scipy` on the same cross-sections |
| `ST-02` | `qresearch/stats/hac.py` — `newey_west`, `newey_west_se`, `nw_t_stat`, `NeweyWestResult` | the Bartlett long-run variance matches `statsmodels` to `1e-8` at the documented call sites (`OLS.fit(cov_type="HAC")`, `sandwich_covariance.cov_hac`); the automatic bandwidth `floor(4 (T/100)^(2/9))`, the `sqrt(T/(T-k_reg))` factor and the AR(1) pre-whitening re-colouring are each pinned against closed forms; on a positively autocorrelated IC series the HAC error is shown to exceed the i.i.d. one, which is the reason 2.2.2 forbids the latter (`tests/test_hac.py`) |

**Statistical note on the audit (`INF-08`).** The test is the one-sided t-test of the
**shuffled-label** IC against zero, not a comparison of the real-label statistic with a shuffle null:
a genuinely skilful model scores high on real labels and a memorising pipeline scores high on
permuted labels, so the latter construction has no power. Its sensitivity limit is documented *and
tested*: a linear audit model cannot exploit duplicate-row contamination, while a lookup-capable
model can. Practical rule: run the audit with a model at least as expressive as the production model.

## 8. Next tasks

`ST-03` (significance tests / DM) → `ST-04`–`ST-07` (bootstrap, quantile monotonicity, multiple
testing, deflated Sharpe) → `ST-08`–`ST-10` (purged splitters, reports, statistical validation
suite).

Interface item that `ST-02` unblocks but does not close: `ic_moments` (`ST-01`) still reports the
i.i.d. reference statistic only, so the `t_nw` / `p_nw` / `nw_bandwidth` fields of the `ic_summary`
row in `PROJECT_SPEC.md` 3.4.6 are not exposed by any function yet. `ST-03` is the task that builds
that inference layer on top of `newey_west`.

Also outstanding from `INF-07`: the `DataHandlerLP`/`Alpha158` expression wiring and a real-store
panel loader, so the bundle builder of `qresearch.data.handlers` can be driven by the Qlib data
layer directly rather than by caller-supplied panels.
