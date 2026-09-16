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
| `environment.yml` | Conda spec: `python=3.12.7` + the two requirement files |
| `requirements.lock.txt` | **Exact transitive closure** (210 distributions) of the verified environment, offline, with provenance notes |
| `requirements.lock.hashes.txt` | **sha256 digest lock** (202 distributions) usable with `pip install --require-hashes` |

```powershell
# reproduce the environment
C:\path\to\conda.exe env create -f environment.yml
conda activate qresearch
D:\Anaconda3\python.exe -m pip install -e . --no-deps     # editable, src-layout

# verify the contract (exit 0 required)
D:\Anaconda3\python.exe scripts\check_env.py

# regenerate the lock artefacts (offline closure; --hashes also fetches sha256 digests)
D:\Anaconda3\python.exe scripts\lock_requirements.py --hashes

# hermetic install from the digest lock
D:\Anaconda3\python.exe -m pip install --require-hashes -r requirements.lock.hashes.txt
```

**Float-critical policy.** `numpy`, `scipy`, `pandas` and `torch` are installed from **PyPI
wheels**, never from a conda channel: conda and PyPI builds differ in BLAS/LAPACK linkage and
therefore in the last bits of every computed moment. `environment.yml` documents this.

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
| `INF-01` exact locking | `requirements.txt`, `requirements-dev.txt`, `environment.yml`, `requirements.lock.txt`, `scripts/lock_requirements.py` | pin-policy tests; hash-lock verification via `--require-hashes` |
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

## 6. Next tasks

`INF-03` (Qlib init + data sanity CLI) → `INF-04` (point-in-time universe) → `INF-05` (labels) →
`INF-06` (features) → `INF-07` (handler/dataset) → `INF-08` (leakage audit harness).
