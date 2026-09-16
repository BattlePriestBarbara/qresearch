# ADR-003 — Hybrid environment management: conda for native extensions, pip hashes for the Python stack

* **Status:** accepted
* **Date:** 2026-09-15
* **Tasks:** `INF-01` (dependency locking), `INF-02` (environment specification)
* **Clauses concerned:** `PROJECT_SPEC.md` 3.1 (runtime environment), 1.2.2 (version drift)

## Context

`INF-01` requires a fully hash-locked dependency set so that no silent version or build drift can
perturb a floating-point result.  A pure `pip` resolution cannot achieve that here:

* `pyqlib==0.9.7` requires `gym`, and `gym` publishes **no wheel at all** on PyPI (sdist only);
* several native extensions — e.g. `psutil==5.9.0`, `pywin32`, `pywinpty==2.0.10` — were released
  before CPython 3.12 and ship wheels for older interpreters only;
* forcing pip to compile those from source would require a working MSVC toolchain, would pull
  unpinned build dependencies into the environment, and would produce artifacts whose bytes
  differ per machine — i.e. exactly the non-determinism this project forbids.

## Decision

Adopt a **hybrid environment strategy**:

> **底层 C/C++ 扩展由 Conda 托管，纯 Python 科学计算栈由 Pip Hash 锁定。**
> *Low-level C/C++ extensions are managed by conda; the pure-Python scientific stack is
> hash-locked by pip.*

Concretely:

1. `environment.yml` is the single entry point.  Its conda `dependencies:` section pins
   `python=3.12.7` plus every distribution that has no wheel compatible with this interpreter
   (the conda-managed set), each at an exact version with its build string.
2. Its `pip:` section references **`requirements.lock.pip.txt`** — the dependency closure with
   the conda-managed distributions removed — so conda and pip never contend for the same package.
3. `requirements.lock.hashes.txt` provides the same pip-managed set with sha256 digests, for
   hermetic (`--require-hashes`) reinstalls and for CI.
4. `scripts/lock_requirements.py` derives the split **mechanically** from wheel tags, so the
   conda-managed set is computed, not hand-maintained:

   | Category | Rule | Managed by |
   |---|---|---|
   | `wheel` | a wheel tagged `cp312`, `py3*`, or `cp3X-abi3` (`X <= 12`) exists for this platform | pip (hash-pinned) |
   | `sdist_pure_python` | the release ships **no wheel at all** → pure Python, no compiler needed | pip (sdist digest pinned) |
   | `conda` | wheels exist but none compatible with this interpreter | conda (`environment.yml`) |

## Rationale

* **The split follows the artefact, not a preference.** A native extension with no compatible
  wheel is precisely the case where conda adds value; a pure-Python package with no wheel is
  precisely the case where pip is safe.  Encoding that rule in code keeps the two managers from
  drifting apart.
* **`gym` is not a C/C++ case.** Its sdist contains no extension modules, so pip builds it
  without a compiler.  It therefore stays pip-managed with its sdist digest pinned, and the
  verified environment (which already contains `gym==0.26.2` built by pip) is preserved exactly.
  conda-forge only reaches `gym 0.26.1`, so delegating it to conda would silently downgrade a
  dependency of the verified environment.
* **An earlier hand-written list of "8 wheel-less packages" was partly wrong.** The first
  classifier ignored stable-ABI (`cp3X-abi3`) wheels and therefore mis-classified `clarabel`,
  `cryptography`, `tornado` and `argon2-cffi-bindings`, all of which install fine on CPython 3.12.
  After the fix the conda-managed surface is smaller and better justified.

## Consequences

* Two managers coexist, so the environment is created in two documented steps (`conda env create`
  then the referenced pip install); `README.md` section 1 walks through it.
* `pip check` cannot see conda packages, so completeness of the environment is verified by
  `scripts/check_env.py` (version assertions) rather than by pip alone.
* A test asserts that `environment.yml`'s conda section contains **exactly** the set that the
  classifier delegates to conda, and that neither lock file lists those distributions.  Adding a
  package to one side without the other fails the suite.
* Regeneration is one command (`scripts/lock_requirements.py --hashes`); nothing in this split is
  maintained by hand.
* **Provenance nuance, recorded deliberately:** the classifier splits by *wheel availability*, not
  by how the current environment happened to be installed.  In the verified environment `pandas`,
  `statsmodels` and `scikit-learn` come from conda builds; a fresh `conda env create` installs the
  same versions from hash-verified PyPI wheels.  Versions are asserted at runtime, so the contract
  holds either way; if byte-level fidelity to the existing environment is ever required, those
  three move into the conda section (a one-line change plus a regeneration).
