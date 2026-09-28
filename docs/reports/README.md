# Progress reports

Dated, human-readable snapshots of an increment. They exist so a reviewer can follow the
reasoning and the intermediate results without reconstructing them from the git log.

| Report | Date | Covers |
|---|---|---|
| `qresearch-项目进度报告-2026-09-26.doc` | 2026-09-26 | `INF-01`…`INF-11`, `ST-01`, `ST-02`: the verified environment, the src-layout and the gate, the point-in-time data pipeline, the permutation leakage audit and the Newey–West HAC estimator |

## Status of these files

* They are **Word documents, i.e. review material — not a source of truth.** Any number quoted
  from a report must be regenerated through a `scripts/` entry point (`PROJECT_SPEC.md` 3.2, 6.1)
  before it is used anywhere; `PROJECT_SPEC.md` stays normative.
* A report records the exact commands that produced its numbers, which on the author's machine
  includes machine-local paths. `ADR-005` governs the **toolchain files** — `src/`, `scripts/`,
  `tests/`, `configs/` and the root configuration files — and `scripts/prepublish_audit.py` scans
  exactly that set; a Word document is additionally skipped as binary. **Before a report leaves
  this repository, export it as text and strip those paths.**
* Wording, tables and figures are the author's, not generated output: unlike `artifacts/**`, a
  report is reviewed once and then frozen, so it is deliberately tracked as a binary.

## Regenerating the numbers behind a report

```powershell
python scripts/check_env.py                 # environment contract + honoured overrides
python scripts/data_check.py                # read-only inventory of the market-data store
python -m pytest -q -m "not slow"           # unit + leakage + statistical validation
python scripts/prepublish_audit.py          # pre-publish gate: paths, data, secrets
```

To diff a report against its predecessor, export it as text (`Save as → Plain text`) rather than
comparing screenshots.
