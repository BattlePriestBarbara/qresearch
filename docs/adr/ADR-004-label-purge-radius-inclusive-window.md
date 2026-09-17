# ADR-004 — Label purge radius: inclusive price window (one extra period, conservatively)

* **Status:** accepted
* **Date:** 2026-09-16
* **Tasks:** `INF-05` (labels), `ST-08` (purged splitters, pending), `INF-08` (audit)
* **Clause concerned:** `PROJECT_SPEC.md` 2.2.5 (purge window)

## Context

`PROJECT_SPEC.md` 2.1.1 defines the target on execution prices,

.. math:: y_{i,t+1} = \frac{P^{open}_{i,\;t+1+h}}{P^{open}_{i,\;t+1}} - 1 ,

so the label decided at :math:`t` consumes prices over the closed interval
:math:`[t + \text{lag},\; t + \text{lag} + h]`.  Two decision dates :math:`t_1 \le t_2` therefore
share information whenever their windows intersect, i.e. whenever
:math:`(t_2 + \text{lag}) \le (t_1 + \text{lag} + h)`.  The execution lag cancels, giving

.. math:: t_2 - t_1 \le h \quad\Longleftrightarrow\quad \text{the labels overlap.}

Section 2.2.5 states the purge window literally as :math:`[j - h + 1,\; j + h - 1]`, which is the
half-open convention (label window :math:`[t, t+h)`), and purges one period fewer than the closed
window above.

## Decision

`LabelSpec.purge_radius` returns :math:`h`, i.e. the **closed-window** radius, and the purged
splitters and the leakage audit use it.  Every sample within :math:`h` periods of an evaluation date
is removed from the training side of the split.

## Rationale

* The two conventions differ by exactly one period, and the difference is *only* in the direction of
  removing more information.  Choosing the larger radius cannot manufacture signal; choosing the
  smaller one can, which is the costly error.
* The closed window matches the target definition literally: the entry price
  :math:`P^{open}_{t+\text{lag}}` is used by the label at :math:`t` **and** is the exit-relevant
  observation of a neighbouring label, so it must not be shared across the boundary.
* The cost is a marginal loss of training rows (one period per fold boundary), which is immaterial
  against the risk of a contaminated fold.

## Consequences

* `LabelSpec.purge_radius == horizon` regardless of the execution lag, which is surprising at first
  sight; the derivation is in the property's docstring so the next reader does not "fix" it back.
* `ST-08` MUST implement `PurgedKFold` with this radius, and its test MUST assert that no training
  date lies within `purge_radius` of any evaluation date.
* The statistical validation suite (`ST-10`) MUST include an empirical check that a purged split with
  this radius yields a shuffled-label IC centred on zero (which `INF-08`'s audit already covers at
  the row level).
