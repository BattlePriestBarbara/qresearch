"""Point-in-time universe, announcement-lag conversion and tradability masks (task ``INF-04``).

The central statistical hazard this module eliminates (``PROJECT_SPEC.md`` 1.2.1, 2.1):

.. math::

    \\text{leakage} \\iff \\exists\\, x^{(j)}_{i,t} \\text{ such that }
    \\sigma\\!\\left(x^{(j)}_{i,t}\\right) \\not\\subseteq \\mathcal{F}_t

Fundamental data carries a **publication delay**: the value for the period ending 31 March may
only be announced on 25 April.  Indexing such data by its *report date* silently reveals it on
31 March and destroys every downstream inference.  :func:`to_point_in_time` re-indexes by the
**announcement** date instead, and :func:`assert_point_in_time` independently verifies the
result, so the property is checked rather than assumed.

The leaky variant (``mode="report_date"``) exists on purpose, guarded by an explicit flag: tests
use it to demonstrate that the audit *has power*, because a leaky pipeline must fail the same
assertion a correct pipeline passes (negative control, ``PROJECT_SPEC.md`` 3.8).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import numpy as np
import pandas as pd

__all__ = [
    "AnnouncementLagSpec",
    "TradabilityRules",
    "assert_point_in_time",
    "build_tradability_mask",
    "first_visible_date",
    "load_instruments",
    "membership_mask",
    "to_panel",
    "to_point_in_time",
]

_DATETIME_LEVEL: Final[str] = "datetime"
_INSTRUMENT_LEVEL: Final[str] = "instrument"
VisibilityMode = Literal["announcement", "report_date"]


@dataclass(frozen=True)
class AnnouncementLagSpec:
    """Column layout and validation policy for a fundamental-data panel.

    Attributes
    ----------
    value_cols : tuple[str, ...]
        Columns carrying the values to be published.
    instrument_col : str
        Column identifying the asset.
    report_col : str
        Column holding the period end date ("which quarter the number describes").
    announce_col : str
        Column holding the publication date ("when the market learned it").
    allow_negative_lag : bool
        Whether ``announce_col < report_col`` is tolerated.  ``False`` by default: a negative lag
        means the source data is inconsistent and MUST be fixed before use.
    """

    value_cols: tuple[str, ...]
    instrument_col: str = "instrument"
    report_col: str = "report_date"
    announce_col: str = "announce_date"
    allow_negative_lag: bool = False


def _visibility_column(spec: AnnouncementLagSpec, mode: VisibilityMode) -> str:
    """Return the column that defines when a value becomes part of the information set."""
    if mode == "announcement":
        return spec.announce_col
    if mode == "report_date":
        return spec.report_col
    raise ValueError(f"mode must be 'announcement' or 'report_date', got {mode!r}")


def to_point_in_time(
    fundamentals: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    spec: AnnouncementLagSpec,
    *,
    mode: VisibilityMode = "announcement",
    allow_leaky: bool = False,
    max_staleness: pd.Timedelta | int | None = None,
) -> pd.DataFrame:
    """Re-index fundamental data so that each value appears only once it is knowable.

    Parameters
    ----------
    fundamentals : pd.DataFrame
        One row per (instrument, report period) with the columns named in ``spec``.
    calendar : pd.DatetimeIndex
        Trading calendar of the target panel.
    spec : AnnouncementLagSpec
        Column layout and validation policy.
    mode : {"announcement", "report_date"}
        ``"announcement"`` (default, correct) publishes a value from its announcement date.
        ``"report_date"`` publishes it from the period end date, which is a **deliberate
        look-ahead**; it requires ``allow_leaky=True`` and exists only for negative controls.
    allow_leaky : bool
        Must be ``True`` to use ``mode="report_date"``.
    max_staleness : pd.Timedelta | int | None
        Optional expiry: a value stops being published this long after becoming visible.  An
        ``int`` is interpreted as **trading periods**; a ``Timedelta`` as calendar time.
        ``None`` (default) keeps the value until it is superseded.

    Returns
    -------
    pd.DataFrame
        Sparse panel indexed by ``MultiIndex(datetime, instrument)`` holding exactly the rows on
        which a value is published.  :func:`to_panel` expands it to a dense grid with ``NaN`` in
        the not-yet-knowable cells.

    Raises
    ------
    ValueError
        On a leaky mode without ``allow_leaky``, on missing columns, or on a negative
        announcement lag when ``spec.allow_negative_lag`` is ``False``.
    """
    if mode == "report_date" and not allow_leaky:
        raise ValueError(
            "mode='report_date' is a deliberate look-ahead and is refused unless allow_leaky=True. "
            "It exists only to build negative controls for the leakage audit (INF-08)."
        )
    required = {spec.instrument_col, spec.report_col, spec.announce_col, *spec.value_cols}
    missing = required.difference(fundamentals.columns)
    if missing:
        raise ValueError(f"fundamentals is missing required columns: {sorted(missing)}")
    if not isinstance(calendar, pd.DatetimeIndex):
        raise TypeError("calendar must be a pd.DatetimeIndex")

    data = fundamentals.copy()
    for column in (spec.report_col, spec.announce_col):
        data[column] = pd.to_datetime(data[column])
    lag_days = (data[spec.announce_col] - data[spec.report_col]).dt.days
    if not spec.allow_negative_lag and bool((lag_days < 0).any()):
        offenders = data.loc[lag_days < 0, [spec.instrument_col, spec.report_col, spec.announce_col]]
        raise ValueError(
            "found an announcement dated before its report period end, which is inconsistent "
            f"source data:\n{offenders.head().to_string()}"
        )

    visibility_column = _visibility_column(spec, mode)
    ordered_calendar = calendar.sort_values()
    calendar_values = ordered_calendar.to_numpy()
    data = data.sort_values([spec.instrument_col, visibility_column])

    blocks: list[pd.DataFrame] = []
    for instrument, events in data.groupby(spec.instrument_col, sort=True):
        visibility = events[visibility_column].to_numpy()
        starts = np.searchsorted(calendar_values, visibility, side="left")
        stops = np.append(starts[1:], len(calendar_values))
        if isinstance(max_staleness, int):
            stops = np.minimum(stops, starts + max_staleness)
        elif max_staleness is not None:
            expiry = np.searchsorted(calendar_values, visibility + max_staleness.to_timedelta64(), side="right")
            stops = np.minimum(stops, expiry)
        for index in range(len(visibility)):
            start, stop = int(starts[index]), int(stops[index])
            if stop <= start:
                continue
            block = pd.DataFrame(
                {
                    _DATETIME_LEVEL: calendar_values[start:stop],
                    _INSTRUMENT_LEVEL: instrument,
                    **{column: events[column].to_numpy()[index] for column in spec.value_cols},
                }
            )
            blocks.append(block)

    if not blocks:
        # No announcement falls inside the calendar: return an explicitly typed empty panel.  A
        # column of dtype ``object`` here would silently propagate into downstream arrays and break
        # numeric code with a casting error far from the cause.
        empty_index = pd.MultiIndex.from_arrays([[], []], names=[_DATETIME_LEVEL, _INSTRUMENT_LEVEL])
        return pd.DataFrame({column: pd.Series(dtype="float64") for column in spec.value_cols}, index=empty_index)
    published = pd.concat(blocks, ignore_index=True)
    published[_DATETIME_LEVEL] = pd.to_datetime(published[_DATETIME_LEVEL])
    return published.set_index([_DATETIME_LEVEL, _INSTRUMENT_LEVEL]).sort_index()


def to_panel(
    published: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    instruments: list[str],
    value_cols: tuple[str, ...],
) -> dict[str, pd.DataFrame]:
    """Expand a sparse published panel into dense ``(date x instrument)`` frames.

    Missing cells - values that were not yet knowable - remain ``NaN``; they are never
    forward-filled silently.  An explicit ``NaN`` is exactly what makes "this was not in
    :math:`\\mathcal{F}_t`" observable, which is what the leakage audit asserts on.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping ``value column -> frame`` indexed by date, one column per instrument.
    """
    grid = pd.MultiIndex.from_product([calendar, instruments], names=[_DATETIME_LEVEL, _INSTRUMENT_LEVEL])
    dense = published.reindex(grid)
    panels: dict[str, pd.DataFrame] = {}
    for column in value_cols:
        # Coerce defensively: an empty or object-dtype published panel must not leak an `object`
        # dtype into downstream numeric arrays (PROJECT_SPEC.md 3.4.1 requires float blocks).
        numeric = pd.to_numeric(dense[column], errors="coerce").astype("float64")
        panels[column] = numeric.unstack(_INSTRUMENT_LEVEL)
    return panels


def first_visible_date(
    fundamentals: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    spec: AnnouncementLagSpec,
    *,
    instrument: str,
) -> pd.Timestamp | None:
    """Return the first trading date on which ``instrument`` has a knowable value.

    This is the first calendar date ``>= announce_date`` for that instrument, i.e. the date on
    which the value enters :math:`\\mathcal{F}_t`.  ``None`` when the announcement falls after
    the end of the calendar.
    """
    rows = fundamentals[fundamentals[spec.instrument_col] == instrument]
    if rows.empty:
        return None
    announce = pd.to_datetime(rows[spec.announce_col]).min().to_datetime64()
    ordered = calendar.sort_values()
    position = int(np.searchsorted(ordered.to_numpy(), announce, side="left"))
    if position >= len(ordered):
        return None
    return pd.Timestamp(ordered[position])


def assert_point_in_time(
    published: pd.DataFrame,
    fundamentals: pd.DataFrame,
    spec: AnnouncementLagSpec,
    *,
    leaky_mode: bool = False,
) -> None:
    """Independently verify that every published row was knowable at its own date.

    For each row of ``published`` at date :math:`t` the check asserts that the value equals the
    most recent announcement with ``announce_date <= t``.  With ``leaky_mode=True`` the same
    assertion is applied to the *report* date, which is what makes a leaky pipeline fail.

    Raises
    ------
    AssertionError
        Listing the offending ``(datetime, instrument)`` pairs, so a failure is diagnosable.
    """
    visibility_column = spec.report_col if leaky_mode else spec.announce_col
    rows = fundamentals.copy()
    rows[visibility_column] = pd.to_datetime(rows[visibility_column])
    violations: list[str] = []
    for (timestamp, instrument), published_row in published.iterrows():
        candidates = rows[(rows[spec.instrument_col] == instrument) & (rows[visibility_column] <= timestamp)]
        if candidates.empty:
            violations.append(f"({timestamp:%Y-%m-%d}, {instrument}) published with no announcement on or before t")
            continue
        latest = candidates.sort_values(visibility_column).iloc[-1]
        for column in spec.value_cols:
            if not np.isclose(float(published_row[column]), float(latest[column])):
                violations.append(
                    f"({timestamp:%Y-%m-%d}, {instrument}) {column}={published_row[column]!r} but the latest "
                    f"knowable value is {latest[column]!r} (visible {latest[visibility_column]:%Y-%m-%d})"
                )
    if violations:
        extra = f"\n  ... and {len(violations) - 10} more" if len(violations) > 10 else ""
        raise AssertionError(
            "point-in-time violation(s) detected - a value was published before it was knowable:\n  "
            + "\n  ".join(violations[:10])
            + extra
        )


# ---------------------------------------------------------------------------------------
# Tradability: suspension, limit-lock, seasoning and index membership
# ---------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TradabilityRules:
    """Rules used to build the boolean tradability mask.

    Every rule is evaluated with data observable at :math:`t` (i.e. from :math:`\\mathcal{F}_t`),
    which is both correct for a decision taken at the close of :math:`t` and sufficient: the
    *execution-time* limit checks belong to the exchange adapter (``PO-01``), not here.

    Attributes
    ----------
    min_listed_days : int
        Minimum number of observed closes before an instrument is considered seasoned.  Avoids
        pricing a fresh listing whose history is too short for a rolling window.
    require_volume : bool
        Treat a zero (or missing) volume as a suspension.
    limit_threshold : float | None
        Exclude names whose absolute close-to-close move reaches this level, which is the
        conventional A-share daily price limit.  ``None`` disables the rule.
    """

    min_listed_days: int = 60
    require_volume: bool = True
    limit_threshold: float | None = 0.095


def load_instruments(path: Path) -> pd.DataFrame:
    """Load a Qlib ``instruments/*.txt`` file into an interval table.

    Returns
    -------
    pd.DataFrame
        Columns ``["instrument", "start", "end"]``; one row per membership interval, which is the
        point-in-time form used by :func:`membership_mask`.  ``SH000905`` ends on 2022-06-09 while
        ``SH600000`` ends on 2020-09-25 in the shipped store, so the file records both entries and
        exits rather than a static snapshot.
    """
    rows: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        rows.append((parts[0], pd.Timestamp(parts[1]), pd.Timestamp(parts[2])))
    frame = pd.DataFrame(rows, columns=["instrument", "start", "end"])
    if frame.empty:
        raise ValueError(f"no instrument intervals parsed from {path}")
    return frame.sort_values(["instrument", "start"]).reset_index(drop=True)


def membership_mask(
    intervals: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    *,
    instruments: list[str] | None = None,
) -> pd.DataFrame:
    """Return a ``(date x instrument)`` boolean membership mask.

    Membership at :math:`t` is decided by the interval ``[start, end]`` recorded for the
    instrument, so a constituent removed from an index stops being tradable on the day it leaves.
    """
    universe = sorted(instruments) if instruments is not None else sorted(intervals["instrument"].unique())
    mask = pd.DataFrame(False, index=pd.DatetimeIndex(calendar), columns=universe)
    for row in intervals.itertuples(index=False):
        if row.instrument not in mask.columns:
            continue
        in_interval = (mask.index >= row.start) & (mask.index <= row.end)
        mask.loc[in_interval, row.instrument] = True
    return mask


def listing_dates(intervals: pd.DataFrame) -> pd.Series:
    """Return the first membership date per instrument (the seasoning reference point)."""
    return intervals.groupby("instrument")["start"].min()


def build_tradability_mask(
    panels: dict[str, pd.DataFrame],
    calendar: pd.DatetimeIndex,
    *,
    intervals: pd.DataFrame | None = None,
    rules: TradabilityRules | None = None,
) -> pd.DataFrame:
    """Build the point-in-time tradability mask from price and volume panels.

    Parameters
    ----------
    panels : dict[str, pd.DataFrame]
        At least ``{"open", "close", "volume"}``, each indexed by date with one column per
        instrument.
    calendar : pd.DatetimeIndex
        Target calendar; columns are restricted to this index.
    intervals : pd.DataFrame | None
        Index membership intervals (see :func:`load_instruments`).  When supplied, membership and
        seasoning are also enforced.
    rules : TradabilityRules | None
        Rules to apply; defaults to :class:`TradabilityRules`.

    Returns
    -------
    pd.DataFrame
        Boolean mask, ``True`` where the instrument could have been traded on that date.

    Notes
    -----
    Suspended names keep ``mask == False`` and are *never* dropped from the panel: the handler
    (``INF-07``) fills their features with a neutral value and propagates the mask, so that a
    suspension cannot silently remove an asset from the sample (survivorship bias).
    """
    active_rules = rules or TradabilityRules()
    missing = {"close", "volume"}.difference(panels)
    if missing:
        raise ValueError(f"panels must contain {sorted(missing)} to evaluate tradability")

    close = panels["close"].reindex(calendar)
    volume = panels["volume"].reindex(calendar)
    mask = close.notna()

    if active_rules.require_volume:
        mask &= volume.fillna(0.0) > 0.0
    if active_rules.min_listed_days > 0:
        mask &= close.notna().cumsum() >= active_rules.min_listed_days
    if intervals is not None:
        member = membership_mask(intervals, calendar, instruments=list(close.columns))
        mask &= member
    if active_rules.limit_threshold is not None:
        previous_close = close.shift(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            relative_move = (close / previous_close - 1.0).abs()
        mask &= ~(relative_move >= active_rules.limit_threshold)
    return mask.astype(bool)
