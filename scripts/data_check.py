"""Qlib initialisation and data-store sanity check (task ``INF-03``).

``PROJECT_SPEC.md`` assigns ``INF-03`` the job of proving the data layer is usable *and* of
reporting what it actually contains, so that no study silently assumes a calendar the store does
not have.  The store shipped with this project ends on **2020-09-25**, which makes every example
date range in the specification invalid - a fact that must be surfaced by a command, not discovered
after a failed run.

The command performs, in order:

1. the environment-contract verification (import-time, ``INF-01``);
2. the read-only provider checks (``assert_provider_isolation``);
3. ``qlib.init`` inside a :class:`~qresearch.env.ProviderWriteGuard`, so a cache written into the
   market data is detected immediately;
4. a data inventory: calendar span, instrument count, feature sample, missing-value rate;
5. a **domain check** comparing the requested range with the store's calendar that fails loudly when
   the request reaches beyond the data.

Exit codes
----------
0 - store is usable and the requested range is covered
1 - environment-contract violation
2 - read-only provider violation
3 - cache-like pollution inside the provider
4 - the qresearch package cannot be imported
5 - the requested date range is not covered by the store (or the store is unusable)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:  # pragma: no cover - bootstrap for uninstalled checkouts
    sys.path.insert(0, str(_SRC))


def _collect_report(provider_uri: Path, market: str, start_time: str, end_time: str, sample: str) -> dict[str, object]:
    """Initialise Qlib under the write guard and collect the data inventory."""
    import qlib
    from qlib.data import D

    from qresearch.env import ProviderWriteGuard, build_qlib_init_kwargs

    kwargs = build_qlib_init_kwargs(provider_uri)
    report: dict[str, object] = {
        "provider_uri": str(provider_uri),
        "requested": {"start": start_time, "end": end_time},
    }

    with ProviderWriteGuard(provider_uri):
        qlib.init(**kwargs)
        calendar = D.calendar(start_time="1990-01-01", end_time="2100-01-01", freq="day")
        report["calendar"] = {
            "first": str(calendar[0].date()),
            "last": str(calendar[-1].date()),
            "n_trading_days": len(calendar),
        }
        instrument_list = D.list_instruments(
            D.instruments(market=market),
            start_time=str(calendar[0].date()),
            end_time=str(calendar[-1].date()),
            as_list=True,
        )
        report["universe"] = {
            "market": market,
            "n_instruments": len(instrument_list),
            "sample": instrument_list[:5],
        }
        frame = D.features(
            [sample],
            fields=["$open", "$close", "$volume", "$factor"],
            start_time=start_time,
            end_time=end_time,
            freq="day",
        )
        report["sample_features"] = {
            "instrument": sample,
            "n_rows": len(frame),
            "columns": list(frame.columns),
            "missing_rate": float(frame.isna().to_numpy().mean()) if len(frame) else 1.0,
            "head": frame.head(2).reset_index().to_dict(orient="records"),
            "tail": frame.tail(2).reset_index().to_dict(orient="records"),
        }
    return report


def _check_domain(report: dict[str, object], start_time: str, end_time: str) -> list[str]:
    """Return domain problems, e.g. a request beyond the store's calendar."""
    problems: list[str] = []
    calendar = report.get("calendar")
    if not isinstance(calendar, dict) or "last" not in calendar:
        return ["the store exposes no calendar, so it cannot be used"]
    store_first, store_last = str(calendar["first"]), str(calendar["last"])
    if start_time < store_first:
        problems.append(f"requested start {start_time} precedes the store's first trading day {store_first}")
    if end_time > store_last:
        problems.append(
            f"requested end {end_time} is beyond the store's last trading day {store_last} - the data "
            "snapshot does not cover it; shorten the study window or obtain a newer snapshot"
        )
    sample = report.get("sample_features")
    if isinstance(sample, dict) and int(sample.get("n_rows", 0)) == 0:
        problems.append(f"no feature rows returned for the sample instrument {sample.get('instrument')!r}")
    return problems


def main() -> int:
    """Run the data-store check and print the inventory."""
    parser = argparse.ArgumentParser(description="Verify the Qlib data store (PROJECT_SPEC.md task INF-03).")
    parser.add_argument("--provider-uri", type=Path, default=None)
    parser.add_argument("--market", default="csi300", help="Qlib market name (csi100/csi300/csi500/all)")
    parser.add_argument("--start-time", default="2019-01-01")
    parser.add_argument("--end-time", default="2020-09-25")
    parser.add_argument("--sample", default="SH600000")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--allow-out-of-range", action="store_true", help="report domain problems without failing")
    args = parser.parse_args()

    try:
        from qresearch import PROVIDER_URI
        from qresearch.env import EnvironmentContractError, ReadOnlyViolationError, assert_provider_isolation
    except EnvironmentContractError as contract_error:
        print(str(contract_error), file=sys.stderr)
        return 1
    except ImportError as import_error:  # pragma: no cover - packaging failure
        print(f"FATAL: cannot import the qresearch package: {import_error!r}", file=sys.stderr)
        return 4

    provider_uri = Path(args.provider_uri) if args.provider_uri is not None else PROVIDER_URI
    try:
        assert_provider_isolation(provider_uri)
    except ReadOnlyViolationError as violation:
        print(f"Read-only data-source violation:\n  {violation}", file=sys.stderr)
        return 2

    try:
        report = _collect_report(provider_uri, args.market, args.start_time, args.end_time, args.sample)
    except ReadOnlyViolationError as violation:
        print(f"Read-only data-source violation:\n  {violation}", file=sys.stderr)
        return 2
    except Exception as error:  # pragma: no cover - depends on the store
        print(f"FATAL: the data store could not be used: {error!r}", file=sys.stderr)
        return 5

    problems = _check_domain(report, args.start_time, args.end_time)
    report["domain_problems"] = problems

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
    else:
        calendar = report["calendar"]
        universe = report["universe"]
        sample = report["sample_features"]
        print("qresearch data store check: INVENTORY")
        print(f"  provider   : {report['provider_uri']} (read-only)")
        print(
            f"  calendar   : {calendar['first']} .. {calendar['last']} " f"({calendar['n_trading_days']} trading days)"
        )
        print(
            f"  universe   : {universe['market']} -> {universe['n_instruments']} instruments, "
            f"e.g. {universe['sample']}"
        )
        print(f"  sample     : {sample['instrument']} cols={sample['columns']}")
        print(f"               rows={sample['n_rows']}, missing_rate={sample['missing_rate']:.4f}")
        print(f"  requested  : {args.start_time} .. {args.end_time}")
        for problem in problems:
            print(f"  PROBLEM    : {problem}", file=sys.stderr)

    if problems and not args.allow_out_of_range:
        print(
            "\nThe requested window is not covered by the store. This is a data-domain failure, not a "
            "code failure: adjust the study window or the snapshot (PROJECT_SPEC.md 3.1, task INF-03).",
            file=sys.stderr,
        )
        return 5
    if problems:
        print("WARNING: domain problems reported but tolerated (--allow-out-of-range)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
