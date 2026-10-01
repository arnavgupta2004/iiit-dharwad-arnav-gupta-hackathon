"""GATE A pilot: one trading week of GDELT GKG at 1/4 sampling (+ the June 2022 FOMC days).

Streams files in memory into data/processed/gdelt_gkg_pilot (gitignored), then writes
reports/gkg_pilot.json with: title coverage, title-linked headlines per ticker per week
(all hours vs a US-session-only subset), unique-title counts, and daily MKT volume.

    python scripts/gkg_pilot.py [--report-only]
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

from riskpulse.common.config import load_config, repo_root
from riskpulse.engine.entities import EntityLinker
from riskpulse.ingestion.gdelt_gkg import GKGStreamer, gkg_timestamps, read_gkg_rows, relink

PILOT_DIR = repo_root() / "data" / "processed" / "gdelt_gkg_pilot"
REPORT = repo_root() / "reports" / "gkg_pilot.json"
ALL_HOURS = {"minute": 0, "weekday_hours_utc": list(range(24)), "weekend_hours_utc": [0, 6, 12, 18]}
US_SESSION_HOURS = set(range(13, 22))  # 13:00-21:00 UTC

WEEK = (datetime(2022, 2, 22), datetime(2022, 3, 1))  # trading days Feb 22-25, 28
FOMC = (datetime(2022, 6, 14), datetime(2022, 6, 17))  # FOMC decision 2022-06-15


def stream() -> dict:
    ts = gkg_timestamps(*WEEK, ALL_HOURS) + gkg_timestamps(*FOMC, ALL_HOURS)
    s = GKGStreamer(out_dir=PILOT_DIR).run(ts)
    return s.__dict__


def report(stream_stats: dict | None) -> dict:
    universe = list(load_config("universe")["tickers"])
    linker = EntityLinker()
    prefixes = load_config("app")["gdelt_gkg"]["mkt_theme_prefixes"]
    stored = list(read_gkg_rows(PILOT_DIR))
    # Re-apply the current (stricter) linking rules to the stored superset of rows.
    rows = [x for x in (relink(r, linker, prefixes) for r in stored) if x]
    week = [r for r in rows if r["published_at"] < "2022-03-01"]
    fomc = [r for r in rows if r["published_at"] >= "2022-06-14"]

    def per_ticker(rs: list[dict], session_only: bool) -> dict[str, dict[str, int]]:
        cnt, uniq = Counter(), defaultdict(set)
        for r in rs:
            dt = datetime.fromisoformat(r["published_at"])
            if session_only and dt.weekday() < 5 and dt.hour not in US_SESSION_HOURS:
                continue
            if session_only and dt.weekday() >= 5:
                continue
            for t in r["tickers"]:
                cnt[t] += 1
                uniq[t].add(r["title"].lower())
        return {t: {"headlines": cnt[t], "unique_titles": len(uniq[t])} for t in universe}

    def mkt_daily(rs: list[dict]) -> dict[str, dict[str, int]]:
        d: dict[str, Counter] = defaultdict(Counter)
        for r in rs:
            if r["is_mkt"]:
                day = r["published_at"][:10]
                d[day]["mkt"] += 1
                for reg in r["regions"]:
                    d[day][reg] += 1
        return {k: dict(v) for k, v in sorted(d.items())}

    all_h = per_ticker(week, session_only=False)
    sess = per_ticker(week, session_only=True)
    out = {
        "pilot_week": [WEEK[0].date().isoformat(), WEEK[1].date().isoformat()],
        "fomc_days": [FOMC[0].date().isoformat(), FOMC[1].date().isoformat()],
        "sampling": "minute 00 of each hour (1/4 of 15-min files); weekends every 6 h",
        "stream_stats": stream_stats,
        "rows_stored_at_stream_time": len(stored),
        "rows_kept_current_rules": len(rows),
        "week_title_linked_per_ticker_all_hours": all_h,
        "week_title_linked_per_ticker_us_session_only": sess,
        "week_mean_headlines_per_ticker_all_hours": sum(v["headlines"] for v in all_h.values())
        / len(universe),
        "week_mean_unique_titles_per_ticker_all_hours": sum(
            v["unique_titles"] for v in all_h.values()
        )
        / len(universe),
        "week_min_unique_titles_per_ticker_all_hours": min(
            v["unique_titles"] for v in all_h.values()
        ),
        "week_tickers_below_5_unique": sorted(
            t for t, v in all_h.items() if v["unique_titles"] < 5
        ),
        "week_org_only_links": sum(1 for r in week if r["org_only_tickers"]),
        "mkt_daily_week": mkt_daily(week),
        "mkt_daily_fomc": mkt_daily(fomc),
        "generated_at": datetime.now(UTC).isoformat(),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    stats = None if "--report-only" in sys.argv else stream()
    if stats is None and Path(REPORT).exists():
        stats = json.loads(REPORT.read_text()).get("stream_stats")
    print(json.dumps(report(stats), indent=1))
