"""GDELT GKG 2.0 historical news adapter: stream 15-minute files, filter, keep compact rows.

Raw files are fetched into memory and never written to disk (GATE A condition). For each
record we keep the page title, URL, source domain, a trimmed theme list, organisations,
tone and the linked tickers. A record is kept if it links to a universe ticker (title or
GKG organisations) or qualifies as market-wide (MKT): a macro/geo/credit title keyword
plus a matching GKG theme prefix.

Output: ``data/processed/gdelt_gkg/YYYY-MM.jsonl.gz`` (append) plus ``_manifest.txt`` of
processed file timestamps so the stream can resume.
"""

from __future__ import annotations

import gzip
import html
import io
import json
import re
import shutil
import threading
import time
import zipfile
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from riskpulse.common.config import load_config, repo_root
from riskpulse.common.logging import get_logger
from riskpulse.engine.entities import MKT, EntityLinker

log = get_logger()

# GKG 2.1 column indices
C_DATE, C_SOURCE, C_URL, C_THEMES, C_LOCATIONS, C_ORGS, C_TONE, C_EXTRAS = 1, 3, 4, 7, 9, 13, 15, 26
_TITLE_RE = re.compile(r"<PAGE_TITLE>(.*?)</PAGE_TITLE>", re.S)


@dataclass
class StreamStats:
    files_ok: int = 0
    files_missing: int = 0
    files_failed: int = 0
    records_raw: int = 0
    records_seen: int = 0
    records_kept: int = 0


def gkg_timestamps(start: datetime, end: datetime, sampling: dict | None = None) -> list[datetime]:
    """15-minute file timestamps in [start, end) chosen by the sampling config."""
    s = sampling or load_config("app")["gdelt_gkg"]["sampling"]
    minute = int(s["minute"])
    wk, we = set(s["weekday_hours_utc"]), set(s["weekend_hours_utc"])
    out, t = [], start.replace(minute=minute, second=0, microsecond=0, tzinfo=UTC)
    while t < end.replace(tzinfo=UTC):
        hours = wk if t.weekday() < 5 else we
        if t >= start.replace(tzinfo=UTC) and t.hour in hours:
            out.append(t)
        t += timedelta(hours=1)
    return out


def parse_gkg(raw_zip: bytes) -> pd.DataFrame:
    """Parse one GKG zip (bytes) into a frame with the columns we use (titled rows only).

    The number of raw rows before the title filter is stored in ``df.attrs["n_raw"]``.
    """
    with zipfile.ZipFile(io.BytesIO(raw_zip)) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as fh:
            df = pd.read_csv(
                fh,
                sep="\t",
                header=None,
                dtype=str,
                quoting=3,
                encoding="utf-8",
                encoding_errors="replace",
                on_bad_lines="skip",
            )
    if df.shape[1] <= C_EXTRAS:
        return pd.DataFrame()
    out = pd.DataFrame(
        {
            "date": df[C_DATE],
            "domain": df[C_SOURCE],
            "url": df[C_URL],
            "themes": df[C_THEMES].fillna(""),
            "locations": df[C_LOCATIONS].fillna(""),
            "orgs": df[C_ORGS].fillna(""),
            "tone": df[C_TONE].fillna(""),
            "title": df[C_EXTRAS].fillna("").map(_extract_title),
        }
    )
    titled = out[out["title"].str.len() > 0].copy()
    titled.attrs["n_raw"] = len(out)
    return titled


def _extract_title(extras: str) -> str:
    m = _TITLE_RE.search(extras)
    return html.unescape(m.group(1)).strip() if m else ""


def _country_codes(locations: str) -> list[str]:
    # V1 locations: "type#fullname#countrycode#adm1#lat#lon#featureid;..."
    codes = {p.split("#")[2] for p in locations.split(";") if p.count("#") >= 3}
    return sorted(c for c in codes if c)


def filter_records(df: pd.DataFrame, linker: EntityLinker, theme_prefixes: list[str]) -> list[dict]:
    """Keep records linked to universe tickers (title or orgs) or market-wide (MKT)."""
    rows = []
    for r in df.itertuples(index=False):
        orgs = [o for o in r.orgs.split(";") if o]
        title_links = linker.link(text="", title=r.title)
        tickers = [m.ticker for m in title_links if m.ticker != MKT]
        org_links = (
            [] if tickers else [m for m in linker.link(text="", orgs=orgs) if m.ticker != MKT]
        )
        themes = [t for t in r.themes.split(";") if t]
        is_mkt = False
        if not tickers and not org_links:
            is_mkt = linker.is_market_wide(r.title) and any(
                t.startswith(tuple(theme_prefixes)) for t in themes
            )
            if not is_mkt:
                continue
        try:
            published = datetime.strptime(r.date, "%Y%m%d%H%M%S").replace(tzinfo=UTC)
        except (TypeError, ValueError):
            continue
        tone = r.tone.split(",")[0] if r.tone else ""
        rows.append(
            {
                "published_at": published.isoformat(),
                "title": r.title,
                "url": r.url,
                "domain": r.domain,
                "tickers": tickers,
                "org_only_tickers": [m.ticker for m in org_links],
                "is_mkt": is_mkt,
                "regions": list(linker.regions_of(r.title)),
                "countries": _country_codes(r.locations)[:10],
                "themes": [t for t in themes if t.startswith(tuple(theme_prefixes))][:15],
                "orgs": orgs[:15],
                "tone": float(tone) if tone else None,
            }
        )
    return rows


class GKGStreamer:
    """Fetch, filter and append GKG files for a list of timestamps (resumable, in-memory)."""

    def __init__(self, out_dir: Path | None = None) -> None:
        self.cfg = load_config("app")["gdelt_gkg"]
        self.out_dir = out_dir or repo_root() / self.cfg["out_dir"]
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = self.out_dir / "_manifest.txt"
        self.linker = EntityLinker()
        self.stats = StreamStats()
        self._lock = threading.Lock()
        self._session = requests.Session()
        self._stop = threading.Event()

    def done(self) -> set[str]:
        if not self.manifest.exists():
            return set()
        return set(self.manifest.read_text().split())

    def _free_gb(self) -> float:
        return shutil.disk_usage(self.out_dir).free / 1e9

    def _fetch(self, ts: datetime) -> bytes | None:
        url = self.cfg["url_template"].format(ts=ts.strftime("%Y%m%d%H%M%S"))
        for attempt in range(int(self.cfg["max_retries"])):
            try:
                r = self._session.get(url, timeout=self.cfg["timeout_seconds"])
                if r.status_code == 404:
                    return None
                r.raise_for_status()
                return r.content
            except requests.RequestException as exc:
                log.warning(f"GKG fetch {ts:%Y%m%d%H%M} attempt {attempt + 1} failed: {exc}")
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"GKG fetch failed after retries: {url}")

    def _process(self, ts: datetime) -> None:
        if self._stop.is_set():
            return
        if self._free_gb() < float(self.cfg["min_free_gb"]):
            log.error(f"Free disk below {self.cfg['min_free_gb']} GB; pausing GKG stream.")
            self._stop.set()
            return
        key = ts.strftime("%Y%m%d%H%M%S")
        try:
            raw = self._fetch(ts)
        except RuntimeError as exc:
            log.error(str(exc))
            with self._lock:
                self.stats.files_failed += 1
            return
        if raw is None:
            with self._lock:
                self.stats.files_missing += 1
                self.manifest.open("a").write(key + "\n")
            return
        df = parse_gkg(raw)
        del raw
        rows = filter_records(df, self.linker, self.cfg["mkt_theme_prefixes"])
        with self._lock:
            path = self.out_dir / f"{ts:%Y-%m}.jsonl.gz"
            with gzip.open(path, "at", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            self.manifest.open("a").write(key + "\n")
            self.stats.files_ok += 1
            self.stats.records_raw += df.attrs.get("n_raw", len(df))
            self.stats.records_seen += len(df)
            self.stats.records_kept += len(rows)

    def run(self, timestamps: Iterable[datetime], workers: int | None = None) -> StreamStats:
        """Process timestamps not yet in the manifest. Returns aggregate stats."""
        done = self.done()
        todo = [t for t in timestamps if t.strftime("%Y%m%d%H%M%S") not in done]
        log.info(f"GKG stream: {len(todo)} files to process ({len(done)} already done)")
        n = workers or int(self.cfg["workers"])
        with ThreadPoolExecutor(max_workers=n) as pool:
            for i, _ in enumerate(pool.map(self._process, todo), 1):
                if i % 50 == 0:
                    s = self.stats
                    log.info(
                        f"GKG {i}/{len(todo)} files | kept {s.records_kept}/{s.records_seen} "
                        f"| free {self._free_gb():.1f} GB"
                    )
        if self._stop.is_set():
            log.error("GKG stream stopped early (disk guard).")
        return self.stats


def read_gkg_rows(out_dir: Path | None = None) -> Iterator[dict]:
    """Iterate all kept GKG rows from the processed store."""
    d = out_dir or repo_root() / load_config("app")["gdelt_gkg"]["out_dir"]
    for path in sorted(d.glob("*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                yield json.loads(line)


def relink(row: dict, linker: EntityLinker, theme_prefixes: list[str]) -> dict | None:
    """Re-apply (possibly stricter) linking rules to a stored row; None if it no longer qualifies.

    Stored rows are a superset of what the current rules keep, so tightening the linker never
    requires re-downloading GKG files.
    """
    title = row["title"]
    low_value = linker.is_low_value(title)
    tickers = (
        [] if low_value else [m.ticker for m in linker.link("", title=title) if m.ticker != MKT]
    )
    org_only = (
        []
        if tickers or low_value
        else [m.ticker for m in linker.link("", orgs=row.get("orgs", [])) if m.ticker != MKT]
    )
    is_mkt = False
    if not tickers and not org_only:
        is_mkt = linker.is_market_wide(title) and any(
            t.startswith(tuple(theme_prefixes)) for t in row.get("themes", [])
        )
        if not is_mkt:
            return None
    return {**row, "tickers": tickers, "org_only_tickers": org_only, "is_mkt": is_mkt}
