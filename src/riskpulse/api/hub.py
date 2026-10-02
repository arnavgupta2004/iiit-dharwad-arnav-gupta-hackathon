"""In-process signal hub: query index + fan-out to SSE subscribers + JSONL persistence."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pandas as pd

from riskpulse.common.schemas import Signal
from riskpulse.store.signal_store import (
    JsonlSignalWriter,
    query_signals,
    read_signals_jsonl,
    signals_frame,
)


class SignalHub:
    """Thread-safe store of signals with async broadcast.

    Producers (replay thread, /inject) call :meth:`publish` from any thread; each SSE
    subscriber owns an asyncio.Queue fed via ``loop.call_soon_threadsafe``.
    """

    def __init__(
        self,
        jsonl_path: Path | None = None,
        load_existing: bool = True,
        load_from: Path | None = None,
    ) -> None:
        """``jsonl_path``: where new signals are appended; ``load_from``: initial signals
        (defaults to ``jsonl_path``; may be the gzipped demo snapshot)."""
        self._lock = threading.Lock()
        self.writer = JsonlSignalWriter(jsonl_path) if jsonl_path else None
        src = load_from or jsonl_path
        self.signals: list[Signal] = read_signals_jsonl(src) if load_existing and src else []
        self._frame: pd.DataFrame | None = None
        self._subs: list[tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = []

    def publish(self, signals: list[Signal], persist: bool = True) -> None:
        if not signals:
            return
        with self._lock:
            self.signals.extend(signals)
            self._frame = None
            subs = list(self._subs)
        if persist and self.writer:
            self.writer.write(signals)
        for loop, q in subs:
            for s in signals:
                loop.call_soon_threadsafe(q.put_nowait, s)

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=10_000)
        with self._lock:
            self._subs.append((asyncio.get_running_loop(), q))
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        with self._lock:
            self._subs = [(lp, x) for lp, x in self._subs if x is not q]

    def frame(self) -> pd.DataFrame:
        with self._lock:
            if self._frame is None:
                self._frame = signals_frame(self.signals)
            return self._frame

    def query(self, **kw) -> list[Signal]:
        df = self.frame()
        if df.empty:
            return []
        ids = set(query_signals(df, **kw)["signal_id"])
        by_id = {s.signal_id: s for s in self.signals if s.signal_id in ids}
        return sorted(by_id.values(), key=lambda s: s.as_of, reverse=True)

    def latest_per_entity(self) -> list[Signal]:
        latest: dict[str, Signal] = {}
        with self._lock:
            for s in self.signals:
                if s.signal_type == "entity" and s.entity:
                    cur = latest.get(s.entity.ticker)
                    if cur is None or s.as_of >= cur.as_of:
                        latest[s.entity.ticker] = s
        return sorted(latest.values(), key=lambda s: s.entity.ticker)
