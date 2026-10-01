"""SignalSubscriber: how Module A and Module B "subscribe" to engine output.

Two interchangeable implementations:
- ``SSESubscriber``: connects to the API's ``GET /stream`` (Server-Sent Events).
- ``StorePollingSubscriber``: tails ``data/signals/signals.jsonl`` (offline / no API).
Both yield :class:`Signal` objects filtered by type and minimum impact.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path

import httpx

from riskpulse.common.schemas import Signal
from riskpulse.store.signal_store import default_paths


class SignalSubscriber(ABC):
    def __init__(self, signal_type: str | None = None, min_impact: int = 1) -> None:
        self.signal_type = signal_type
        self.min_impact = min_impact

    def accepts(self, s: Signal) -> bool:
        if self.signal_type and s.signal_type != self.signal_type:
            return False
        return s.impact_score >= self.min_impact

    @abstractmethod
    def __iter__(self) -> Iterator[Signal]: ...


class SSESubscriber(SignalSubscriber):
    """Subscribe to the API's SSE stream."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000", **kw) -> None:
        super().__init__(**kw)
        self.base_url = base_url.rstrip("/")

    def __iter__(self) -> Iterator[Signal]:
        params = {"min_impact": self.min_impact}
        if self.signal_type:
            params["signal_type"] = self.signal_type
        with httpx.stream("GET", f"{self.base_url}/stream", params=params, timeout=None) as resp:
            event = None
            for line in resp.iter_lines():
                if line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:") and event == "signal":
                    s = Signal.model_validate(json.loads(line[5:]))
                    if self.accepts(s):
                        yield s


class StorePollingSubscriber(SignalSubscriber):
    """Tail the JSONL signal file. ``follow=False`` reads what exists and stops."""

    def __init__(
        self, path: Path | None = None, follow: bool = False, poll_seconds: float = 0.5, **kw
    ) -> None:
        super().__init__(**kw)
        self.path = path or default_paths()[0]
        self.follow = follow
        self.poll = poll_seconds

    def __iter__(self) -> Iterator[Signal]:
        pos = 0
        while True:
            if self.path.exists():
                with self.path.open(encoding="utf-8") as fh:
                    fh.seek(pos)
                    while True:
                        line = fh.readline()
                        if not line or not line.endswith("\n"):
                            break  # wait for a complete line
                        pos = fh.tell()
                        s = Signal.model_validate_json(line)
                        if self.accepts(s):
                            yield s
            if not self.follow:
                return
            time.sleep(self.poll)
