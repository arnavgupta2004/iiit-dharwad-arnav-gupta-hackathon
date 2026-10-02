"""FastAPI service (spec §5.8): REST queries, SSE subscription, /analyze and demo /inject.

Modes:
- ``fast``: serves precomputed signals only (no models loaded); /analyze and /inject return 503.
- ``full``: loads the models; /analyze scores text instantly, /inject pushes a labelled synthetic
  headline through the live engine, and an optional paced replay publishes new signals.
"""

from __future__ import annotations

import asyncio
import json
import threading
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from riskpulse import __version__
from riskpulse.api.hub import SignalHub
from riskpulse.common.config import load_config
from riskpulse.common.logging import get_logger
from riskpulse.common.schemas import Signal
from riskpulse.store.signal_store import default_paths, serving_signals_path

log = get_logger()


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=3, max_length=2000)
    source: str = "synthetic_demo"


class InjectRequest(BaseModel):
    text: str = Field(min_length=3, max_length=500)
    outlets: int = Field(default=3, ge=1, le=10, description="Simulated distinct outlets.")


def create_app(
    mode: str = "fast",
    hub: SignalHub | None = None,
    live=None,
    replay: tuple[str, str, float] | None = None,
) -> FastAPI:
    """Build the app. ``live`` is a LiveEngine (full mode); ``replay`` = (start, end, sec/day)."""
    hub = hub or SignalHub(default_paths()[0], load_existing=True, load_from=serving_signals_path())
    state: dict = {
        "mode": mode,
        "live": live,
        "replay_status": "idle",
        "started": datetime.now(UTC),
    }

    def _replay_worker(start: str, end: str, spd: float) -> None:
        from riskpulse.ingestion.replay import replay as paced

        state["replay_status"] = f"running {start} -> {end}"
        batch, last_flush = [], None
        for doc in paced(start, end, spd):
            batch.append(doc)
            if last_flush is None:
                last_flush = doc.published_at
            if len(batch) >= 32 or (doc.published_at - last_flush).total_seconds() > 3600:
                _, sigs = state["live"].process(batch)
                hub.publish(sigs)
                batch, last_flush = [], None
        if batch:
            _, sigs = state["live"].process(batch)
            hub.publish(sigs)
        state["replay_status"] = "finished"

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if live is not None and hasattr(live, "warm_up"):
            live.warm_up()
        if replay and live is not None:
            threading.Thread(target=_replay_worker, args=replay, daemon=True).start()
        yield

    app = FastAPI(title="RiskPulse Signal API", version=__version__, lifespan=lifespan)
    app.state.hub = hub

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "mode": state["mode"],
            "n_signals": len(hub.signals),
            "replay": state["replay_status"],
            "version": __version__,
        }

    @app.get("/signals", response_model=list[Signal])
    def signals(
        ticker: str | None = None,
        event_class: str | None = None,
        min_impact: int | None = Query(None, ge=1, le=10),
        since: str | None = None,
        signal_type: str | None = Query(None, pattern="^(entity|event)$"),
        limit: int = Query(100, ge=1, le=5000),
    ) -> list[Signal]:
        return hub.query(
            ticker=ticker,
            event_class=event_class,
            min_impact=min_impact,
            since=since,
            signal_type=signal_type,
            limit=limit,
        )

    @app.get("/signals/latest", response_model=list[Signal])
    def latest() -> list[Signal]:
        """Latest entity signal per ticker (the Module A view)."""
        return hub.latest_per_entity()

    @app.get("/events", response_model=list[Signal])
    def events(min_impact: int = Query(1, ge=1, le=10), limit: int = Query(200, ge=1, le=5000)):
        return hub.query(signal_type="event", min_impact=min_impact, limit=limit)

    @app.get("/entities")
    def entities() -> list[dict]:
        u = load_config("universe")
        out = [
            {"ticker": t, "name": s["name"], "sector": s["sector"]} for t, s in u["tickers"].items()
        ]
        out.append({"ticker": "MKT", "name": "Market-wide", "sector": None})
        return out

    @app.get("/stream")
    async def stream(
        request: Request,
        signal_type: str | None = Query(None, pattern="^(entity|event)$"),
        min_impact: int = Query(1, ge=1, le=10),
    ):
        """Server-Sent Events: each new signal is one `signal` event with JSON data."""
        q = hub.subscribe()

        async def gen():
            try:
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        s: Signal = await asyncio.wait_for(q.get(), timeout=15)
                    except TimeoutError:
                        yield {"event": "ping", "data": datetime.now(UTC).isoformat()}
                        continue
                    if signal_type and s.signal_type != signal_type:
                        continue
                    if s.impact_score < min_impact:
                        continue
                    yield {"event": "signal", "id": s.signal_id, "data": s.model_dump_json()}
            finally:
                hub.unsubscribe(q)

        return EventSourceResponse(gen())

    @app.post("/analyze")
    def analyze(req: AnalyzeRequest) -> dict:
        if state["live"] is None:
            raise HTTPException(
                503, "Models not loaded (fast mode). Start with: riskpulse serve --full"
            )
        return state["live"].analyze(req.text, req.source)

    @app.post("/inject")
    def inject(req: InjectRequest) -> dict:
        """Demo only: push a labelled synthetic headline from N simulated outlets."""
        if state["live"] is None:
            raise HTTPException(
                503, "Models not loaded (fast mode). Start with: riskpulse serve --full"
            )
        now = datetime.now(UTC)
        out: list[Signal] = []
        for i in range(req.outlets):
            out += state["live"].inject(req.text, published_at=now, outlet=f"outlet{i + 1}")
        hub.publish(out)
        return {
            "label": "SYNTHETIC DEMO HEADLINE - not real news",
            "source": "synthetic_demo",
            "signals": [json.loads(s.model_dump_json()) for s in out],
        }

    return app
