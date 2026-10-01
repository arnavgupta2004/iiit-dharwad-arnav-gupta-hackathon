"""Signal aggregation (spec §5.6-5.7): per-document scores -> entity and event signals.

Entity signal (Module A): per ticker, exponentially decayed weighted mean of entity sentiment
    s_i(t) = sum_k w_k * 2^(-(t - t_k)/H) * s_k / sum_k w_k * 2^(-(t - t_k)/H)
    with w_k = relevance_k * event_confidence_k; confidence = 1 - exp(-W/w0), W the decayed weight.
Event signal (Module B): per story, confidence-weighted class vote, impact = max doc impact +
    breadth bonus, distinct outlets, regions, sectors and the strongest evidence.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime

from riskpulse.common.config import load_config
from riskpulse.common.schemas import EntityRef, Evidence, Signal
from riskpulse.engine.sentiment import label_from_score


@dataclass
class ScoredMention:
    """One (document, entity) pair after the NLP stages."""

    doc_id: str
    source: str
    outlet: str
    published_at: datetime
    title: str | None
    url: str | None
    ticker: str
    relevance: float
    sentiment: float
    event_class: str
    event_confidence: float
    event_id: str
    impact_score: int
    impact_raw: float
    drivers: dict[str, float]
    regions: list[str] = field(default_factory=list)


def _sig_id(*parts: str) -> str:
    return "sig_" + hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


class EntityAggregator:
    """Decayed sentiment state per ticker; O(1) update per mention."""

    def __init__(self, cfg: dict | None = None) -> None:
        c = cfg or load_config("app")["aggregation"]
        self.h = float(c["entity_half_life_hours"])
        self.w0 = float(c["entity_confidence_w0"])
        self.state: dict[str, tuple[float, float, datetime]] = {}  # num, den, last_ts
        self.recent: dict[str, list[ScoredMention]] = defaultdict(list)
        self.universe = load_config("universe")["tickers"]

    def _decay(self, ticker: str, ts: datetime) -> tuple[float, float]:
        if ticker not in self.state:
            return 0.0, 0.0
        num, den, last = self.state[ticker]
        f = 2 ** (-(ts - last).total_seconds() / 3600 / self.h)
        return num * f, den * f

    def update(self, m: ScoredMention) -> None:
        num, den = self._decay(m.ticker, m.published_at)
        w = max(m.relevance, 1e-3) * max(m.event_confidence, 0.1)
        self.state[m.ticker] = (num + w * m.sentiment, den + w, m.published_at)
        r = self.recent[m.ticker]
        r.append(m)
        if len(r) > 50:
            del r[:-50]

    def signal(self, ticker: str, as_of: datetime) -> Signal | None:
        """Entity signal for a ticker at ``as_of`` (None if never mentioned)."""
        if ticker not in self.state:
            return None
        num, den = self._decay(ticker, as_of)
        if den <= 0:
            return None
        s = max(-1.0, min(1.0, num / den))
        conf = 1 - math.exp(-den / self.w0)
        recent = [m for m in self.recent[ticker] if m.published_at <= as_of]
        top = max(recent, key=lambda m: m.impact_raw) if recent else None
        classes = Counter(m.event_class for m in recent)
        spec = self.universe.get(ticker, {})
        ev = sorted(recent, key=lambda m: m.impact_raw, reverse=True)[:5]
        return Signal(
            signal_id=_sig_id("entity", ticker, as_of.isoformat()),
            signal_type="entity",
            as_of=as_of,
            entity=EntityRef(
                ticker=ticker, name=spec.get("name", ticker), sector=spec.get("sector")
            ),
            event_id=top.event_id if top else None,
            sentiment_score=round(s, 4),
            sentiment_label=label_from_score(s),
            event_class=classes.most_common(1)[0][0] if classes else "OTHER",
            event_confidence=round(top.event_confidence, 4) if top else 0.0,
            impact_score=top.impact_score if top else 1,
            impact_raw=round(top.impact_raw, 6) if top else 0.0,
            confidence=round(conf, 4),
            n_docs=max(1, len(recent)),
            n_sources=max(1, len({m.outlet for m in recent})),
            sectors=[spec["sector"]] if spec.get("sector") else [],
            drivers=top.drivers if top else {},
            explanation=f"EWMA sentiment {s:+.2f} (half-life {self.h:g} h) over "
            f"{len(recent)} recent mentions; evidence weight {den:.2f}",
            evidence=[
                Evidence(
                    doc_id=m.doc_id,
                    source=m.source,
                    title=m.title,
                    url=m.url,
                    published_at=m.published_at,
                )
                for m in ev
            ],
        )


@dataclass
class _Story:
    mentions: list[ScoredMention] = field(default_factory=list)
    outlets: set[str] = field(default_factory=set)
    emitted_impact: int = 0


class EventAggregator:
    """Per-story aggregation into event signals."""

    def __init__(self, cfg: dict | None = None) -> None:
        c = cfg or load_config("app")["aggregation"]
        self.bonus = sorted(c["event_breadth_bonus"], key=lambda b: b["min_sources"])
        self.min_docs = int(c["event_emit_min_docs"])
        self.max_ev = int(c["evidence_max"])
        self.stories: dict[str, _Story] = defaultdict(_Story)
        self.universe = load_config("universe")["tickers"]

    def update(self, m: ScoredMention) -> Signal | None:
        """Add a mention; return a (refreshed) event signal when the story's impact changes."""
        st = self.stories[m.event_id]
        st.mentions.append(m)
        st.outlets.add(m.outlet)
        if len(st.mentions) < self.min_docs:
            return None
        sig = self.signal(m.event_id, m.published_at)
        if sig.impact_score != st.emitted_impact:
            st.emitted_impact = sig.impact_score
            return sig
        return None

    def signal(self, event_id: str, as_of: datetime) -> Signal:
        st = self.stories[event_id]
        ms = st.mentions
        votes: Counter = Counter()
        for m in ms:
            votes[m.event_class] += m.event_confidence
        cls, wsum = votes.most_common(1)[0]
        conf = wsum / max(sum(votes.values()), 1e-9)
        n_src = len(st.outlets)
        bonus = max([b["bonus"] for b in self.bonus if n_src >= b["min_sources"]], default=0)
        top = max(ms, key=lambda m: m.impact_raw)
        impact = min(10, top.impact_score + bonus)
        sent = sum(m.sentiment for m in ms) / len(ms)
        tickers = sorted({m.ticker for m in ms})
        sectors = sorted({self.universe[t]["sector"] for t in tickers if t in self.universe})
        regions = sorted({r for m in ms for r in m.regions})
        ev = sorted(ms, key=lambda m: m.impact_raw, reverse=True)[: self.max_ev]
        entity = None
        if len(tickers) == 1:
            t = tickers[0]
            spec = self.universe.get(t, {})
            entity = EntityRef(
                ticker=t, name=spec.get("name", "Market-wide"), sector=spec.get("sector")
            )
        return Signal(
            signal_id=_sig_id("event", event_id, str(len(ms))),
            signal_type="event",
            as_of=as_of,
            entity=entity,
            event_id=event_id,
            sentiment_score=round(max(-1.0, min(1.0, sent)), 4),
            sentiment_label=label_from_score(sent),
            event_class=cls,
            event_confidence=round(
                conf * max(m.event_confidence for m in ms if m.event_class == cls), 4
            ),
            impact_score=impact,
            impact_raw=round(top.impact_raw, 6),
            confidence=round(conf * (1 - math.exp(-len(ms) / 3)), 4),
            n_docs=len(ms),
            n_sources=n_src,
            regions=regions,
            sectors=sectors,
            drivers=top.drivers,
            explanation=(
                f"{cls.replace('_', ' ').title()}; {label_from_score(sent)} tone ({sent:+.2f}); "
                f"{len(ms)} items from {n_src} outlets; top item impact {top.impact_score}"
                + (f" + breadth bonus {bonus}" if bonus else "")
            ),
            evidence=[
                Evidence(
                    doc_id=m.doc_id,
                    source=m.source,
                    title=m.title,
                    url=m.url,
                    published_at=m.published_at,
                )
                for m in ev
            ],
        )
