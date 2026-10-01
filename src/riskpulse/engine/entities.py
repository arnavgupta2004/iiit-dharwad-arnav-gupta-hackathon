"""Entity linking: map text to universe tickers, or to the market-wide pseudo-entity MKT.

Rules (spec §5.2):
- Hits come from cashtags ($AAPL), company aliases, brands, executives (half weight) and a
  whitelist of bare uppercase tickers.
- Ambiguous aliases (Apple, Amazon, Chase, Meta, Tesla, ...) only count when the text has
  financial/tech context words, a cashtag, or another unambiguous hit for the same ticker.
- Relevance in [0, 1] combines hit strength (title hits x2) and the ticker's share of all hits.
- Items with no company but macro / geopolitical / credit keywords map to MKT with region tags.
- Optional extra evidence: an organisation list (e.g. GDELT GKG V2Organizations).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import cached_property

from riskpulse.common.config import load_config

MKT = "MKT"
_NON_WORD = re.compile(r"[^0-9a-z&$.+'\-]+")


def normalise(text: str) -> str:
    """Lowercase, collapse punctuation to spaces, pad with spaces for phrase matching."""
    return f" {_NON_WORD.sub(' ', text.lower()).strip()} "


def _phrase_regex(phrases: list[str], case_sensitive: bool = False) -> re.Pattern | None:
    """Regex matching any phrase as a whole word (hyphens count as word characters)."""
    phrases = sorted({p for p in phrases if p}, key=len, reverse=True)
    if not phrases:
        return None
    body = "|".join(re.escape(p) for p in phrases)
    flags = 0 if case_sensitive else re.IGNORECASE
    return re.compile(rf"(?<![\w\-$]){'(?:' + body + ')'}(?![\w\-])", flags)


@dataclass(frozen=True)
class EntityMention:
    """One linked entity in a document."""

    ticker: str
    relevance: float
    in_title: bool
    matched: tuple[str, ...]
    regions: tuple[str, ...] = ()


@dataclass
class _TickerPatterns:
    ticker: str
    strong: re.Pattern | None  # unambiguous aliases + brands (case-insensitive)
    ambiguous: re.Pattern | None  # ambiguous aliases (case-sensitive, capitalised form)
    execs: re.Pattern | None
    cashtags: re.Pattern | None
    org_aliases: set[str] = field(default_factory=set)


@dataclass
class _Hits:
    score: float = 0.0
    title_score: float = 0.0
    terms: list[str] = field(default_factory=list)
    ambiguous_only: bool = True


class EntityLinker:
    """Links documents to universe tickers (or MKT) using config-driven alias rules."""

    def __init__(self, universe: dict | None = None, taxonomy: dict | None = None) -> None:
        self.universe = universe or load_config("universe")
        self.taxonomy = taxonomy or load_config("taxonomy")
        self.cfg = self.universe["linking"]
        self.tickers: dict[str, dict] = self.universe["tickers"]

    # ---------- pattern construction ----------
    @cached_property
    def _patterns(self) -> list[_TickerPatterns]:
        out = []
        for t, spec in self.tickers.items():
            amb = set(spec.get("ambiguous", []))
            names = spec["aliases"] + spec.get("brands", [])
            strong = [n for n in names if n not in amb]
            orgs = {normalise(n).strip() for n in spec["aliases"]}
            orgs |= {f"{o} inc" for o in orgs} | {normalise(spec["name"]).strip()}
            out.append(
                _TickerPatterns(
                    ticker=t,
                    strong=_phrase_regex(strong),
                    ambiguous=_phrase_regex(sorted(amb), case_sensitive=True),
                    execs=_phrase_regex(spec.get("execs", [])),
                    cashtags=_phrase_regex(spec["cashtags"]),
                    org_aliases=orgs,
                )
            )
        return out

    @cached_property
    def _bare(self) -> re.Pattern | None:
        return _phrase_regex(self.cfg.get("bare_tickers", []), case_sensitive=True)

    @cached_property
    def _context(self) -> re.Pattern | None:
        return _phrase_regex(self.cfg.get("context_words", []))

    @cached_property
    def _mkt_keywords(self) -> re.Pattern | None:
        words: list[str] = []
        for cls in self.cfg.get("mkt_classes", []):
            words += self.taxonomy["classes"][cls]["keywords"]
        return _phrase_regex(words)

    @cached_property
    def _regions(self) -> dict[str, list[str]]:
        return self.cfg.get("regions", {})

    # ---------- matching ----------
    def _hits_in(
        self, text: str, weight: float, hits: dict[str, _Hits], is_title: bool, has_context: bool
    ) -> None:
        if not text:
            return
        for p in self._patterns:
            h = hits.setdefault(p.ticker, _Hits())
            found: list[tuple[str, float, bool]] = []  # (term, score, ambiguous)
            for pat, w, amb in (
                (p.cashtags, 1.0, False),
                (p.strong, 1.0, False),
                (p.execs, float(self.cfg.get("exec_weight", 0.5)), False),
                (p.ambiguous, 1.0, True),
            ):
                if pat is None:
                    continue
                for m in pat.finditer(text):
                    found.append((m.group(0), w, amb))
            unambiguous_here = any(not a for _, _, a in found)
            for term, w, amb in found:
                if amb and not (has_context or unambiguous_here or not h.ambiguous_only):
                    continue
                h.score += w * weight
                if is_title:
                    h.title_score += w * weight
                h.terms.append(term)
                if not amb:
                    h.ambiguous_only = False
        if self._bare:
            mapping = self.cfg.get("bare_ticker_map", {})
            for m in self._bare.finditer(text):
                t = mapping.get(m.group(0), m.group(0))
                h = hits.setdefault(t, _Hits())
                h.score += weight
                if is_title:
                    h.title_score += weight
                h.terms.append(m.group(0))
                h.ambiguous_only = False

    def regions_of(self, text: str) -> tuple[str, ...]:
        """Region tags detected from keyword lists (config `linking.regions`)."""
        norm = normalise(text)
        return tuple(r for r, kws in self._regions.items() if any(k in norm for k in kws))

    def is_market_wide(self, text: str) -> bool:
        """True if the text carries macro / geopolitical / credit keywords."""
        return bool(self._mkt_keywords and self._mkt_keywords.search(text))

    def link(
        self,
        text: str,
        title: str | None = None,
        orgs: list[str] | None = None,
        prior_ticker: str | None = None,
    ) -> list[EntityMention]:
        """Return linked entities sorted by relevance (highest first).

        Args:
            text: body text (or the tweet).
            title: headline; hits here count ``title_weight`` times.
            orgs: optional organisation names (lowercase), e.g. GDELT V2Organizations.
            prior_ticker: ticker the source already attributes the item to (tweet dataset label).
        """
        hits: dict[str, _Hits] = {}
        title_w = float(self.cfg.get("title_weight", 2.0))
        full_text = f"{title or ''} {text or ''}"
        ctx = bool(self._context and self._context.search(full_text))
        self._hits_in(title or "", title_w, hits, is_title=True, has_context=ctx)
        self._hits_in(text or "", 1.0, hits, is_title=False, has_context=ctx)
        if orgs:
            org_set = {normalise(o).strip() for o in orgs}
            for p in self._patterns:
                matched = sorted(org_set & p.org_aliases)
                if matched:
                    h = hits.setdefault(p.ticker, _Hits())
                    h.score += 0.5 * len(matched)
                    h.terms += [f"org:{m}" for m in matched]
                    h.ambiguous_only = False
        if prior_ticker and prior_ticker in self.tickers:
            h = hits.setdefault(prior_ticker, _Hits())
            h.score += 1.0
            h.terms.append(f"prior:{prior_ticker}")
            h.ambiguous_only = False

        live = {t: h for t, h in hits.items() if h.score > 0}
        regions = self.regions_of(full_text)
        if not live:
            if self.is_market_wide(full_text):
                return [EntityMention(MKT, 1.0, False, ("mkt",), regions)]
            return []
        total = sum(h.score for h in live.values())
        out = []
        for t, h in live.items():
            strength = min(1.0, h.score / 3.0)
            share = h.score / total
            relevance = round(strength * (0.5 + 0.5 * share), 4)
            out.append(EntityMention(t, relevance, h.title_score > 0, tuple(h.terms), regions))
        return sorted(out, key=lambda m: m.relevance, reverse=True)
