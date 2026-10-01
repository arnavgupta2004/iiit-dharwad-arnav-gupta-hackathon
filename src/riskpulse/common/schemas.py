"""Pydantic data contracts: normalised documents (spec §5.1) and signals (spec §5.7)."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

SCHEMA_VERSION = 1


class Source(StrEnum):
    GDELT = "gdelt"
    NEWSAPI = "newsapi"
    KAGGLE_NEWS = "kaggle_news"
    KAGGLE_TWEETS = "kaggle_tweets"
    REDDIT = "reddit"
    SYNTHETIC_DEMO = "synthetic_demo"


SOURCE_TYPE: dict[Source, Literal["news", "social"]] = {
    Source.GDELT: "news",
    Source.NEWSAPI: "news",
    Source.KAGGLE_NEWS: "news",
    Source.KAGGLE_TWEETS: "social",
    Source.REDDIT: "social",
    Source.SYNTHETIC_DEMO: "news",
}


class EventClass(StrEnum):
    GEOPOLITICAL = "GEOPOLITICAL"
    MACROECONOMIC = "MACROECONOMIC"
    CREDIT_EVENT = "CREDIT_EVENT"
    MERGER_ACQUISITION = "MERGER_ACQUISITION"
    PRODUCT_LAUNCH = "PRODUCT_LAUNCH"
    EARNINGS = "EARNINGS"
    REGULATORY_LEGAL = "REGULATORY_LEGAL"
    MANAGEMENT_CHANGE = "MANAGEMENT_CHANGE"
    OPERATIONAL_ESG = "OPERATIONAL_ESG"
    OTHER = "OTHER"


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def make_doc_id(source: str, url: str | None, text: str) -> str:
    """Stable document id: sha1 of source + url (or text when no url)."""
    key = f"{source}|{url or text}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


class Document(BaseModel):
    """A normalised text item from any source."""

    doc_id: str
    source: Source
    source_type: Literal["news", "social"]
    published_at: datetime
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    title: str | None = None
    text: str
    url: str | None = None
    author: str | None = None
    lang: str = "en"
    meta: dict[str, Any] = Field(default_factory=dict)

    @field_validator("published_at", "ingested_at")
    @classmethod
    def _to_utc(cls, v: datetime) -> datetime:
        return _utc(v)

    @classmethod
    def build(
        cls,
        source: Source | str,
        text: str,
        published_at: datetime,
        *,
        title: str | None = None,
        url: str | None = None,
        author: str | None = None,
        lang: str = "en",
        meta: dict[str, Any] | None = None,
    ) -> Document:
        """Construct a document, deriving ``doc_id`` and ``source_type``."""
        src = Source(source)
        return cls(
            doc_id=make_doc_id(src.value, url, text),
            source=src,
            source_type=SOURCE_TYPE[src],
            published_at=published_at,
            title=title,
            text=text,
            url=url,
            author=author,
            lang=lang,
            meta=meta or {},
        )


class EntityRef(BaseModel):
    ticker: str
    name: str
    sector: str | None = None


class Evidence(BaseModel):
    doc_id: str
    source: str
    title: str | None = None
    url: str | None = None
    published_at: datetime


class Signal(BaseModel):
    """Structured engine output consumed by Module A and Module B (schema_version 1)."""

    schema_version: int = SCHEMA_VERSION
    signal_id: str
    signal_type: Literal["entity", "event"]
    as_of: datetime
    entity: EntityRef | None = None
    event_id: str | None = None
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    sentiment_label: Literal["positive", "negative", "neutral"]
    event_class: EventClass
    event_confidence: float = Field(ge=0.0, le=1.0)
    impact_score: int = Field(ge=1, le=10)
    impact_raw: float
    confidence: float = Field(ge=0.0, le=1.0)
    n_docs: int = Field(ge=1)
    n_sources: int = Field(ge=1)
    regions: list[str] = Field(default_factory=list)
    sectors: list[str] = Field(default_factory=list)
    drivers: dict[str, float] = Field(default_factory=dict)
    explanation: str = ""
    evidence: list[Evidence] = Field(default_factory=list)
    model_versions: dict[str, str] = Field(default_factory=dict)

    @field_validator("as_of")
    @classmethod
    def _to_utc(cls, v: datetime) -> datetime:
        return _utc(v)
