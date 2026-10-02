"""Without a trained event model (fresh clone), the scorer falls back to the keyword baseline."""

from datetime import UTC, datetime

from riskpulse.common.schemas import Document, Source
from riskpulse.engine import pipeline
from riskpulse.engine.events import KeywordClassifier


class _NoEmbed:
    def encode(self, texts):
        import numpy as np

        return np.zeros((len(texts), 4), dtype=np.float32)


class _FixedSentiment:
    version = "stub"

    def score(self, texts):
        from riskpulse.engine.sentiment import SentimentResult

        return [SentimentResult(0.0, "neutral", 0.0, 0.0, 1.0) for _ in texts]


def test_missing_event_model_uses_keywords(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(pipeline, "repo_root", lambda: tmp_path)  # no model file there
    scorer = pipeline.NLPScorer(finbert=_FixedSentiment(), embedder=_NoEmbed())
    assert isinstance(scorer.event_clf, KeywordClassifier)
    doc = Document.build(
        Source.SYNTHETIC_DEMO,
        "Bank files for bankruptcy after missed bond payment",
        datetime(2022, 3, 1, tzinfo=UTC),
        title="Bank files for bankruptcy after missed bond payment",
    )
    (ds,) = scorer.score([doc])
    assert ds.event.event_class == "CREDIT_EVENT"
