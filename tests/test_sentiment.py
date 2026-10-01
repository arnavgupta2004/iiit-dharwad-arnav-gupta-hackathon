import pytest

from riskpulse.engine.sentiment import (
    LoughranMcDonaldScorer,
    VaderScorer,
    entity_sentiment,
    label_from_score,
    split_clauses,
)
from tests.conftest import requires_finbert


def test_label_thresholds() -> None:
    assert label_from_score(0.5) == "positive"
    assert label_from_score(-0.5) == "negative"
    assert label_from_score(0.1) == "neutral"
    assert label_from_score(0.15) == "neutral"  # strict inequality


def test_split_clauses_on_contrast_and_sentences() -> None:
    assert split_clauses("JPM beats estimates while BAC misses") == [
        "JPM beats estimates",
        "BAC misses",
    ]
    assert len(split_clauses("Apple rises. Tesla falls; Ford flat")) == 3


def test_lm_baseline_polarity() -> None:
    lm = LoughranMcDonaldScorer()
    neg, pos, zero = lm.score(
        ["The company reported a loss amid litigation", "Strong gains", "The meeting is on Tuesday"]
    )
    assert neg < 0 < pos and zero == 0


def test_vader_baseline_range() -> None:
    v = VaderScorer().score(["great results!", "terrible collapse"])
    assert v[0] > 0 > v[1] and all(-1 <= x <= 1 for x in v)


@requires_finbert
def test_finbert_doc_scores_have_expected_signs(finbert) -> None:
    pos, neg = finbert.score(
        [
            "JPMorgan beats earnings estimates and raises guidance",
            "Bank misses as loan losses surge",
        ]
    )
    assert pos.score > 0.15 and pos.label == "positive"
    assert neg.score < -0.15 and neg.label == "negative"
    assert pytest.approx(pos.p_pos + pos.p_neg + pos.p_neu, abs=1e-5) == 1.0


@requires_finbert
def test_entity_level_sentiment_gives_opposite_signs(finbert, linker) -> None:
    text = "JPMorgan beats earnings estimates while Bank of America misses on revenue"
    s = entity_sentiment(text, ["JPM", "BAC"], finbert, linker)
    assert s["JPM"] > 0 > s["BAC"]


@requires_finbert
def test_entity_not_in_any_clause_falls_back_to_doc_score(finbert, linker) -> None:
    text = "Shares surged after record deliveries. Analysts cheered the results."
    s = entity_sentiment(text, ["TSLA"], finbert, linker, doc_score=0.42)
    assert s["TSLA"] == 0.42
