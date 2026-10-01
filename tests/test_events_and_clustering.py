from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from riskpulse.engine.clustering import StoryClusterer
from riskpulse.engine.events import CLASSES, KeywordClassifier

T0 = datetime(2022, 2, 24, 6, tzinfo=UTC)


def unit(v: list[float]) -> np.ndarray:
    a = np.asarray(v, dtype=float)
    return a / np.linalg.norm(a)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Russia invades Ukraine as sanctions loom", "GEOPOLITICAL"),
        ("Fed raises rates by 75 basis points", "MACROECONOMIC"),
        ("Developer misses coupon payment and is downgraded", "CREDIT_EVENT"),
        ("Microsoft to acquire Activision in $69bn deal", "MERGER_ACQUISITION"),
        ("SEC fines bank over disclosures", "REGULATORY_LEGAL"),
        ("Stocks drift higher in quiet trade", "OTHER"),
    ],
)
def test_keyword_baseline(text: str, expected: str) -> None:
    assert KeywordClassifier().predict([text])[0].event_class == expected


def test_keyword_priority_breaks_ties_toward_credit() -> None:
    # one GEO hit ("sanctions") and one CREDIT hit ("default") -> CREDIT by priority
    p = KeywordClassifier().predict(["Sanctions push sovereign toward default"])[0]
    assert p.event_class == "CREDIT_EVENT"


def test_all_taxonomy_classes_known() -> None:
    assert "OTHER" in CLASSES and len(CLASSES) == 10


def test_clusterer_groups_similar_and_splits_dissimilar() -> None:
    c = StoryClusterer(threshold=0.8, window_hours=48)
    e1, _, new1 = c.assign("a" * 40, unit([1, 0.05, 0]), T0)
    e2, sim, new2 = c.assign("b" * 40, unit([1, 0.1, 0]), T0 + timedelta(hours=1))
    e3, _, new3 = c.assign("c" * 40, unit([0, 1, 0]), T0 + timedelta(hours=2))
    assert new1 and not new2 and new3
    assert e1 == e2 != e3 and sim > 0.8
    assert e1 == "evt_" + "a" * 12


def test_clusterer_window_expiry_starts_new_story() -> None:
    c = StoryClusterer(threshold=0.8, window_hours=48)
    e1, _, _ = c.assign("a" * 40, unit([1, 0, 0]), T0)
    e2, _, new = c.assign("b" * 40, unit([1, 0, 0]), T0 + timedelta(hours=49))
    assert new and e1 != e2


def test_novelty_is_one_minus_max_similarity() -> None:
    c = StoryClusterer(threshold=0.8, window_hours=48)
    assert c.novelty(unit([1, 0, 0]), T0, 72) == 1.0
    c.assign("a" * 40, unit([1, 0, 0]), T0)
    assert c.novelty(unit([1, 0, 0]), T0, 72) == pytest.approx(0.0, abs=1e-9)
    assert c.novelty(unit([0, 1, 0]), T0, 72) == pytest.approx(1.0)


def test_novelty_sees_beyond_assignment_window() -> None:
    c = StoryClusterer(threshold=0.8, window_hours=48, retention_hours=72)
    c.assign("a" * 40, unit([1, 0, 0]), T0)
    later = T0 + timedelta(hours=60)  # outside 48 h assignment window, inside 72 h lookback
    assert c.novelty(unit([1, 0, 0]), later, 72) == pytest.approx(0.0, abs=1e-9)
    _, _, new = c.assign("b" * 40, unit([1, 0, 0]), later)
    assert new


def test_sample_per_group_keeps_label_column() -> None:
    import pandas as pd

    from riskpulse.engine.event_training import sample_per_group

    df = pd.DataFrame({"text": list("abcdef"), "label": ["X", "X", "X", "Y", "Y", "Z"]})
    out = sample_per_group(df, "label", 2, seed=0)
    assert "label" in out.columns
    assert out["label"].value_counts().to_dict() == {"X": 2, "Y": 2, "Z": 1}
