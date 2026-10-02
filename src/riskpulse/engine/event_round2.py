"""Event classification round 2 (D-052): candidate selection by cross-validation on gold-1 only.

Candidates (all CPU-cheap at feed scale):
  C1  calibrated logistic regression on MiniLM embeddings; weak labels + gold-1 train rows (x10)
  C2  as C1, features = embedding + log(1 + keyword hits) per class
  C3  C1, but when the keyword baseline hits exactly one non-OTHER class, output that class
References scored in the same folds (not selectable): the deployed model, the keyword baseline,
zero-shot (base). Gold-2 is never read here: its texts are only *excluded* from training, via the
unlabelled `to_label_2.csv`.

    python -m riskpulse.engine.event_round2
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from riskpulse.common.config import data_path, load_config, repo_root, reports_path
from riskpulse.common.logging import get_logger
from riskpulse.common.metrics import update_metrics
from riskpulse.engine.event_training import training_data
from riskpulse.engine.events import (
    CLASSES,
    Embedder,
    EmbeddingClassifier,
    KeywordClassifier,
    ZeroShotClassifier,
)
from riskpulse.engine.pipeline import EVENT_MODEL_PATH

log = get_logger()
GOLD_WEIGHT = 10.0
SEEDS = (1, 2, 3)
N_FOLDS = 5
SIMPLICITY = ("C1", "C3", "C2")
CV_PATH = reports_path("event_round2_cv.json")


def kw_features(texts: list[str]) -> np.ndarray:
    kw = KeywordClassifier()
    return np.array([[np.log1p(kw.hits(t).get(c, 0)) for c in CLASSES] for t in texts])


def kw_single(texts: list[str]) -> list[str | None]:
    """The keyword class when exactly one non-OTHER class is hit, else None (C3 override)."""
    kw, out = KeywordClassifier(), []
    for t in texts:
        h = [c for c, n in kw.hits(t).items() if n > 0 and c != "OTHER"]
        out.append(h[0] if len(h) == 1 else None)
    return out


def fit_lr(x: np.ndarray, y: list[str], w: np.ndarray, seed: int):
    """Same estimator as the deployed classifier (EmbeddingClassifier.fit), with sample weights."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression

    base = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced", random_state=seed)
    return CalibratedClassifierCV(base, method="isotonic", cv=3).fit(x, y, sample_weight=w)


def predict_lr(model, x: np.ndarray) -> np.ndarray:
    min_conf = float(load_config("app")["events"]["min_confidence"])
    p = model.predict_proba(x)
    cls = np.array(model.classes_)[p.argmax(1)]
    return np.where(p.max(1) < min_conf, "OTHER", cls)


def macro_f1(y: np.ndarray, p: np.ndarray) -> float:
    return float(f1_score(y, p, labels=CLASSES, average="macro", zero_division=0))


def load_inputs() -> dict:
    gold = pd.read_csv(data_path("gold", "labels.csv"))
    weak, _ = training_data()  # gold-1 and gold-2 texts excluded (load_gold_texts)
    emb = Embedder()
    men = pd.read_parquet(
        data_path("processed", "mentions.parquet"), columns=["doc_id", "event_id"]
    )
    story = men.drop_duplicates("doc_id").set_index("doc_id")["event_id"]
    g_texts = gold["text"].tolist()
    return {
        "gold_y": gold["label_event_class"].str.strip().to_numpy(),
        "gold_texts": g_texts,
        "groups": np.array([story.get(d, f"solo_{d}") for d in gold["doc_id"]]),
        "gold_x": emb.encode(g_texts),
        "gold_kw": kw_features(g_texts),
        "gold_kw1": kw_single(g_texts),
        "weak_y": weak["label"].tolist(),
        "weak_x": emb.encode(weak["text"].tolist()),
        "weak_kw": kw_features(weak["text"].tolist()),
        "n_weak": len(weak),
    }


def candidate_predictions(d: dict, tr: np.ndarray, te: np.ndarray, seed: int) -> dict:
    y_tr = list(d["weak_y"]) + list(d["gold_y"][tr])
    w = np.r_[np.ones(d["n_weak"]), np.full(len(tr), GOLD_WEIGHT)]
    x1 = np.vstack([d["weak_x"], d["gold_x"][tr]])
    m1 = fit_lr(x1, y_tr, w, seed)
    p1 = predict_lr(m1, d["gold_x"][te])
    x2 = np.hstack([x1, np.vstack([d["weak_kw"], d["gold_kw"][tr]])])
    m2 = fit_lr(x2, y_tr, w, seed)
    p2 = predict_lr(m2, np.hstack([d["gold_x"][te], d["gold_kw"][te]]))
    p3 = np.array(
        [k or c for k, c in zip(np.array(d["gold_kw1"], dtype=object)[te], p1, strict=True)]
    )
    return {"C1": p1, "C2": p2, "C3": p3}


def run() -> dict:
    d = load_inputs()
    y = d["gold_y"]
    current = EmbeddingClassifier.load(repo_root() / EVENT_MODEL_PATH)
    refs = {
        "current_deployed": np.array(
            [p.event_class for p in current.predict_from_embeddings(d["gold_x"])]
        ),
        "keyword_baseline": np.array(
            [p.event_class for p in KeywordClassifier().predict(d["gold_texts"])]
        ),
        "zero_shot_base": np.array(
            [p.event_class for p in ZeroShotClassifier().predict(d["gold_texts"])]
        ),
    }
    folds: list[dict] = []
    for seed in SEEDS:
        cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for k, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y, d["groups"])):
            preds = candidate_predictions(d, tr, te, seed)
            row = {"seed": seed, "fold": k, "n_test": int(len(te))}
            row |= {c: macro_f1(y[te], p) for c, p in preds.items()}
            row |= {r: macro_f1(y[te], p[te]) for r, p in refs.items()}
            folds.append(row)
            log.info(
                f"round-2 CV seed {seed} fold {k}: " + ", ".join(f"{c} {row[c]:.3f}" for c in preds)
            )
    df = pd.DataFrame(folds)
    methods = ["C1", "C2", "C3", *refs]
    summary = {
        m: {
            "mean": round(float(df[m].mean()), 4),
            "se": round(float(df[m].std(ddof=1) / np.sqrt(len(df))), 4),
        }
        for m in methods
    }
    best = max(("C1", "C2", "C3"), key=lambda c: summary[c]["mean"])
    floor = summary[best]["mean"] - summary[best]["se"]
    chosen = next(c for c in SIMPLICITY if summary[c]["mean"] >= floor)
    beats_current = summary[chosen]["mean"] > summary["current_deployed"]["mean"]
    payload = {
        "protocol": "D-052: StratifiedGroupKFold 5 folds x seeds (1, 2, 3) on gold-1 only, "
        "grouped by story; macro-F1 over 10 classes; highest mean, then one-SE rule toward "
        "simpler (C1, C3, C2); must beat the deployed model's CV mean",
        "n_gold1": int(len(y)),
        "n_weak_rows": int(d["n_weak"]),
        "gold_weight": GOLD_WEIGHT,
        "summary": summary,
        "best_by_mean": best,
        "one_se_floor": round(floor, 4),
        "chosen_by_one_se_rule": chosen,
        "chosen_beats_current_deployed": bool(beats_current),
        "selected": chosen if beats_current else "current_deployed",
        "folds": folds,
        "note": "Gold-2 not read. References are not retrained; scored on the same folds.",
    }
    CV_PATH.write_text(json.dumps(payload, indent=2))
    update_metrics("events_round2_cv", payload, script="python -m riskpulse.engine.event_round2")
    return payload


if __name__ == "__main__":
    out = run()
    print(
        json.dumps(
            {
                k: out[k]
                for k in (
                    "summary",
                    "best_by_mean",
                    "one_se_floor",
                    "chosen_by_one_se_rule",
                    "chosen_beats_current_deployed",
                    "selected",
                )
            },
            indent=2,
        )
    )


FINAL_PATH = data_path("trained", "event_clf_round2.pkl")


def train_final(candidate: str = "C1") -> dict:
    """Fit the selected recipe on weak labels + all of gold-1 and save it (D-052 step 4).
    Only C1 is implemented here, as it is the recipe the rule selected (D-053)."""
    import hashlib

    if candidate != "C1":
        raise ValueError("only C1 was selected (D-053)")
    d = load_inputs()
    y = list(d["weak_y"]) + list(d["gold_y"])
    w = np.r_[np.ones(d["n_weak"]), np.full(len(d["gold_y"]), GOLD_WEIGHT)]
    model = fit_lr(np.vstack([d["weak_x"], d["gold_x"]]), y, w, seed=20261002)
    clf = EmbeddingClassifier(embedder=Embedder())
    clf.model, clf.classes_ = model, list(model.classes_)
    clf.version = f"embed-lr+gold1@{clf.embedder.model_name}"
    clf.save(FINAL_PATH)
    sha = hashlib.sha256(FINAL_PATH.read_bytes()).hexdigest()
    return {
        "path": str(FINAL_PATH.relative_to(repo_root())),
        "sha256": sha,
        "rows": len(y),
        "version": clf.version,
    }
