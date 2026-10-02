"""Impact v2: monotone gradient boosting on v1's drivers, calibrated on the Benzinga event study.

Protocol (D-014, Arnav's GATE C item 2), fixed before any test result was seen:
- Features per ticker-day (identical for train and test): max |sentiment| (S), class prior (T),
  velocity (V), breadth (B), novelty (N), credibility (C), relevance (R), log(1 + mentions).
- Target: |CAR[0,+1]| (market model on SPY, [-120,-20]); trained on its percentile rank.
- Train: events <= 2018-12-31. Validation (early stopping only): 2019-01-01 -> 2020-07.
- Model: LightGBM, every feature monotone increasing, hyperparameters fixed a priori.
- Test (scored once): the 2021-22 replay ticker-days used for v1 (post-burn-in main, full window
  secondary), same metrics and CIs as v1.
- Pre-registered adoption rule: v2 replaces v1 only if the paired-bootstrap 95% CIs of
  rho(v2) - rho(|s|) and rho(v2) - rho(v1) both lie above zero. Not met at the first test; the
  rule was revised afterwards to the spec rule "v2 beats v1" on validation and test (D-039).
- `train()` fits in a spawned subprocess and saves the model text plus a JSON tree dump
  (`riskpulse train impact_v2`); `evaluate()` scores the JSON trees with engine/gbm.py
  (`riskpulse eval impact_v2`), so this process never loads lightgbm (OpenMP clash with torch).
"""

from __future__ import annotations

import json
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from riskpulse.common.config import data_path, load_config, reports_path
from riskpulse.common.logging import get_logger
from riskpulse.common.metrics import update_metrics
from riskpulse.engine.clustering import StoryClusterer
from riskpulse.engine.gbm import TreeEnsemble
from riskpulse.engine.impact import BreadthTracker, ImpactFeatures, VelocityTracker, raw_impact
from riskpulse.eval.event_study_data import embeddings, stage1, universe_headlines
from riskpulse.ingestion.prices import trading_days, wide

log = get_logger()
FEATS = ["S", "T", "V", "B", "N", "C", "R", "log_n"]
TRAIN_END, VALID_END = pd.Timestamp("2018-12-31"), pd.Timestamp("2020-07-31")
EST, GAP, VOL_LO, VOL_HI = 120, 20, 30, 5
PARAMS = dict(
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=15,
    min_child_samples=200,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=20261002,
    verbose=-1,
)
MODEL_PATH = data_path("processed", "models", "impact_v2.txt")


# ---------- features ----------
def benzinga_item_features(meta: pd.DataFrame, emb: np.ndarray) -> pd.DataFrame:
    """v1 drivers for every Benzinga headline, computed in time order with the engine's trackers."""
    cfg = load_config("impact")
    vel, br = VelocityTracker(cfg), BreadthTracker(cfg)
    clus = StoryClusterer()
    look = float(cfg["novelty"]["lookback_hours"])
    priors, cred = cfg["type_prior"], float(cfg["credibility"]["kaggle_news"])
    rows = []
    for i, r in enumerate(meta.itertuples(index=False)):
        ts = r.published_at.to_pydatetime()
        v, _ = vel.observe(r.ticker, ts)
        b, _ = br.observe(r.ticker, "benzinga", ts)
        _, _, _, nov = clus.observe(f"bz{i:012d}", emb[i], ts, look)
        f = ImpactFeatures(float(priors[r.event_class]), abs(r.sentiment), v, b, nov, cred, 1.0)
        raw, _ = raw_impact(f, cfg)
        rows.append(
            (r.ticker, r.published_at, f.sentiment, f.type_prior, v, b, nov, cred, 1.0, raw)
        )
        if i % 50_000 == 0:
            log.info(f"v2 features: {i:,}/{len(meta):,}")
    return pd.DataFrame(rows, columns=["ticker", "published_at", *FEATS[:-1], "v1"])


def v1_from_drivers(drivers: pd.Series) -> np.ndarray:
    """v1 raw impact recomputed from mentions' stored driver JSON (not the live model's raw)."""
    cfg = load_config("impact")
    keys = ("type_prior", "sentiment", "velocity", "breadth", "novelty", "credibility", "relevance")
    return np.array(
        [raw_impact(ImpactFeatures(*(x[k] for k in keys)), cfg)[0] for x in drivers.map(json.loads)]
    )


def replay_item_features(mentions: pd.DataFrame) -> pd.DataFrame:
    """Same columns from the replay's scored mentions (drivers stored per mention)."""
    tickers = list(load_config("universe")["tickers"])
    m = mentions[mentions["ticker"].isin(tickers)].copy()
    d = m["drivers"].map(json.loads)
    out = pd.DataFrame(
        {
            "ticker": m["ticker"].to_numpy(),
            "published_at": pd.to_datetime(m["published_at"], utc=True).to_numpy(),
            "S": m["sentiment"].abs().to_numpy(),
            "T": d.map(lambda x: x["type_prior"]).to_numpy(),
            "V": d.map(lambda x: x["velocity"]).to_numpy(),
            "B": d.map(lambda x: x["breadth"]).to_numpy(),
            "N": d.map(lambda x: x["novelty"]).to_numpy(),
            "C": d.map(lambda x: x["credibility"]).to_numpy(),
            "R": d.map(lambda x: x["relevance"]).to_numpy(),
        }
    )
    # v1 recomputed from the stored drivers: once v2 scores company mentions, `impact_raw` holds v2.
    out["v1"] = v1_from_drivers(m["drivers"])
    out["published_at"] = pd.to_datetime(out["published_at"], utc=True)
    return out


def market_days(ts: pd.Series, days: pd.DatetimeIndex) -> pd.Series:
    """Vectorised 16:00-ET rule: first session at/after publication."""
    local = pd.DatetimeIndex(ts).tz_convert(ZoneInfo("America/New_York"))
    d = local.normalize().tz_localize(None)
    d = d + pd.to_timedelta((local.hour >= 16).astype(int), unit="D")
    pos = days.searchsorted(d, side="left")
    pos = np.minimum(pos, len(days) - 1)
    return pd.Series(days[pos], index=ts.index)


def ticker_days(items: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
    it = items.copy()
    it["day0"] = market_days(it["published_at"], days)
    g = it.groupby(["ticker", "day0"])
    out = g[["S", "T", "V", "B", "N", "C", "R", "v1"]].max()
    out["log_n"] = np.log1p(g.size())
    return out.reset_index()


def attach_car(obs: pd.DataFrame, px: pd.DataFrame, vol: pd.DataFrame) -> pd.DataFrame:
    ret = px.pct_change()
    idx = ret.index
    car, av = np.full(len(obs), np.nan), np.full(len(obs), np.nan)
    for j, (t, d0) in enumerate(zip(obs["ticker"], obs["day0"], strict=True)):
        if t not in ret:
            continue
        k = idx.searchsorted(d0)
        if k - EST < 1 or k + 1 >= len(idx):
            continue
        est = ret.iloc[k - EST : k - GAP][[t, "SPY"]].dropna()
        if len(est) < 60:
            continue
        b, a = np.polyfit(est["SPY"], est[t], 1)
        win = ret.iloc[k : k + 2]
        car[j] = float((win[t] - (a + b * win["SPY"])).sum())
        base, v0 = vol[t].iloc[k - VOL_LO : k - VOL_HI].mean(), vol[t].iloc[k]
        if base > 0 and v0 > 0:
            av[j] = float(np.log(v0 / base))
    out = obs.copy()
    out["abs_car01"], out["abn_volume"] = np.abs(car), av
    return out.dropna(subset=["abs_car01"])


# ---------- metrics ----------
def _rho(a, b) -> float:
    return float(spearmanr(a, b)[0])


def _deciles(score: pd.Series, target: pd.Series) -> list[float]:
    q = pd.qcut(score.rank(method="first"), 10, labels=False)
    return [round(float(target[q == i].mean()), 6) for i in range(10)]


def _hit(score: pd.Series, target: pd.Series) -> float:
    top_s, top_t = score >= score.quantile(0.9), target >= target.quantile(0.9)
    return round(float((top_s & top_t).sum() / top_s.sum()), 4)


def compare(df: pd.DataFrame, target: str, seed: int = 0, n_boot: int = 1000) -> dict:
    d = df.dropna(subset=[target])
    scores = {"impact_v2": d["v2"], "impact_v1": d["v1"], "abs_sent": d["S"]}
    res: dict = {"n": int(len(d))}
    for k, s in scores.items():
        rho, p = spearmanr(s, d[target])
        res[k] = {
            "spearman_rho": round(float(rho), 4),
            "p_value": float(f"{p:.3g}"),
            "top_decile_hit_rate": _hit(s, d[target]),
            "decile_means": _deciles(s, d[target]),
        }
    rng = np.random.default_rng(seed)
    dv2s, dv2v1 = [], []
    n = len(d)
    y, v2, v1, s = (d[c].to_numpy() for c in (target, "v2", "v1", "S"))
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        r2 = _rho(v2[i], y[i])
        dv2s.append(r2 - _rho(s[i], y[i]))
        dv2v1.append(r2 - _rho(v1[i], y[i]))
    ci = lambda x: [round(float(np.percentile(x, 2.5)), 4), round(float(np.percentile(x, 97.5)), 4)]  # noqa: E731
    res["rho_diff_v2_minus_abs_sent_95ci"] = ci(dv2s)
    res["rho_diff_v2_minus_v1_95ci"] = ci(dv2v1)
    return res


# ---------- train / evaluate ----------
META_PATH = data_path("processed", "models", "impact_v2_meta.json")
JSON_PATH = data_path("processed", "models", "impact_v2.json")


def _benzinga() -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """(train, valid) Benzinga ticker-days with features and |CAR|, and the ticker count."""
    sub, px_long = universe_headlines()
    meta = stage1(sub)
    items = benzinga_item_features(meta, embeddings())
    pxw = px_long.pivot(index="date", columns="symbol", values="adj_close").sort_index()
    volw = px_long.pivot(index="date", columns="symbol", values="volume").sort_index()
    bz = attach_car(ticker_days(items, pd.DatetimeIndex(pxw.index)), pxw, volw)
    bz["rank_y"] = bz["abs_car01"].rank(pct=True)
    train = bz[bz["day0"] <= TRAIN_END]
    valid = bz[(bz["day0"] > TRAIN_END) & (bz["day0"] <= VALID_END)]
    log.info(f"Benzinga ticker-days: train {len(train):,}, valid {len(valid):,}")
    return train, valid, int(sub["ticker"].nunique())


def _fit_in_subprocess(tr: pd.DataFrame, va: pd.DataFrame) -> dict:
    """LightGBM fit in a spawned process: lightgbm's OpenMP runtime must not share a process with
    torch on macOS (see engine/gbm.py). Returns the model text, its JSON dump and metadata."""
    import lightgbm as lgb

    model = lgb.LGBMRegressor(monotone_constraints=[1] * len(FEATS), **PARAMS)
    model.fit(
        tr[FEATS],
        tr["rank_y"].rank(pct=True),
        eval_set=[(va[FEATS], va["rank_y"].rank(pct=True))],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    b = model.booster_
    return {
        "check_pred": b.predict(va[FEATS].head(2000)).tolist(),
        "text": b.model_to_string(),
        "dump": b.dump_model(),
        "best_iteration": int(model.best_iteration_ or 0),
        "gain": [float(g) for g in b.feature_importance("gain")],
    }


def train() -> dict:
    """Fit v2 on Benzinga <= 2018 (early stopping on 2019 -> 2020-07) and save it. No test data."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    tr, va, n_tickers = _benzinga()
    with ProcessPoolExecutor(1, mp_context=mp.get_context("spawn")) as ex:
        fit = ex.submit(_fit_in_subprocess, tr[[*FEATS, "rank_y"]], va[[*FEATS, "rank_y"]]).result()
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.write_text(fit["text"])
    JSON_PATH.write_text(json.dumps(fit["dump"]))
    ens = TreeEnsemble.from_dump(fit["dump"])
    check = va[FEATS].head(2000).to_numpy()
    meta = {
        "tickers": n_tickers,
        "tickers_dropped_no_prices": 300 - n_tickers,
        "train_ticker_days": int(len(tr)),
        "valid_ticker_days": int(len(va)),
        "best_iteration": fit["best_iteration"],
        "n_trees": len(fit["dump"]["tree_info"]),
        "feature_importance_gain": {
            f: round(g, 1) for f, g in zip(FEATS, fit["gain"], strict=True)
        },
        "json_evaluator_max_abs_diff_vs_lightgbm": float(
            np.max(np.abs(ens.predict(check) - np.array(fit["check_pred"])))
        ),
    }
    if meta["json_evaluator_max_abs_diff_vs_lightgbm"] > 1e-9:
        raise RuntimeError(f"JSON tree evaluator disagrees with LightGBM: {meta}")
    META_PATH.write_text(json.dumps(meta, indent=2))
    log.info(f"impact v2 trained: {meta['n_trees']} trees, best iteration {meta['best_iteration']}")
    return meta


def evaluate() -> dict:
    """Score the saved v2 on validation and on the 2021-22 replay test; write metrics."""
    booster = TreeEnsemble.load(JSON_PATH)
    meta = json.loads(META_PATH.read_text())
    _, valid, _ = _benzinga()
    valid = valid.assign(v2=booster.predict(valid[FEATS]))

    mentions = pd.read_parquet(data_path("processed", "mentions.parquet"))
    rep = replay_item_features(mentions)
    tickers = sorted(rep["ticker"].unique())
    test = attach_car(
        ticker_days(rep, trading_days()),
        wide("adj_close", [*tickers, "SPY"]),
        wide("volume", tickers),
    )
    test = test.assign(v2=booster.predict(test[FEATS]))
    burn_end = pd.Timestamp(load_config("impact")["binning"]["burn_in"][1])
    post = test[test["day0"] >= burn_end]

    v_car = compare(valid, "abs_car01")
    t_car = compare(post, "abs_car01")
    pre_registered = bool(
        t_car["rho_diff_v2_minus_abs_sent_95ci"][0] > 0
        and t_car["rho_diff_v2_minus_v1_95ci"][0] > 0
    )
    beats_v1 = bool(
        v_car["rho_diff_v2_minus_v1_95ci"][0] > 0 and t_car["rho_diff_v2_minus_v1_95ci"][0] > 0
    )
    payload = {
        "protocol": "Benzinga train <= 2018, valid 2019-2020-07 (early stopping only); "
        "test on the 2021-22 replay ticker-days (post burn-in main, full window secondary)",
        "pre_registered_rule": "adopt only if 95% CIs of rho(v2)-rho(|s|) and rho(v2)-rho(v1) "
        "are both > 0 on test",
        "pre_registered_rule_met": pre_registered,
        "applied_rule": "spec rule, revised after the first test was seen (D-039): adopt if v2 "
        "beats v1 (CI of rho(v2)-rho(v1) > 0) on validation and on test",
        "v2_beats_v1_on_validation_and_test": beats_v1,
        "adopted": beats_v1,
        "scope": "company mentions; market-wide items keep v1 (D-042)",
        "benzinga": {**meta, "validation_abs_car01": v_car},
        "test_post_burn_in": {"abs_car01": t_car, "abn_volume": compare(post, "abn_volume")},
        "test_full_window": {"abs_car01": compare(test, "abs_car01")},
        "test_note": "v1 on the test set is recomputed from each mention's stored drivers.",
        "limitations": [
            "survivorship: delisted Benzinga tickers have no yfinance prices and are excluded",
            "Benzinga is single-publisher, so breadth is constant in training (no learned effect)",
            "domain shift: Benzinga headlines (2009-2020) vs GDELT news + tweets (2021-22)",
        ],
    }
    update_metrics("impact_v2", payload, script="riskpulse eval impact_v2")
    _figure(post)
    return payload


def _figure(df: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), sharey=True)
    for ax, (c, label, color) in zip(
        axes,
        [
            ("v2", "Impact v2 (event-study calibrated)", "#1f3a6e"),
            ("v1", "Impact v1 (heuristic)", "#4a6fa5"),
            ("S", "|sentiment| baseline", "#8a8f98"),
        ],
        strict=True,
    ):
        dm = np.array(_deciles(df[c], df["abs_car01"])) * 100
        ax.bar(range(1, 11), dm, color=color, width=0.7)
        ax.set_title(
            f"{label}: Spearman {_rho(df[c], df['abs_car01']):.3f}", fontsize=10, color="#1f2a44"
        )
        ax.set_xticks(range(1, 11))
        ax.set_xlabel("score decile (1 = lowest)")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("mean |CAR[0,+1]| (%)")
    fig.suptitle(f"Test set: 2021-22 ticker-days after burn-in (n = {len(df):,})", fontsize=11)
    fig.tight_layout()
    fig.savefig(reports_path("figures", "impact_v2_deciles.png"), dpi=160)
    plt.close(fig)
