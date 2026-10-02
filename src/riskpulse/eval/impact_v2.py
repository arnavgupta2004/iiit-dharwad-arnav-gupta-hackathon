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
  rho(v2) - rho(|s|) and rho(v2) - rho(v1) both lie above zero. Otherwise report and stop.
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
from riskpulse.engine.impact import BreadthTracker, ImpactFeatures, VelocityTracker, raw_impact
from riskpulse.eval.event_study_data import (
    embeddings,
    headlines,
    select_universe,
    stage1,
)
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
            "v1": m["impact_raw"].to_numpy(),
        }
    )
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


# ---------- run ----------
def run() -> dict:
    import lightgbm as lgb

    df = headlines()
    keep, px_long = select_universe(df)
    sub = df[df["ticker"].isin(keep)].sort_values("published_at").reset_index(drop=True)
    meta = stage1(sub)
    emb = embeddings()
    items = benzinga_item_features(meta, emb)
    pxw = px_long.pivot(index="date", columns="symbol", values="adj_close").sort_index()
    volw = px_long.pivot(index="date", columns="symbol", values="volume").sort_index()
    bz = attach_car(ticker_days(items, pd.DatetimeIndex(pxw.index)), pxw, volw)
    bz["rank_y"] = bz["abs_car01"].rank(pct=True)
    train = bz[bz["day0"] <= TRAIN_END]
    valid = bz[(bz["day0"] > TRAIN_END) & (bz["day0"] <= VALID_END)]
    log.info(f"Benzinga ticker-days: train {len(train):,}, valid {len(valid):,}")

    model = lgb.LGBMRegressor(monotone_constraints=[1] * len(FEATS), **PARAMS)
    model.fit(
        train[FEATS],
        train["rank_y"].rank(pct=True),
        eval_set=[(valid[FEATS], valid["rank_y"].rank(pct=True))],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(MODEL_PATH))
    valid = valid.assign(v2=model.predict(valid[FEATS]))

    # ---- test: 2021-22 replay ticker-days, scored once ----
    mentions = pd.read_parquet(data_path("processed", "mentions.parquet"))
    rep = replay_item_features(mentions)
    tickers = sorted(rep["ticker"].unique())
    test = attach_car(
        ticker_days(rep, trading_days()),
        wide("adj_close", [*tickers, "SPY"]),
        wide("volume", tickers),
    )
    test = test.assign(v2=model.predict(test[FEATS]))
    burn_end = pd.Timestamp(load_config("impact")["binning"]["burn_in"][1])
    post = test[test["day0"] >= burn_end]

    t_car = compare(post, "abs_car01")
    lo_s, lo_v1 = t_car["rho_diff_v2_minus_abs_sent_95ci"][0], t_car["rho_diff_v2_minus_v1_95ci"][0]
    adopted = bool(lo_s > 0 and lo_v1 > 0)
    payload = {
        "protocol": "Benzinga train <= 2018, valid 2019-2020-07 (early stopping only); "
        "test once on the 2021-22 replay ticker-days used for v1",
        "adoption_rule": "adopt v2 only if 95% CIs of rho(v2)-rho(|s|) and "
        "rho(v2)-rho(v1) are both > 0",
        "adopted": adopted,
        "benzinga": {
            "tickers": len(keep),
            "tickers_dropped_no_prices": 300 - len(keep),
            "train_ticker_days": int(len(train)),
            "valid_ticker_days": int(len(valid)),
            "best_iteration": int(model.best_iteration_ or 0),
            "feature_importance_gain": {
                f: round(float(g), 1)
                for f, g in zip(FEATS, model.booster_.feature_importance("gain"), strict=True)
            },
            "validation_abs_car01": compare(valid, "abs_car01"),
        },
        "test_post_burn_in": {"abs_car01": t_car, "abn_volume": compare(post, "abn_volume")},
        "test_full_window": {"abs_car01": compare(test, "abs_car01")},
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
