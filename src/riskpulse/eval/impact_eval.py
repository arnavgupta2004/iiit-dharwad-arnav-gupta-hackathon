"""Impact evaluation against realised market reactions (spec §5.5, §9; D-014).

Unit: company ticker-day. Day 0 = first trading session on which an item could be acted on
(16:00 ET rule, ``market_date``). For each ticker-day:
  impact_v1  = max raw v1 impact of that ticker's mentions mapped to day 0
  abs_sent   = max |entity sentiment| of the same mentions (the naive baseline)
Target: |CAR[0,+1]| from a market model r_i = a + b r_SPY fitted on [-120, -20] trading days;
secondary: abnormal volume log(V_0 / mean V[-30, -5]).
Reported: Spearman rho (with p-value), paired-bootstrap 95% CI of rho(v1) - rho(|s|), mean |CAR|
by decile, and the top-decile hit rate (share of top-decile scores whose |CAR| is in the top
decile; 10% by chance).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from riskpulse.common.config import data_path, load_config, reports_path
from riskpulse.common.metrics import update_metrics
from riskpulse.common.timeutils import market_date
from riskpulse.eval.impact_v2 import v1_from_drivers
from riskpulse.ingestion.prices import trading_days, wide

EST, GAP, VOL_LO, VOL_HI = 120, 20, 30, 5


def ticker_day_frame(mentions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate company mentions to ticker x market-day observations."""
    tickers = list(load_config("universe")["tickers"])
    m = mentions[mentions["ticker"].isin(tickers)].copy()
    days = trading_days()
    m["day0"] = [market_date(t, days) for t in pd.to_datetime(m["published_at"], utc=True)]
    m["abs_sent"] = m["sentiment"].abs()
    # v1 from the stored drivers (company mentions are scored live by v2 once it is adopted).
    m["v1_raw"] = v1_from_drivers(m["drivers"])
    return (
        m.groupby(["ticker", "day0"])
        .agg(
            impact_v1=("v1_raw", "max"),
            abs_sent=("abs_sent", "max"),
            n_mentions=("doc_id", "nunique"),
        )
        .reset_index()
    )


def add_car(obs: pd.DataFrame) -> pd.DataFrame:
    """Attach CAR[0,+1] and abnormal volume (market model on SPY)."""
    tickers = sorted(obs["ticker"].unique())
    px = wide("adj_close", [*tickers, "SPY"])
    vol = wide("volume", tickers)
    ret = px.pct_change()
    idx = ret.index
    cars, avs = [], []
    for t, d0 in zip(obs["ticker"], obs["day0"], strict=True):
        k = idx.searchsorted(d0)
        if k - EST < 1 or k + 1 >= len(idx):
            cars.append(np.nan)
            avs.append(np.nan)
            continue
        est = ret.iloc[k - EST : k - GAP][[t, "SPY"]].dropna()
        if len(est) < 60:
            cars.append(np.nan)
            avs.append(np.nan)
            continue
        b, a = np.polyfit(est["SPY"], est[t], 1)
        win = ret.iloc[k : k + 2]
        ar = win[t] - (a + b * win["SPY"])
        cars.append(float(ar.sum()))
        base = vol[t].iloc[k - VOL_LO : k - VOL_HI].mean()
        v0 = vol[t].iloc[k]
        avs.append(float(np.log(v0 / base)) if base > 0 and v0 > 0 else np.nan)
    out = obs.copy()
    out["car01"] = cars
    out["abs_car01"] = np.abs(out["car01"])
    out["abn_volume"] = avs
    return out.dropna(subset=["abs_car01"])


def _deciles(score: pd.Series, target: pd.Series) -> list[float]:
    q = pd.qcut(score.rank(method="first"), 10, labels=False)
    return [round(float(target[q == i].mean()), 6) for i in range(10)]


def _top_decile_hit(score: pd.Series, target: pd.Series) -> float:
    top_s = score >= score.quantile(0.9)
    top_t = target >= target.quantile(0.9)
    return round(float((top_s & top_t).sum() / top_s.sum()), 4)


def evaluate(df: pd.DataFrame, seed: int = 0, n_boot: int = 1000) -> dict:
    res: dict = {"n_ticker_days": int(len(df)), "n_tickers": int(df["ticker"].nunique())}
    for target in ("abs_car01", "abn_volume"):
        d = df.dropna(subset=[target])
        block = {}
        for name in ("impact_v1", "abs_sent"):
            rho, p = spearmanr(d[name], d[target])
            block[name] = {
                "spearman_rho": round(float(rho), 4),
                "p_value": float(f"{p:.3g}"),
                "decile_means": _deciles(d[name], d[target]),
                "top_decile_hit_rate": _top_decile_hit(d[name], d[target]),
            }
        rng = np.random.default_rng(seed)
        diffs = []
        n = len(d)
        for _ in range(n_boot):
            i = rng.integers(0, n, n)
            s = d.iloc[i]
            diffs.append(
                spearmanr(s["impact_v1"], s[target])[0] - spearmanr(s["abs_sent"], s[target])[0]
            )
        block["rho_diff_v1_minus_abs_sent_95ci"] = [
            round(float(np.percentile(diffs, 2.5)), 4),
            round(float(np.percentile(diffs, 97.5)), 4),
        ]
        res[target] = block
    return res


def run() -> dict:
    m = pd.read_parquet(data_path("processed", "mentions.parquet"))
    obs = add_car(ticker_day_frame(m))
    burn_end = pd.Timestamp(load_config("impact")["binning"]["burn_in"][1])
    post = obs[obs["day0"] >= burn_end]
    payload = {
        "method": "market model on SPY over [-120,-20] trading days; CAR[0,+1]; ticker-day level; "
        "day 0 = first session at/after publication (16:00 ET rule)",
        "replay_window_post_burn_in": evaluate(post),
        "replay_window_full": evaluate(obs),
        "post_burn_in_start": str(burn_end.date()),
        "v2_status": "TBD (Benzinga event study, time split train <= 2018 / validate 2019-2020)",
        "note": "v1 raw score has no fitted parameters (weights set a priori).",
    }
    update_metrics("impact", payload, script="riskpulse eval impact")
    _figure(post)
    return payload


def _figure(df: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, (name, label) in zip(
        axes, [("impact_v1", "Impact v1 (raw)"), ("abs_sent", "|sentiment| baseline")], strict=True
    ):
        dm = np.array(_deciles(df[name], df["abs_car01"])) * 100
        ax.bar(range(1, 11), dm, color="#1f3a6e" if name == "impact_v1" else "#8a8f98", width=0.7)
        rho = spearmanr(df[name], df["abs_car01"])[0]
        ax.set_title(f"{label}: Spearman {rho:.3f}", fontsize=10, color="#1f2a44")
        ax.set_xlabel("score decile (1 = lowest)")
        ax.set_xticks(range(1, 11))
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("mean |CAR[0,+1]| (%)")
    fig.suptitle(
        f"Mean absolute abnormal return by score decile (n = {len(df):,} ticker-days)", fontsize=11
    )
    fig.tight_layout()
    fig.savefig(reports_path("figures", "impact_deciles.png"), dpi=160)
    plt.close(fig)
