"""Module A runner: daily backtest over the replay window -> reports (metrics, figures, weights)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from riskpulse.common.config import data_path, load_config, reports_path
from riskpulse.common.metrics import update_metrics
from riskpulse.ingestion.prices import trading_days, wide
from riskpulse.moduleA.backtest import daily_sentiment, run_strategies
from riskpulse.moduleA.metrics import information_coefficient, performance


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """(mentions of universe tickers, daily returns aligned to the window, tickers)."""
    tickers = list(load_config("universe")["tickers"])
    win = load_config("app")["replay_feed"]
    m = pd.read_parquet(data_path("processed", "mentions.parquet"))
    m = m[m["ticker"].isin(tickers)].copy()
    m["published_at"] = pd.to_datetime(m["published_at"], utc=True)
    days = trading_days()
    days = days[(days >= pd.Timestamp(win["start"])) & (days < pd.Timestamp(win["end"]))]
    px = wide("adj_close", tickers).reindex(days)
    rets = px.pct_change().fillna(0.0)[tickers]
    return m, rets, tickers


def run(write: bool = True) -> dict:
    cfg = load_config("moduleA")
    m, rets, tickers = load_inputs()
    hl = float(cfg["signal"]["half_life_hours"])
    s, c, net = daily_sentiment(m, rets.index, tickers, hl)
    res = run_strategies(rets, s, c, net, cfg)
    perf = {k: performance(v) for k, v in res.items()}
    ic = information_coefficient(s, rets)
    ic_series = ic.pop("series")
    # robustness grid: kappa x half-life (no parameter is chosen from this grid)
    grid = []
    for hl_g in cfg["robustness_grid"]["half_life_hours"]:
        s_g, c_g, net_g = daily_sentiment(m, rets.index, tickers, float(hl_g))
        ew = performance(run_strategies(rets, s_g, c_g, net_g, cfg, kappa=0.0)["sentiment_tilt"])
        for k in cfg["robustness_grid"]["kappa"]:
            p = performance(
                run_strategies(rets, s_g, c_g, net_g, cfg, kappa=float(k))["sentiment_tilt"]
            )
            grid.append(
                {
                    "kappa": k,
                    "half_life_hours": hl_g,
                    "cumulative_return": p["cumulative_return"],
                    "sharpe_rf0": p["sharpe_rf0"],
                    "excess_vs_ew_rebalanced": round(
                        p["cumulative_return"] - ew["cumulative_return"], 4
                    ),
                    "avg_turnover": p["avg_daily_one_way_turnover"],
                }
            )
        ic_g = information_coefficient(s_g, rets)
        ic_g.pop("series")
        grid.append(
            {"kappa": "IC", "half_life_hours": hl_g, **{f"ic_{k}": v for k, v in ic_g.items()}}
        )
    coverage = (c > 0).mean().round(3).to_dict()
    payload = {
        "window": [str(rets.index[0].date()), str(rets.index[-1].date())],
        "n_tickers": len(tickers),
        "config": {
            "kappa": cfg["tilt"]["kappa"],
            "half_life_hours": hl,
            "deadband": cfg["tilt"]["deadband"],
            "bounds": [cfg["constraints"]["w_min"], cfg["constraints"]["w_max"]],
            "tau_max": cfg["turnover"]["tau_max"],
            "cost_bps": cfg["turnover"]["cost_bps"],
        },
        "performance": perf,
        "information_coefficient": ic,
        "robustness_grid": grid,
        "signal_coverage_share_of_days": coverage,
        "note": "Sentiment-tilt demonstration on a 1-year window; not an alpha claim. rf = 0.",
    }
    if write:
        update_metrics("moduleA", payload, script="riskpulse backtest")
        out = data_path("processed", "moduleA")
        out.mkdir(parents=True, exist_ok=True)
        res["sentiment_tilt"].weights.to_parquet(out / "weights_tilt.parquet")
        pd.DataFrame({k: v.returns for k, v in res.items()}).to_parquet(out / "returns.parquet")
        s.to_parquet(out / "daily_sentiment.parquet")
        c.to_parquet(out / "daily_confidence.parquet")
        ic_series.to_frame("ic").to_parquet(out / "ic.parquet")
        _figures(res, ic_series)
    return payload


def _figures(res: dict, ic: pd.Series) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "sentiment_tilt": "#1f3a6e",
        "equal_weight_buy_hold": "#8a8f98",
        "equal_weight_rebalanced": "#b8bcc4",
        "naive_sign_rule": "#c9a227",
    }
    fig, ax = plt.subplots(figsize=(9, 4))
    for k, r in res.items():
        ax.plot((1 + r.returns).cumprod(), label=k.replace("_", " "), color=colors[k], lw=1.6)
    ax.set_title("Module A: growth of 1 (net of costs)", fontsize=11, color="#1f2a44")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(reports_path("figures", "moduleA_performance.png"), dpi=160)
    plt.close(fig)
    w = res["sentiment_tilt"].weights
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.stackplot(
        w.index,
        w.T.to_numpy(),
        labels=w.columns,
        alpha=0.9,
        colors=plt.cm.tab20(np.linspace(0, 1, w.shape[1])),
    )
    ax.set_ylim(0, 1)
    ax.set_title("Module A: sentiment-tilted weights over time", fontsize=11, color="#1f2a44")
    ax.legend(ncol=10, fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    fig.savefig(reports_path("figures", "moduleA_weights.png"), dpi=160)
    plt.close(fig)
    if not ic.empty:
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.bar(ic.index, ic.to_numpy(), color=np.where(ic > 0, "#2e7d32", "#c62828"), width=1.0)
        ax.plot(ic.rolling(20).mean(), color="#1f3a6e", lw=1.4, label="20-day mean")
        ax.axhline(0, color="#444", lw=0.6)
        ax.set_title(
            "Daily information coefficient (Spearman, s_t vs r_t+1)", fontsize=11, color="#1f2a44"
        )
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(reports_path("figures", "moduleA_ic.png"), dpi=160)
        plt.close(fig)


def summary_json() -> str:
    return json.dumps(run(write=False)["performance"], indent=2)
