"""Module A runner: daily backtest over the replay window -> reports (metrics, figures, weights)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from riskpulse.common.config import data_path, load_config, reports_path
from riskpulse.common.metrics import update_metrics
from riskpulse.ingestion.prices import trading_days, wide
from riskpulse.moduleA.backtest import daily_sentiment, run_strategies
from riskpulse.moduleA.calibrate import calibrate_kappa, save
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
    """Calibrate kappa on months 1-2 (signals only), freeze it, evaluate from month 3 (D-038)."""
    cfg = load_config("moduleA")
    m, rets, tickers = load_inputs()
    hl = float(cfg["signal"]["half_life_hours"])
    s_all, c_all, net_all = daily_sentiment(m, rets.index, tickers, hl)
    cal = cfg["calibration"]
    cal_end = rets.index[0] + pd.DateOffset(months=int(cal["calibration_months"]))
    in_cal = rets.index < cal_end
    calib = calibrate_kappa(
        s_all[in_cal], c_all[in_cal], cfg, float(cal["target_median_abs_active_weight"])
    )
    calib.update(
        {
            "calibration_window": [
                str(rets.index[in_cal][0].date()),
                str(rets.index[in_cal][-1].date()),
            ],
            "half_life_hours": hl,
            "uses_returns": False,
        }
    )
    kappa = float(calib["kappa"])
    ev = ~in_cal
    s, c, net, r = s_all[ev], c_all[ev], net_all[ev], rets[ev]
    res = run_strategies(r, s, c, net, cfg, kappa=kappa)
    perf = {k: performance(v) for k, v in res.items()}
    ic = information_coefficient(s, r)
    ic_series = ic.pop("series")
    ic_full = information_coefficient(s_all, rets)
    ic_full.pop("series")
    # sensitivity only: kappa x half-life on the evaluation window (nothing is chosen from it)
    grid = []
    for hl_g in cfg["robustness_grid"]["half_life_hours"]:
        s_g, c_g, net_g = (x[ev] for x in daily_sentiment(m, rets.index, tickers, float(hl_g)))
        ew = performance(run_strategies(r, s_g, c_g, net_g, cfg, kappa=0.0)["sentiment_tilt"])
        for k in sorted({*cfg["robustness_grid"]["kappa"], round(kappa, 2)}):
            p = performance(
                run_strategies(r, s_g, c_g, net_g, cfg, kappa=float(k))["sentiment_tilt"]
            )
            grid.append(
                {
                    "kappa": k,
                    "half_life_hours": hl_g,
                    "cumulative_return": p["cumulative_return"],
                    "excess_vs_ew_rebalanced": round(
                        p["cumulative_return"] - ew["cumulative_return"], 4
                    ),
                    "avg_turnover": p["avg_daily_one_way_turnover"],
                }
            )
    active = (res["sentiment_tilt"].weights - 1.0 / len(tickers)).abs()
    payload = {
        "headline_information_coefficient": {
            **ic,
            "pct_positive_days": ic.get("hit_rate"),
            "definition": "daily cross-sectional Spearman of s_t vs next-day return",
            "window": "evaluation window (after the calibration months)",
        },
        "ic_full_window_for_reference": ic_full,
        "calibration": calib,
        "evaluation_window": [str(r.index[0].date()), str(r.index[-1].date())],
        "realised_median_abs_active_weight_eval": round(float(active.stack().median()), 5),
        "config": {
            "kappa": kappa,
            "half_life_hours": hl,
            "deadband": cfg["tilt"]["deadband"],
            "bounds": [cfg["constraints"]["w_min"], cfg["constraints"]["w_max"]],
            "tau_max": cfg["turnover"]["tau_max"],
            "cost_bps": cfg["turnover"]["cost_bps"],
        },
        "performance_secondary": perf,
        "sensitivity_grid": grid,
        "signal_coverage_share_of_days": (c > 0).mean().round(3).to_dict(),
        "n_tickers": len(tickers),
        "note": "IC is the headline; returns are secondary (sentiment-tilt demonstration, "
        "not an alpha claim; rf = 0).",
    }
    if write:
        save(calib)
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
    return json.dumps(run(write=False)["performance_secondary"], indent=2)
