"""Render docs/architecture.png (light consulting style).

python scripts/make_architecture_diagram.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from riskpulse.common.config import repo_root

NAVY, CHARCOAL, MUTED, LINE = "#1f2a44", "#3a3f47", "#6b7280", "#c7ccd4"
FILL, FILL_ENGINE, FILL_MOD = "#f7f8fa", "#eef3fb", "#f5f7fb"


def box(ax, x, y, w, h, title, lines, fill=FILL, title_color=NAVY):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.008,rounding_size=0.012",
            linewidth=1.1,
            edgecolor=LINE,
            facecolor=fill,
        )
    )
    ax.text(
        x + 0.012,
        y + h - 0.028,
        title,
        fontsize=10.5,
        fontweight="bold",
        color=title_color,
        va="top",
    )
    for i, line in enumerate(lines):
        ax.text(x + 0.014, y + h - 0.062 - i * 0.031, line, fontsize=8.4, color=CHARCOAL, va="top")


def arrow(ax, x0, y0, x1, y1, label=None, rad=0.0, label_xy=None):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=12,
            color=MUTED,
            linewidth=1.2,
            connectionstyle=f"arc3,rad={rad}",
        )
    )
    if label:
        lx, ly = label_xy or ((x0 + x1) / 2, max(y0, y1) + 0.022)
        ax.text(lx, ly, label, fontsize=7.8, color=MUTED, ha="center", va="bottom")


def main() -> None:
    fig, ax = plt.subplots(figsize=(15, 9.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1)
    ax.axis("off")
    ax.text(
        0.01,
        0.975,
        "RiskPulse: system architecture",
        fontsize=16,
        fontweight="bold",
        color=NAVY,
        va="top",
    )
    ax.text(
        0.01,
        0.94,
        "Unstructured news and social text -> structured risk signals -> index rebalancing "
        "and portfolio stress testing (CPU-only, no paid keys)",
        fontsize=9.5,
        color=MUTED,
        va="top",
    )

    box(
        ax,
        0.01,
        0.60,
        0.22,
        0.30,
        "Sources",
        [
            "News: GDELT GKG 2.0 (2021-22 replay)",
            "News: GDELT DOC API (live mode)",
            "News: NewsAPI (optional key)",
            "Social: Kaggle stock tweets",
            "Calibration: Benzinga headlines",
            "Prices: yfinance cache (46 symbols)",
            "Book seed: card-transaction merchants",
        ],
    )
    box(
        ax,
        0.27,
        0.60,
        0.20,
        0.30,
        "Ingestion",
        [
            "Adapters (batch / live / replay)",
            "In-memory GKG stream + filter",
            "Normalise text, UTC timestamps",
            "Exact + near-duplicate removal",
            "Replay feed: 203k documents",
            "Inject: labelled synthetic_demo",
        ],
    )
    box(
        ax,
        0.51,
        0.47,
        0.25,
        0.43,
        "NLP Risk Engine",
        [
            "1  Entity linking: aliases, cashtags,",
            "    context rules, MKT + region tags",
            "2  Story clustering: MiniLM, 48 h window",
            "3  Sentiment: FinBERT, doc + entity level",
            "4  Event class: calibrated embedding LR",
            "    (keyword + zero-shot NLI baselines)",
            "5  Impact 1-10: type prior x (sentiment,",
            "    velocity, breadth, novelty, credibility)",
            "    x relevance; burn-in quantile bins",
            "6  Aggregation: entity EWMA + event stories",
        ],
        fill=FILL_ENGINE,
    )
    box(
        ax,
        0.80,
        0.60,
        0.19,
        0.30,
        "Delivery",
        [
            "signals.jsonl (append-only file)",
            "DuckDB store (queries)",
            "FastAPI: /signals /events",
            "/stream (Server-Sent Events)",
            "/analyze  (type a headline)",
            "/inject   (demo only)",
        ],
    )
    box(
        ax,
        0.40,
        0.10,
        0.27,
        0.27,
        "Module A: Tactical Index Rebalancer",
        [
            "Subscribes to entity sentiment",
            "w = w0 * exp(k * s * c), deadband, 2-12% bounds",
            "Turnover cap, 5 bps costs",
            "Daily backtest (t -> t+1, no look-ahead)",
            "vs equal-weight and naive sign rule; IC",
        ],
        fill=FILL_MOD,
    )
    box(
        ax,
        0.71,
        0.10,
        0.28,
        0.27,
        "Module B: Strategic Stress Testing",
        [
            "Subscribes to event class + impact",
            "Trigger: impact >= 8, conf >= 0.6, >= 2 sources",
            "Shocks measured on pre-2021 analogues",
            "Synthetic book: loans, bonds, IRS, FX, CDS,",
            "options, TRS, equity; dEL; CET1 (CRE20)",
        ],
        fill=FILL_MOD,
    )
    box(
        ax,
        0.01,
        0.10,
        0.34,
        0.27,
        "Dashboard (Streamlit) + Evaluation",
        [
            "Signal Monitor | Module A | Module B | Model Quality",
            "Renders from cached outputs when the API is down",
            "riskpulse eval all -> reports/metrics.json",
            "Sentiment vs VADER / LM; events vs baselines",
            "Every quoted number traces to metrics.json",
        ],
    )
    arrow(ax, 0.23, 0.75, 0.27, 0.75)
    arrow(ax, 0.47, 0.75, 0.51, 0.75, "documents")
    arrow(ax, 0.76, 0.75, 0.80, 0.75, "signals")
    arrow(ax, 0.84, 0.60, 0.56, 0.37, "subscribe: SSE or store", label_xy=(0.80, 0.47))
    arrow(ax, 0.90, 0.60, 0.86, 0.37)
    arrow(ax, 0.40, 0.23, 0.35, 0.23, "weights, P&L")
    arrow(ax, 0.85, 0.12, 0.18, 0.12, "stress results", rad=-0.12, label_xy=(0.52, 0.025))
    out = repo_root() / "docs" / "architecture.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(out)


if __name__ == "__main__":
    main()
