"""Merchant-level aggregates from the Kaggle transactions dataset (seeds synthetic obligors).

Only merchant fields are read (merchant_id, mcc, merchant_state, amount, date); card/user files
are never touched. Output: data/portfolio/merchant_aggregates.parquet with total volume, counts,
modal MCC/state and the coefficient of variation of monthly flows.
"""

from __future__ import annotations

import zipfile

import numpy as np
import pandas as pd

from riskpulse.common.config import data_path, load_config, repo_root

COLS = ["date", "amount", "merchant_id", "merchant_state", "mcc"]


def build_merchant_aggregates(chunksize: int = 1_000_000) -> pd.DataFrame:
    cfg = load_config("moduleB")["portfolio"]
    monthly: list[pd.DataFrame] = []
    meta: list[pd.DataFrame] = []
    with (
        zipfile.ZipFile(repo_root() / cfg["transactions_zip"]) as zf,
        zf.open("transactions_data.csv") as fh,
    ):
        for chunk in pd.read_csv(fh, usecols=COLS, chunksize=chunksize):
            chunk["amount"] = (
                chunk["amount"].str.replace(r"[$,]", "", regex=True).astype(float).abs()
            )
            chunk["month"] = chunk["date"].str[:7]
            monthly.append(chunk.groupby(["merchant_id", "month"])["amount"].sum().reset_index())
            meta.append(
                chunk.groupby(["merchant_id", "mcc", "merchant_state"], dropna=False)
                .size()
                .rename("n")
                .reset_index()
            )
    m = pd.concat(monthly).groupby(["merchant_id", "month"])["amount"].sum().reset_index()
    stats = m.groupby("merchant_id")["amount"].agg(
        total_volume="sum", n_months="count", monthly_mean="mean", monthly_std="std"
    )
    stats["flow_cv"] = stats["monthly_std"] / stats["monthly_mean"]
    md = pd.concat(meta).groupby(["merchant_id", "mcc", "merchant_state"], dropna=False)["n"].sum()
    md = md.reset_index().sort_values("n", ascending=False)
    modal_mcc = md.groupby("merchant_id")["mcc"].first()
    st = md.dropna(subset=["merchant_state"]).groupby("merchant_id")["merchant_state"].first()
    stats["mcc"] = modal_mcc
    stats["merchant_state"] = st.reindex(stats.index)
    stats["n_tx"] = md.groupby("merchant_id")["n"].sum()
    stats = stats.replace([np.inf, -np.inf], np.nan).reset_index()
    out = data_path("portfolio", "merchant_aggregates.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    stats.to_parquet(out, index=False)
    return stats
