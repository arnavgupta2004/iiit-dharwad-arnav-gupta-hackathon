"""Create data/gold/entity_check.csv: 100 random (headline, ticker) links, stratified by ticker.

News headlines only (title-linked GDELT items), excluding texts in the gold labelling file.
Arnav marks `correct` as y / n. Never used for training or tuning.

    python scripts/make_entity_check.py
"""

from __future__ import annotations

import pandas as pd

from riskpulse.common.config import data_path, load_config

SEED = 20261002
N = 100


def main() -> None:
    tickers = list(load_config("universe")["tickers"])
    names = {t: s["name"] for t, s in load_config("universe")["tickers"].items()}
    m = pd.read_parquet(data_path("processed", "mentions.parquet"))
    m = m[(m["source"] == "gdelt") & (m["ticker"].isin(tickers))].drop_duplicates(
        ["title", "ticker"]
    )
    gold = set(pd.read_csv(data_path("gold", "to_label.csv"))["text"])
    m = m[~m["title"].isin(gold)]
    per = N // len(tickers)
    picks = [g.sample(min(per, len(g)), random_state=SEED) for _, g in m.groupby("ticker")]
    out = pd.concat(picks)
    if len(out) < N:  # top up from the remaining pool if a ticker had fewer than `per`
        rest = m.drop(out.index)
        out = pd.concat([out, rest.sample(N - len(out), random_state=SEED)])
    out = out.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    res = pd.DataFrame(
        {
            "check_id": [f"E{i:03d}" for i in range(1, len(out) + 1)],
            "doc_id": out["doc_id"],
            "published_at": out["published_at"].astype(str),
            "ticker": out["ticker"],
            "company": out["ticker"].map(names),
            "headline": out["title"],
            "correct": "",
            "notes": "",
        }
    )
    path = data_path("gold", "entity_check.csv")
    res.to_csv(path, index=False)
    print(f"Wrote {len(res)} links to {path}")
    print(res["ticker"].value_counts().to_dict())


if __name__ == "__main__":
    main()
