"""Download the Hugging Face labelled financial-tweet datasets (CSV files, MIT licence).

- zeroshot/twitter-financial-news-sentiment: sentiment evaluation set (train/valid only;
  there is no test split, so `valid` is our test set and is never trained on).
- zeroshot/twitter-financial-news-topic: 20 topic labels, mapped to our event taxonomy
  as weak labels (configs/taxonomy.yaml `hf_topic_map`).

    python scripts/download_hf.py
"""

from __future__ import annotations

from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "_downloads" / "hf"
BASE = "https://huggingface.co/datasets/{repo}/resolve/main/{file}"

FILES: dict[str, list[str]] = {
    "zeroshot/twitter-financial-news-sentiment": [
        "sent_train.csv",
        "sent_valid.csv",
        "sent_dataset_meta.txt",
    ],
    "zeroshot/twitter-financial-news-topic": [
        "topic_train.csv",
        "topic_valid.csv",
        "topic_dataset_meta.txt",
    ],
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for repo, files in FILES.items():
        for f in files:
            dest = OUT / f
            if dest.exists():
                print(f"skip  {repo}/{f}")
                continue
            r = requests.get(BASE.format(repo=repo, file=f), timeout=60)
            r.raise_for_status()
            dest.write_bytes(r.content)
            print(f"done  {repo}/{f} ({len(r.content) / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
