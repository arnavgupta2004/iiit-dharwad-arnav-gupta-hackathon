"""Rebuild the fine-tuned sentiment weights locally (D-043), for when the Hub copy is unavailable.

Same protocol as `riskpulse eval finetune` (90% of the HF tweet-sentiment train split, 10% of train
as dev for early stopping, CPU, 45-minute budget), but it never touches the test split. About 25 CPU
minutes.

    python scripts/build_finetuned_sentiment.py
"""

from __future__ import annotations

from riskpulse.eval.finetune_sentiment import build

if __name__ == "__main__":
    print(f"written to {build()}")
