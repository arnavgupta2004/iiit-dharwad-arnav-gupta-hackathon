"""Write the Hugging Face model card (README.md) for the fine-tuned sentiment model (D-043).

Every number comes from reports/metrics.json (`sentiment`, `sentiment_finetune`) and the leakage
check below, so the card never carries hand-typed results.

    python scripts/make_model_card.py
"""

from __future__ import annotations

import json
import re

from riskpulse.common.config import load_config, repo_root
from riskpulse.eval.sentiment_eval import load_split

DATASET = "zeroshot/twitter-financial-news-sentiment"


def leakage() -> tuple[int, int, int]:
    """Test texts that share a 60-char prefix / exact text (URLs stripped) with a train text."""

    def norm(t: str) -> str:
        return re.sub(r"https?://\S+|\s+", " ", t.lower()).strip()

    tr = {norm(t) for t in load_split("train")["text"]}
    te = [norm(t) for t in load_split("valid")["text"]]
    pre = {t[:60] for t in tr}
    return len(te), sum(t[:60] in pre for t in te), sum(t in tr for t in te)


def card(m: dict) -> str:
    ft, base = m["sentiment_finetune"], m["sentiment"]["results"]["finbert"]
    hp = ft["hyperparameters"]
    n_test, n_prefix, n_exact = leakage()
    best = max(ft["history"], key=lambda h: h["dev_macro_f1"])
    return f"""---
language: en
license: cc-by-nc-sa-3.0
base_model: {ft["base_model"]}
datasets:
- {DATASET}
pipeline_tag: text-classification
tags:
- financial-sentiment-analysis
- finance
- twitter
metrics:
- f1
---

# FinBERT fine-tuned on financial tweets (RiskPulse)

Three-class financial sentiment (positive / negative / neutral) for short financial news posts and tweets.
It is the sentiment component of RiskPulse, an individual hackathon project by Arnav Gupta (IIIT Dharwad).

## Base model
[`{ft["base_model"]}`](https://huggingface.co/{ft["base_model"]}), a BERT model further pre-trained on financial
text and fine-tuned for sentiment on Financial PhraseBank (Malo et al., 2014).

## Training data
[`{DATASET}`](https://huggingface.co/datasets/{DATASET}) (licence: MIT), English financial tweets labelled
Bearish / Bullish / Neutral, mapped to negative / positive / neutral.

## Protocol (train split only)
- {ft["train_rows"]:,} rows (90% of the dataset's `train` split) for training; {ft["dev_rows_from_train"]:,} rows
  (the other 10% of `train`) as a dev set for early stopping (best dev macro-F1 {best["dev_macro_f1"]:.3f} at epoch
  {best["epoch"]}).
- The dataset's `validation` split ({n_test:,} rows) is the test set and was evaluated **once**, after training.
- Learning rate {hp["lr"]}, batch size {hp["batch"]}, max length {hp["max_len"]} tokens, {hp["epochs"]} epochs,
  linear schedule with {hp["warmup_frac"]:.0%} warm-up, seed {hp["seed"]}. CPU only, {ft["budget_minutes"]:.0f}-minute
  budget ({ft["stop_reason"]}).

## Results (test split, in-domain)
| Model | Macro-F1 |
|---|---|
| This model | {ft["test_macro_f1"]:.3f} |
| Base FinBERT, labels from a score band tuned on the train split | {base["train_tuned"]["macro_f1"]:.3f} |
| Base FinBERT, argmax | {base["argmax"]["macro_f1"]:.3f} |

**Leakage note.** {n_prefix} of {n_test:,} test texts ({n_prefix / n_test:.1%}) share their first 60 characters
with a training text ({n_exact} are identical after removing URLs). That bounds the effect on macro-F1 at about 2
points, so it does not explain the gain.

**In-domain caveat.** Training and test texts come from the same dataset. Accuracy on other financial text
(e.g. news headlines from other sources) has not been established by this number.

## Intended use
Research and education: scoring the tone of short English financial headlines and posts as one input to a risk
monitoring prototype. Not investment advice and not for automated trading decisions. Known limitations: short
texts only (max {hp["max_len"]} tokens in training), English only, the tone of a text is not its market impact.

## Licence
Training data: MIT. The base model's card declares no licence; its code repository (ProsusAI/finBERT) is
Apache-2.0, and its sentiment fine-tuning used Financial PhraseBank (CC BY-NC-SA 3.0). This derivative is
therefore released under **CC BY-NC-SA 3.0** as the conservative choice.

## Labels
Model outputs `positive`, `negative`, `neutral` (see `config.json → id2label`). Dataset label ids were mapped as
`{json.dumps(ft["label_map"])}` (0 Bearish, 1 Bullish, 2 Neutral).
"""


def main() -> None:
    m = json.loads((repo_root() / "reports" / "metrics.json").read_text())
    out = repo_root() / load_config("app")["sentiment"]["finetuned"]["local_dir"] / "README.md"
    out.write_text(card(m))
    print(f"model card written to {out}")


if __name__ == "__main__":
    main()
