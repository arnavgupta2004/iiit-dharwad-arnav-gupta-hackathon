"""P1: fine-tune FinBERT on the HF tweet-sentiment TRAIN split only (GATE C item 6).

Protocol: 90% of `train` for training, 10% of `train` as the dev set for early stopping; the HF
`valid` split (our test set) is touched once, at the end. CPU only, hard 45-minute wall-clock
budget. The fine-tuned model is kept only if its test macro-F1 beats FinBERT's 0.661
(train-tuned band); the argmax figure (0.668) is reported alongside as the stricter bar.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from riskpulse.common.config import data_path, load_config
from riskpulse.common.logging import get_logger
from riskpulse.common.metrics import load_metrics, update_metrics
from riskpulse.eval.sentiment_eval import CLASSES, LABEL_MAP, load_split

log = get_logger()
BUDGET_SECONDS = 45 * 60
OUT_DIR = data_path("processed", "models", "finbert_tweets_ft")
HP = {"lr": 2e-5, "batch": 32, "max_len": 64, "epochs": 3, "warmup_frac": 0.1, "seed": 20261002}


def _train() -> tuple:
    """Fine-tune on 90% of train with the remaining 10% as dev; returns the best-dev model.

    Returns (model, tokenizer, predict_fn, base_name, n_train, n_dev, history, stop_reason).
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    torch.manual_seed(HP["seed"])
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    name = load_config("app")["sentiment"]["model"]
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name)
    label2id = {v.lower(): k for k, v in model.config.id2label.items()}

    train_all = load_split("train")
    rng = np.random.default_rng(HP["seed"])
    dev_mask = rng.random(len(train_all)) < 0.10
    tr, dev = train_all[~dev_mask], train_all[dev_mask]

    def tensors(df: pd.DataFrame) -> TensorDataset:
        enc = tok(
            df["text"].tolist(),
            padding="max_length",
            truncation=True,
            max_length=HP["max_len"],
            return_tensors="pt",
        )
        y = torch.tensor([label2id[g] for g in df["gold"]])
        return TensorDataset(enc["input_ids"], enc["attention_mask"], y)

    def predict(df: pd.DataFrame) -> np.ndarray:
        model.eval()
        ds = tensors(df)
        preds = []
        with torch.no_grad():
            for ids, mask, _ in DataLoader(ds, batch_size=128):
                preds.append(model(input_ids=ids, attention_mask=mask).logits.argmax(-1).numpy())
        id2lab = {i: lab.lower() for i, lab in model.config.id2label.items()}
        return np.array([id2lab[i] for i in np.concatenate(preds)])

    loader = DataLoader(tensors(tr), batch_size=HP["batch"], shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=HP["lr"])
    total = HP["epochs"] * len(loader)
    sched = get_linear_schedule_with_warmup(opt, int(HP["warmup_frac"] * total), total)
    t0 = time.perf_counter()
    best_dev, best_state, history, stopped = -1.0, None, [], "completed"
    for epoch in range(HP["epochs"]):
        model.train()
        for step, (ids, mask, y) in enumerate(loader):
            if time.perf_counter() - t0 > BUDGET_SECONDS:
                stopped = f"time budget reached in epoch {epoch + 1}, step {step}"
                break
            loss = model(input_ids=ids, attention_mask=mask, labels=y).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
        dev_f1 = float(f1_score(dev["gold"], predict(dev), labels=CLASSES, average="macro"))
        history.append(
            {
                "epoch": epoch + 1,
                "dev_macro_f1": round(dev_f1, 4),
                "elapsed_min": round((time.perf_counter() - t0) / 60, 1),
            }
        )
        log.info(f"fine-tune epoch {epoch + 1}: dev macro-F1 {dev_f1:.4f}")
        if dev_f1 > best_dev:
            best_dev = dev_f1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if stopped != "completed":
            break
    model.load_state_dict(best_state)
    return model, tok, predict, name, len(tr), len(dev), history, stopped


def build(out_dir: Path = OUT_DIR) -> Path:
    """Rebuild the fine-tuned weights from the train split only (no test evaluation)."""
    model, tok, *_ = _train()
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    log.info(f"fine-tuned sentiment model written to {out_dir}")
    return out_dir


def run() -> dict:
    """Train, evaluate once on the held-out split, keep the weights only if they beat the bar."""
    model, tok, predict, name, n_tr, n_dev, history, stopped = _train()
    test = load_split("valid")
    test_pred = predict(test)
    test_f1 = round(float(f1_score(test["gold"], test_pred, labels=CLASSES, average="macro")), 4)
    base = load_metrics()["sentiment"]["results"]["finbert"]
    bar, strict = base["train_tuned"]["macro_f1"], base["argmax"]["macro_f1"]
    keep = test_f1 > bar
    if keep:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(OUT_DIR)
        tok.save_pretrained(OUT_DIR)
    payload = {
        "base_model": name,
        "train_rows": int(n_tr),
        "dev_rows_from_train": int(n_dev),
        "test_split": "valid (touched once)",
        "hyperparameters": HP,
        "budget_minutes": BUDGET_SECONDS / 60,
        "stop_reason": stopped,
        "history": history,
        "test_macro_f1": test_f1,
        "bar_finbert_train_tuned": bar,
        "finbert_argmax_for_reference": strict,
        "kept": bool(keep),
        "decision": "kept (beats FinBERT 0.661)" if keep else "discarded (does not beat FinBERT)",
        "label_map": LABEL_MAP,
    }
    update_metrics("sentiment_finetune", payload, script="riskpulse eval finetune")
    return payload
