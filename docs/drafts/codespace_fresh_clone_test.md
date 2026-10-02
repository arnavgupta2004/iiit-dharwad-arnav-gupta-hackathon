# Fresh-clone test in a GitHub Codespace (empty model cache)

Goal: show that a clean machine with no local caches runs the **reported** system. Every model must load from the
repo or the Hugging Face Hub, with no `FALLBACK IN USE` lines, `demo --fast` must work, and `/analyze` must answer.

## 0. Create the Codespace
GitHub → the repo → **Code** → **Codespaces** → **⋯** → **New with options** → branch `main`, machine type
**4-core / 16 GB** (2-core works but model loading is slow). A new Codespace has an empty Hugging Face cache. The
repo is already cloned at `/workspaces/iiit-dharwad-arnav-gupta-hackathon`.

## 1. Python 3.11 (the package requires >=3.11,<3.12)
```bash
cd /workspaces/iiit-dharwad-arnav-gupta-hackathon
python3.11 --version || (pip install -q uv && uv python install 3.11)
```

## 2. Install exactly as the README says
```bash
PY=$(command -v python3.11 || uv python find 3.11)
$PY -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
On Linux, pip pulls the default torch wheel with CUDA libraries (a few GB). It works on CPU but takes several
minutes. Note the install time.

## 3. Confirm the cache is empty and nothing points to external storage
```bash
ls ~/.cache/huggingface/hub 2>/dev/null | wc -l      # expect 0
test -f .env && echo ".env PRESENT (unexpected)" || echo "no .env (expected)"
python -c "import os, riskpulse; print('HF_HUB_CACHE =', os.environ.get('HF_HUB_CACHE'))"   # expect None
```

## 4. Tests
```bash
pytest -q 2>&1 | tail -3
```
Expect all passed or skipped. Model-dependent tests are skipped when the base FinBERT isn't cached yet.

## 5. `demo --fast` (precomputed outputs, no model downloads)
```bash
python -m riskpulse demo --fast
```
Codespaces forwards ports 8000 and 8501 automatically. Open the **Ports** tab, then the globe icon for **8501**
(dashboard) and **8000** + `/docs` (API). Check all four dashboard pages render, then press Ctrl+C.

## 6. Full mode: `serve --full` (downloads models on first start)
```bash
time python -m riskpulse serve --full --port 8000 2>&1 | tee serve.log
```
Wait until `Uvicorn running on http://127.0.0.1:8000` appears and note the time. In a **second terminal**:
```bash
cd /workspaces/iiit-dharwad-arnav-gupta-hackathon && source .venv/bin/activate
grep "model origin" serve.log
grep -c "FALLBACK IN USE" serve.log          # expect 0
curl -s http://127.0.0.1:8000/health
time curl -s -X POST http://127.0.0.1:8000/analyze -H 'Content-Type: application/json' \
  -d '{"text": "JPMorgan beats estimates while Bank of America misses"}' | python -m json.tool | head -40
```

### What to check
| Check | Expected |
|---|---|
| `model origin | sentiment_news` | `base:ProsusAI/finbert` (downloaded from the Hub) |
| `model origin | sentiment_social` | `hub:…/models--arnavguptas--riskpulse-finbert-tweets/…` |
| `model origin | event_classifier` | `repo:data/trained/event_clf_round2.pkl` |
| `model origin | impact_company` | `repo:data/trained/impact_v2.json` |
| `model origin | embeddings` | `hub:sentence-transformers/all-MiniLM-L6-v2` |
| `FALLBACK IN USE` lines | **0** |
| `/analyze` | JSON with `sentiment_score`, `event_class`, `entities` (each with its own `sentiment_score`, `impact_score` and `drivers`; JPM and BAC should get different clause-level scores) and `model_versions` (`news:finbert@…; social:finbert-tweets-ft`); well under 1 s after start-up |

## 7. Optional: full demo with the dashboard and live replay
Stop `serve` (Ctrl+C), then:
```bash
python -m riskpulse demo
```
Open port 8501. The Signal Monitor should stream the Feb–Mar 2022 replay; "Analyze a headline" calls `/analyze`.

## 8. Send back
Install time, the `model origin` lines, the FALLBACK count, the `/analyze` output and timing, and a screenshot of the
dashboard on 8501. Then delete the Codespace (it uses your free-hours quota).
