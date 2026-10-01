"""Shared fixtures. Tests that need downloaded models are skipped when the model is not cached,
so `pytest -q` on a fresh clone never triggers large downloads."""

import pytest


def model_cached(repo_id: str) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache

        return isinstance(try_to_load_from_cache(repo_id, "config.json"), str)
    except Exception:
        return False


requires_finbert = pytest.mark.skipif(
    not model_cached("ProsusAI/finbert"), reason="ProsusAI/finbert not in local HF cache"
)


@pytest.fixture(scope="session")
def finbert():
    from riskpulse.engine.sentiment import FinBertScorer

    return FinBertScorer()


@pytest.fixture(scope="session")
def linker():
    from riskpulse.engine.entities import EntityLinker

    return EntityLinker()
