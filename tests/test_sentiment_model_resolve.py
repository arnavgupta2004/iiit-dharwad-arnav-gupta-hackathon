"""Sentiment weights resolution: Hub -> local rebuild -> base FinBERT, with the origin reported."""

import pytest

from riskpulse.engine import sentiment as sm


def _cfg(variant: str, hub_repo=None, local_dir: str = "ft") -> dict:
    return {
        "sentiment": {
            "model": "ProsusAI/finbert",
            "variant": variant,
            "finetuned": {"hub_repo": hub_repo, "local_dir": local_dir},
        }
    }


@pytest.fixture
def patched(monkeypatch, tmp_path):
    def apply(cfg: dict):
        monkeypatch.setattr(sm, "load_config", lambda name: cfg)
        monkeypatch.setattr(sm, "repo_root", lambda: tmp_path)
        monkeypatch.delenv("RISKPULSE_BUILD_SENTIMENT", raising=False)
        sm.resolve_model.cache_clear()

    yield apply
    sm.resolve_model.cache_clear()


def test_base_variant(patched) -> None:
    patched(_cfg("finbert"))
    assert sm.resolve_model() == ("ProsusAI/finbert", "base")


def test_finetuned_falls_back_to_base_when_nothing_available(patched) -> None:
    patched(_cfg("finetuned"))
    assert sm.resolve_model() == ("ProsusAI/finbert", "base_fallback")


def test_finetuned_uses_local_copy(patched, tmp_path) -> None:
    d = tmp_path / "ft"
    d.mkdir()
    (d / "config.json").write_text("{}")
    (d / "model.safetensors").write_bytes(b"x")
    patched(_cfg("finetuned"))
    assert sm.resolve_model() == (str(d), "local")


def test_hub_failure_falls_through_to_local(patched, tmp_path, monkeypatch) -> None:
    import huggingface_hub

    def boom(repo_id):
        raise OSError("offline")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", boom)
    d = tmp_path / "ft"
    d.mkdir()
    (d / "config.json").write_text("{}")
    (d / "model.safetensors").write_bytes(b"x")
    patched(_cfg("finetuned", hub_repo="someone/missing-repo"))
    assert sm.resolve_model()[1] == "local"


def test_unknown_variant_rejected(patched) -> None:
    patched(_cfg("vader"))
    with pytest.raises(ValueError):
        sm.resolve_model()
