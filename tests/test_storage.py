"""External storage (D-056): a configured but missing drive stops with a clear error."""

import pytest

from riskpulse import _storage


def test_missing_external_root_raises(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("RISKPULSE_EXTERNAL_ROOT", str(tmp_path / "unplugged-drive"))
    with pytest.raises(RuntimeError, match="External storage not found"):
        _storage.configure()


def test_dotenv_parsing(tmp_path) -> None:
    env = tmp_path / ".env"
    env.write_text('# comment\nRISKPULSE_EXTERNAL_ROOT="/Volumes/Some Drive/x"\nEMPTY=\n')
    assert _storage._read_dotenv(env) == {
        "RISKPULSE_EXTERNAL_ROOT": "/Volumes/Some Drive/x",
        "EMPTY": "",
    }


def test_unset_root_changes_nothing(monkeypatch, tmp_path) -> None:
    """A fresh clone (no .env, variable unset): default cache paths, no error, no redirection."""
    monkeypatch.setattr(_storage, "REPO_ROOT", tmp_path)  # a repo root with no .env
    monkeypatch.delenv("RISKPULSE_EXTERNAL_ROOT", raising=False)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    (tmp_path / "data" / "processed").mkdir(parents=True)
    _storage.configure()  # must not raise
    import os

    assert "RISKPULSE_EXTERNAL_ROOT" not in os.environ
    assert "HF_HUB_CACHE" not in os.environ
