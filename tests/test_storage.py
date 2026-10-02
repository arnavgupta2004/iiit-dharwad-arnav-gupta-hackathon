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
