from typer.testing import CliRunner

from riskpulse import __version__
from riskpulse.cli import app

runner = CliRunner()


def test_help_lists_all_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ["ingest", "process", "replay", "serve", "demo", "eval", "backtest", "stress"]:
        assert cmd in result.output


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output
