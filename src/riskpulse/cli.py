"""RiskPulse command-line interface (typer).

Commands are thin wrappers; the logic lives in the sub-packages. Commands that
are not built yet exit with a clear "not implemented" message and a non-zero
code, so nothing silently pretends to work.
"""

from __future__ import annotations

from enum import StrEnum
from typing import NoReturn

import typer

from riskpulse import __version__

app = typer.Typer(
    name="riskpulse",
    help="RiskPulse: AI/NLP risk engine with an index rebalancer and a stress tester.",
    no_args_is_help=True,
    add_completion=False,
)


class RunMode(StrEnum):
    replay = "replay"
    live = "live"
    batch = "batch"


class EvalTarget(StrEnum):
    all = "all"
    sentiment = "sentiment"
    events = "events"
    impact = "impact"
    pipeline = "pipeline"
    moduleA = "moduleA"
    moduleB = "moduleB"


def _not_yet(what: str, phase: int) -> NoReturn:
    typer.secho(f"'{what}' is not implemented yet (planned for build phase {phase}).", fg="yellow")
    raise typer.Exit(code=2)


@app.callback(invoke_without_command=True)
def main(
    version: bool = typer.Option(False, "--version", help="Print version and exit."),
) -> None:
    if version:
        typer.echo(f"riskpulse {__version__}")
        raise typer.Exit()


@app.command()
def ingest(
    source: str = typer.Option(
        "all", help="Source adapter: gdelt, newsapi, kaggle_news, kaggle_tweets, all."
    ),
) -> None:
    """Pull documents from a source into the document store."""
    _not_yet("ingest", 1)


@app.command()
def process(mode: RunMode = typer.Option(RunMode.batch, help="Run mode.")) -> None:
    """Run the NLP engine over stored documents and emit signals."""
    _not_yet("process", 4)


@app.command()
def replay(
    speed: float = typer.Option(60.0, help="Seconds of wall time per simulated market day."),
) -> None:
    """Stream the saved replay feed through the engine."""
    _not_yet("replay", 4)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8000),
) -> None:
    """Start the FastAPI signal server (REST + SSE)."""
    _not_yet("serve", 4)


@app.command()
def demo(
    fast: bool = typer.Option(False, "--fast", help="Use precomputed signals; no model downloads."),
) -> None:
    """One-command demo: engine + API + dashboard."""
    _not_yet("demo", 7)


@app.command(name="eval")
def eval_(target: EvalTarget = typer.Argument(EvalTarget.all)) -> None:
    """Regenerate evaluation metrics into reports/."""
    _not_yet(f"eval {target.value}", 2)


@app.command()
def backtest() -> None:
    """Module A: historical daily backtest of the sentiment-tilted index."""
    _not_yet("backtest", 5)


@app.command()
def stress(
    scenario: str = typer.Option(None, help="Scenario name from configs/scenarios.yaml."),
) -> None:
    """Module B: run a stress test on the synthetic wholesale book."""
    _not_yet("stress", 6)
