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
        "gdelt_gkg", help="gdelt_gkg (stream GKG files) or feed (build the replay feed)."
    ),
    start: str = typer.Option(
        "2021-09-30", help="Start date (UTC, inclusive) for historical pulls."
    ),
    end: str = typer.Option("2022-09-30", help="End date (UTC, exclusive) for historical pulls."),
    workers: int = typer.Option(None, help="Parallel download workers (default from config)."),
) -> None:
    """Pull historical documents from a source into data/processed."""
    from datetime import datetime

    if source == "feed":
        from riskpulse.ingestion.replay import build_feed

        stats = build_feed()
        typer.echo(f"Replay feed: {stats['n_docs']} docs {stats['by_source']}")
        return
    if source != "gdelt_gkg":
        _not_yet(f"ingest --source {source}", 1)
    from riskpulse.ingestion.gdelt_gkg import GKGStreamer, gkg_timestamps

    ts = gkg_timestamps(datetime.fromisoformat(start), datetime.fromisoformat(end))
    stats = GKGStreamer().run(ts, workers=workers)
    typer.echo(stats)


@app.command()
def process(
    mode: RunMode = typer.Option(RunMode.batch, help="Run mode."),
    limit: int = typer.Option(None, help="Only the first N feed documents (smoke runs)."),
) -> None:
    """Run the NLP engine over the replay feed and write signals (JSONL + DuckDB)."""
    if mode != RunMode.batch:
        _not_yet(f"process --mode {mode.value}", 4)
    from riskpulse.engine.batch import run_batch
    from riskpulse.engine.pipeline import event_model_exists

    if not event_model_exists():
        typer.echo("Event model missing; training it first (riskpulse eval events).")
        from riskpulse.engine.event_training import train_and_evaluate

        train_and_evaluate()
    typer.echo(run_batch(limit=limit))


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
    from riskpulse.eval import registry

    ran = registry.run(target.value)
    typer.echo(f"Updated reports/metrics.json sections: {', '.join(ran)}")


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
