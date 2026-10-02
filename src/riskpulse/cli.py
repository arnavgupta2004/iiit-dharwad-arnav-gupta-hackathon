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
    linking = "linking"
    impact_v2 = "impact_v2"


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
    full: bool = typer.Option(False, "--full", help="Load models: enables /analyze and /inject."),
    replay_start: str = typer.Option(None, help="With --full: replay the feed from this date."),
    replay_end: str = typer.Option(None, help="With --full: replay end date (exclusive)."),
    seconds_per_day: float = typer.Option(None, help="Replay pace (default from config)."),
) -> None:
    """Start the FastAPI signal server (REST + SSE)."""
    import uvicorn

    from riskpulse.api.app import create_app

    live, replay = None, None
    if full:
        from riskpulse.api.live import LiveEngine

        live = LiveEngine()
        if replay_start and replay_end:
            replay = (replay_start, replay_end, seconds_per_day or 60.0)
    application = create_app("full" if full else "fast", live=live, replay=replay)
    uvicorn.run(application, host=host, port=port, log_level="info")


@app.command()
def demo(
    fast: bool = typer.Option(False, "--fast", help="Use precomputed signals; no model downloads."),
    replay_start: str = typer.Option("2022-02-14", help="Full mode: replay window start."),
    replay_end: str = typer.Option("2022-03-12", help="Full mode: replay window end (exclusive)."),
    seconds_per_day: float = typer.Option(60.0, help="Full mode: wall seconds per market day."),
    api_port: int = typer.Option(8000),
    dashboard_port: int = typer.Option(8501),
) -> None:
    """One-command demo: signal API + Streamlit dashboard (Ctrl+C stops both)."""
    import os
    import subprocess
    import sys
    import time

    from riskpulse.common.config import repo_root

    env = {**os.environ, "RISKPULSE_API": f"http://127.0.0.1:{api_port}"}
    api_cmd = [sys.executable, "-m", "riskpulse", "serve", "--port", str(api_port)]
    if not fast:
        api_cmd += [
            "--full",
            "--replay-start",
            replay_start,
            "--replay-end",
            replay_end,
            "--seconds-per-day",
            str(seconds_per_day),
        ]
    app_path = repo_root() / "src" / "riskpulse" / "dashboard" / "app.py"
    dash_cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.headless",
        "true",
        "--server.port",
        str(dashboard_port),
        "--theme.base",
        "light",
    ]
    procs = [subprocess.Popen(api_cmd, env=env), subprocess.Popen(dash_cmd, env=env)]
    typer.secho(
        f"RiskPulse demo ({'fast: cached outputs' if fast else 'full: live models + replay'})\n"
        f"  Dashboard: http://localhost:{dashboard_port}\n  API:       http://127.0.0.1:{api_port}/docs",
        fg="green",
    )
    try:
        while all(p.poll() is None for p in procs):
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            p.terminate()


@app.command(name="eval")
def eval_(target: EvalTarget = typer.Argument(EvalTarget.all)) -> None:
    """Regenerate evaluation metrics into reports/."""
    from riskpulse.eval import registry

    ran = registry.run(target.value)
    typer.echo(f"Updated reports/metrics.json sections: {', '.join(ran)}")


@app.command()
def backtest() -> None:
    """Module A: daily backtest of the sentiment-tilted index -> reports/."""
    from riskpulse.moduleA.run import run

    perf = run()["performance"]
    for name, p in perf.items():
        typer.echo(f"{name:26s} cum {p['cumulative_return']:+.2%}  sharpe {p['sharpe_rf0']}")


@app.command()
def stress(
    replay: bool = typer.Option(False, "--replay", help="Trigger from stored event signals."),
    event_class: str = typer.Option(None, help="What-if: event class, e.g. CREDIT_EVENT."),
    impact: int = typer.Option(9, help="What-if: impact score 8-10."),
    build_portfolio: bool = typer.Option(False, "--build-portfolio", help="Regenerate the book."),
) -> None:
    """Module B: stress the synthetic wholesale book (trigger replay or what-if)."""
    if build_portfolio:
        from riskpulse.moduleB.portfolio import generate_portfolio

        cps, pos = generate_portfolio()
        typer.echo(f"Book: {len(cps)} counterparties, {len(pos)} positions")
    if replay:
        from riskpulse.moduleB.run import replay_triggers

        summ = replay_triggers()
        typer.echo(
            f"{summ['n_triggers_fired']} triggers fired; {summ['n_stress_runs']} stress runs"
        )
    if event_class:
        from riskpulse.moduleB.run import run_named

        r = run_named(event_class, impact)
        c = r["capital"]
        typer.echo(
            f"{r['scenario']['name']}: impact {r['total_impact'] / 1e6:,.1f}m USD "
            f"({r['total_impact_pct']:.2%}); CET1 {c['cet1_ratio_before']:.1%} -> "
            f"{c['cet1_ratio_after']:.2%}"
        )
