"""Maps `riskpulse eval <target>` to eval functions; unbuilt targets are reported, not faked."""

from __future__ import annotations

from collections.abc import Callable

from riskpulse.common.logging import get_logger

log = get_logger()


def _sentiment() -> None:
    from riskpulse.eval.sentiment_eval import run

    run()


def _events() -> None:
    from riskpulse.engine.event_training import train_and_evaluate

    train_and_evaluate()


def _pipeline() -> None:
    from riskpulse.eval.pipeline_bench import run

    run()


def _module_a() -> None:
    from riskpulse.moduleA.run import run

    run()


def _module_b() -> None:
    from riskpulse.moduleB.run import replay_triggers

    replay_triggers()


def _linking() -> None:
    from riskpulse.eval.linking_eval import run

    run()


EVALS: dict[str, Callable[[], None]] = {
    "linking": _linking,
    "sentiment": _sentiment,
    "events": _events,
    "pipeline": _pipeline,
    "moduleA": _module_a,
    "moduleB": _module_b,
}


def run(target: str) -> list[str]:
    names = list(EVALS) if target == "all" else [target]
    ran = []
    for n in names:
        if n not in EVALS:
            log.warning(f"eval '{n}' not implemented yet; skipped")
            continue
        log.info(f"Running eval: {n}")
        EVALS[n]()
        ran.append(n)
    return ran
