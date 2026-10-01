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


EVALS: dict[str, Callable[[], None]] = {
    "sentiment": _sentiment,
    "events": _events,
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
