"""Maps `riskpulse eval <target>` to eval functions; unbuilt targets are reported, not faked."""

from __future__ import annotations

from collections.abc import Callable

from riskpulse.common.logging import get_logger

log = get_logger()


def _sentiment() -> None:
    from riskpulse.eval.sentiment_eval import run

    run()


def _events() -> None:
    from riskpulse.engine.event_training import evaluate

    evaluate()  # the saved model; training is `riskpulse train events`


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


def _impact() -> None:
    from riskpulse.eval.impact_eval import run

    run()


def _impact_v2() -> None:
    from riskpulse.eval.impact_v2 import evaluate

    evaluate()  # the saved model; training is `riskpulse train impact_v2`


def _pvr() -> None:
    from riskpulse.eval.predicted_vs_realised import run

    run()


def _impact_market() -> None:
    from riskpulse.eval.impact_market_check import run

    run()


def _sentiment_gold() -> None:
    from riskpulse.eval.sentiment_gold import run

    run()


def _sentiment_news_pooled() -> None:
    from riskpulse.eval.sentiment_gold import run_news_pooled

    run_news_pooled()


def _trigger_validation() -> None:
    from riskpulse.eval.trigger_validation import run

    run()


def _finetune() -> None:
    from riskpulse.eval.finetune_sentiment import run

    run()


EVALS: dict[str, Callable[[], None]] = {
    "predicted_vs_realised": _pvr,
    "finetune": _finetune,
    "trigger_validation": _trigger_validation,
    "sentiment_gold": _sentiment_gold,
    "sentiment_news_pooled": _sentiment_news_pooled,
    "impact_market_check": _impact_market,
    "impact_v2": _impact_v2,
    "impact": _impact,
    "linking": _linking,
    "sentiment": _sentiment,
    "events": _events,
    "pipeline": _pipeline,
    "moduleA": _module_a,
    "moduleB": _module_b,
}


# Dependency order for `eval all`. The fine-tune is a one-off, time-boxed experiment and is run
# explicitly (`riskpulse eval finetune`), not as part of `all`.
ALL_ORDER = [
    "sentiment",
    "sentiment_gold",
    "events",
    "linking",
    "impact",
    "impact_v2",
    "impact_market_check",
    "moduleA",
    "moduleB",
    "predicted_vs_realised",
    "trigger_validation",
    "pipeline",
]


def run(target: str) -> list[str]:
    names = ALL_ORDER if target == "all" else [target]
    ran = []
    for n in names:
        if n not in EVALS:
            log.warning(f"eval '{n}' not implemented yet; skipped")
            continue
        log.info(f"Running eval: {n}")
        EVALS[n]()
        ran.append(n)
    return ran
