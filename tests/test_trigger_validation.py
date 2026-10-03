"""Trigger validation scoring (D-062): hit window = the trigger session and the next 2."""

from riskpulse.eval.trigger_validation import scores


def test_precision_and_recall_windows() -> None:
    stress = {5, 20}
    # 3 -> 5 within +2 (hit); 5 -> same day (hit); 10 -> no stress in 10..12 (miss)
    prec, rec = scores({3, 5, 10}, stress, 30)
    assert prec == 2 / 3
    assert rec == 1 / 2  # day 5 covered (triggers at 3 and 5), day 20 not (no trigger in 18..20)


def test_trigger_after_stress_day_is_not_a_hit() -> None:
    prec, rec = scores({6}, {5}, 30)  # the trigger came a session after the stress day
    assert prec == 0 and rec == 0
