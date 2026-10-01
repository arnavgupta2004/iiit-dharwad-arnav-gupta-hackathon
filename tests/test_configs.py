"""Cross-config consistency checks: configs must agree with each other and the schemas."""

import pytest

from riskpulse.common.config import CONFIG_NAMES, load_config
from riskpulse.common.schemas import EventClass

BRIEF_CLASSES = {
    "GEOPOLITICAL",
    "MACROECONOMIC",
    "CREDIT_EVENT",
    "MERGER_ACQUISITION",
    "PRODUCT_LAUNCH",
}


@pytest.mark.parametrize("name", CONFIG_NAMES)
def test_every_config_loads(name: str) -> None:
    assert load_config(name)


def test_universe_has_20_unique_tickers_with_sectors() -> None:
    tickers = load_config("universe")["tickers"]
    assert len(tickers) == 20
    sectors = set(load_config("mcc_sector_map")["sector_etf"])
    for t, spec in tickers.items():
        assert spec["sector"] in sectors, t
        assert spec["aliases"], t
        assert f"${t}" in spec["cashtags"], t
        assert set(spec["ambiguous"]) <= set(spec["aliases"] + spec["brands"]), t


def test_taxonomy_matches_schema_and_brief() -> None:
    classes = set(load_config("taxonomy")["classes"])
    assert classes == {c.value for c in EventClass}
    assert BRIEF_CLASSES <= classes


def test_impact_priors_cover_all_classes_and_weights_sum_to_one() -> None:
    cfg = load_config("impact")
    assert set(cfg["type_prior"]) == {c.value for c in EventClass}
    assert all(0 <= v <= 1 for v in cfg["type_prior"].values())
    assert sum(cfg["weights"].values()) == pytest.approx(1.0)


def test_scenario_mapping_covers_all_classes_and_points_to_analogues() -> None:
    cfg = load_config("scenarios")
    assert set(cfg["mapping"]) == {c.value for c in EventClass}
    for rules in cfg["mapping"].values():
        for target in rules.values():
            assert target is None or target in cfg["analogues"]


def test_mcc_map_covers_dataset_codes() -> None:
    cfg = load_config("mcc_sector_map")
    assert len(cfg["mcc_to_sector"]) == 109
    assert set(cfg["mcc_to_sector"].values()) <= set(cfg["sector_etf"])


def test_moduleB_mix_and_severity() -> None:
    cfg = load_config("moduleB")
    assert sum(cfg["portfolio"]["mix"].values()) == pytest.approx(1.0)
    assert cfg["trigger"]["min_impact"] == 8
    assert set(cfg["severity_multiplier"]) == {8, 9, 10}
