"""Entity-linking edge cases (spec §5.2): ambiguity, cashtags, relevance, MKT routing."""

import pytest

from riskpulse.engine.entities import MKT, EntityLinker


@pytest.fixture(scope="module")
def linker() -> EntityLinker:
    return EntityLinker()


def tickers(mentions) -> list[str]:
    return [m.ticker for m in mentions]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Apple shares rise after strong iPhone sales", ["AAPL"]),
        ("How to bake the perfect apple pie this autumn", []),
        ("Apple orchards suffer from late frost", []),
        ("Amazon rainforest deforestation hits record", []),
        ("Amazon stock falls after AWS outage", ["AMZN"]),
        ("$TSLA to the moon", ["TSLA"]),
        ("Nikola Tesla's inventions celebrated at museum", []),
        ("Chase the dream, says coach after the final", []),
        ("A meta-analysis of sleep studies", []),
        ("Meta shares slide as Instagram ad growth slows", ["META"]),
        ("Google unveils new Pixel phone", ["GOOGL"]),
        ("$GOOG breaks out above resistance", ["GOOGL"]),
        ("Exxon Mobil and Chevron lift dividends", ["XOM", "CVX"]),
        ("Coke prices rise at the movies", []),
        ("Coca-Cola raises prices as costs climb", ["KO"]),
    ],
)
def test_ambiguity_rules(linker: EntityLinker, text: str, expected: list[str]) -> None:
    assert sorted(tickers(linker.link(text))) == sorted(expected)


def test_bare_uppercase_ticker_but_not_common_words(linker: EntityLinker) -> None:
    assert set(tickers(linker.link("JPM beats while BAC misses"))) == {"JPM", "BAC"}
    # 'cost' and 'ba' are ordinary words / not whitelisted
    assert linker.link("The cost of living keeps rising in BA flats") == []


def test_title_mentions_outrank_body_mentions(linker: EntityLinker) -> None:
    ms = linker.link(
        text="Analysts also mentioned Microsoft.", title="JPMorgan profit beats estimates"
    )
    assert ms[0].ticker == "JPM"
    assert ms[0].in_title and not ms[1].in_title
    assert ms[0].relevance > ms[1].relevance


def test_relevance_is_bounded_and_increases_with_mentions(linker: EntityLinker) -> None:
    one = linker.link("Boeing delivers jets")[0].relevance
    many = linker.link("Boeing delivers jets; Boeing shares rise; Boeing CEO Dave Calhoun upbeat")[
        0
    ]
    assert 0 < one < many.relevance <= 1


def test_exec_counts_half(linker: EntityLinker) -> None:
    exec_only = linker.link("Jamie Dimon warns of storm clouds")[0]
    alias = linker.link("JPMorgan warns of storm clouds")[0]
    assert exec_only.ticker == alias.ticker == "JPM"
    assert exec_only.relevance < alias.relevance


def test_org_list_and_prior_ticker_add_evidence(linker: EntityLinker) -> None:
    ms = linker.link("Bank results season begins", orgs=["jpmorgan chase", "xinhua"])
    assert tickers(ms) == ["JPM"]
    tw = linker.link("to the moon", prior_ticker="TSLA")
    assert tickers(tw) == ["TSLA"]


def test_market_wide_items_route_to_mkt_with_regions(linker: EntityLinker) -> None:
    ms = linker.link("Russia launches invasion of Ukraine as West readies sanctions")
    assert tickers(ms) == [MKT]
    assert "RUSSIA_UKRAINE" in ms[0].regions
    fed = linker.link("Fed raises rates by 75 basis points to fight inflation")
    assert tickers(fed) == [MKT] and "US" in fed[0].regions


def test_company_takes_precedence_over_mkt(linker: EntityLinker) -> None:
    ms = linker.link("Goldman Sachs warns inflation will force more rate hikes")
    assert tickers(ms) == ["GS"]


def test_irrelevant_text_links_nothing(linker: EntityLinker) -> None:
    assert linker.link("Local team wins the county cricket final") == []
