from datetime import UTC, datetime

from fastapi.testclient import TestClient

from riskpulse.api.app import create_app
from riskpulse.api.hub import SignalHub
from riskpulse.engine.aggregate import ScoredMention
from riskpulse.engine.entities import EntityLinker
from riskpulse.ingestion.live import LABEL, DocQueries, LivePoller, gkg_documents
from tests.test_gdelt_gkg import _row, _zip

CFG = {"poll_minutes": 15, "timespan": "1h", "gdelt": {"queries": ["q1", "q2"]}}


def art(title: str, hh: int, url: str | None = None) -> dict:
    return {
        "title": title,
        "url": url or f"https://news.example/{title.replace(' ', '-')}",
        "seendate": f"20261003T{hh:02d}0000Z",
        "domain": "news.example",
        "sourcecountry": "United States",
    }


class FakeClient:
    def __init__(self, batches: list[list[list[dict]]]) -> None:
        self.batches = batches  # per poll, per query
        self.poll = -1
        self.q = 0

    def search(self, query: str, timespan: str = "1h") -> list[dict]:
        if query == "q1":
            self.poll += 1
        return self.batches[self.poll][0 if query == "q1" else 1]


def fake_process(docs):
    mentions = [
        ScoredMention(
            doc_id=d.doc_id,
            source=d.source.value,
            outlet=d.meta["domain"],
            published_at=d.published_at,
            title=d.title,
            url=d.url,
            ticker="MSFT",
            relevance=1.0,
            sentiment=-0.5,
            event_class="EARNINGS",
            event_confidence=0.7,
            event_id="ev1",
            impact_score=6,
            impact_raw=0.4,
            drivers={},
        )
        for d in docs
    ]
    return mentions, []


def test_poller_drops_duplicates_and_late_items_and_labels_output(tmp_path) -> None:
    published: list = []
    client = FakeClient(
        [
            [[art("Microsoft cuts outlook", 10)], [art("Fed holds rates", 11)]],
            [
                [art("Microsoft cuts outlook", 10)],  # same URL: duplicate
                [
                    art("Fed holds rates", 12, url="https://other.example/x"),  # same headline
                    art("Older story arrives late", 9),  # before the watermark
                    art("Intel wins contract", 13),
                ],
            ],
        ]
    )
    src = {"gdelt_doc": DocQueries(CFG, client=client)}
    p = LivePoller(fake_process, published.extend, sources=src, cfg=CFG, out_dir=tmp_path)
    assert p.poll_once() == {"fetched": 2, "new": 2, "mentions": 2, "signals": 0}
    second = p.poll_once()
    assert second["new"] == 1
    assert p.stats["duplicates"] == 2 and p.stats["late_dropped"] == 1
    titles = [r["title"] for r in p.recent]
    assert titles == ["Microsoft cuts outlook", "Fed holds rates", "Intel wins contract"]
    assert all(r["label"] == LABEL and r["live_source"] == "gdelt_doc" for r in p.recent)
    logged = list(tmp_path.glob("mentions_*.jsonl"))[0].read_text().splitlines()
    assert len(logged) == 3 and all(LABEL in line for line in logged)


def test_live_endpoint_off_in_fast_mode_and_serves_poller(tmp_path) -> None:
    hub = SignalHub(None, load_existing=False)
    assert TestClient(create_app("fast", hub=hub)).get("/live").status_code == 503

    p = LivePoller(fake_process, hub.publish, sources={}, cfg=CFG, out_dir=tmp_path)
    p.recent.append({"label": LABEL, "title": "x", "published_at": datetime.now(UTC).isoformat()})
    p.run = lambda stop: None  # no network in tests
    with TestClient(create_app("live", hub=hub, poller=p)) as c:
        body = c.get("/live").json()
        assert body["status"]["label"] == LABEL and body["mentions"][0]["title"] == "x"
        assert c.get("/health").json()["live"]["polls"] == 0


def test_failing_source_does_not_stop_the_others(tmp_path) -> None:
    def down():
        raise ConnectionError("HTTP 429")

    client = FakeClient([[[art("Tesla recalls cars", 10)], []]])
    src = {"gdelt_gkg": down, "gdelt_doc": DocQueries(CFG, client=client)}
    p = LivePoller(fake_process, lambda s: None, sources=src, cfg=CFG, out_dir=tmp_path)
    assert p.poll_once()["new"] == 1
    assert p.stats["source_errors"] == {"gdelt_gkg": 1, "gdelt_doc": 0}


def test_gkg_documents_apply_the_replay_feed_gates() -> None:
    raw = _zip(
        [
            _row("20261003110000", "u1", "TAX_FNCACT", "", "Boeing shares fall on delivery delays"),
            _row(
                "20261003110000", "u2", "ECON_INFLATION", "", "Inflation surges as Fed weighs rates"
            ),
            _row("20261003110000", "u3", "ARMEDCONFLICT", "", "Russia sanctions talks stall"),
            _row(
                "20261003110000", "u4", "ECON_STOCKMARKET", "jpmorgan chase", "Bank stocks wobble"
            ),
        ]
    )
    docs = {d.url: d for d in gkg_documents(raw, EntityLinker())}
    # u3: market item without an economic theme; u4: company only in the article's orgs
    assert set(docs) == {"u1", "u2"}
    assert docs["u1"].meta["tickers"] == ["BA"] and docs["u2"].meta["tickers"] == ["MKT"]
    assert docs["u1"].published_at == datetime(2026, 10, 3, 11, tzinfo=UTC)


class _R:
    def __init__(self, status: int, body: bytes = b"") -> None:
        self.status_code, self.content, self.text = status, body, body.decode("latin-1")

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


def test_gkg_latest_steps_back_past_files_not_yet_downloadable() -> None:
    from riskpulse.ingestion.live import GkgLatest

    base = "http://data.example/gdeltv2/"
    zipped = _zip([_row("20261003113000", "u1", "TAX_FNCACT", "", "Boeing shares fall")])
    files = {f"{base}20261003113000.gkg.csv.zip": zipped}  # 12:00 and 11:45 are still 404
    lastupdate = f"1 abc {base}20261003120000.gkg.csv.zip\n".encode()

    class S:
        def get(self, url, timeout=None):
            if url.endswith("lastupdate.txt"):
                return _R(200, lastupdate)
            return _R(200, files[url]) if url in files else _R(404)

    src = GkgLatest({"lastupdate_url": base + "lastupdate.txt"}, session=S())
    docs = src()
    assert [d.url for d in docs] == ["u1"] and src.lag_files == 2
    assert src() == []  # already read; newer files still missing
