import io
import zipfile
from datetime import UTC, datetime

from riskpulse.engine.entities import EntityLinker
from riskpulse.ingestion.gdelt_gkg import filter_records, gkg_timestamps, parse_gkg

PREFIXES = ["ECON_", "EPU_", "ARMEDCONFLICT", "SANCTIONS"]


def _row(date: str, url: str, themes: str, orgs: str, title: str | None) -> str:
    cols = [""] * 27
    cols[0], cols[1], cols[3], cols[4] = "rec", date, "example.com", url
    cols[7], cols[9], cols[13], cols[15] = themes, "1#United States#US#US#0#0#US", orgs, "-3.2,1,4"
    cols[26] = f"<PAGE_TITLE>{title}</PAGE_TITLE>" if title else ""
    return "\t".join(cols)


def _zip(lines: list[str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("x.gkg.csv", "\n".join(lines) + "\n")
    return buf.getvalue()


def test_parse_and_filter_keeps_linked_and_mkt_rows_only() -> None:
    raw = _zip(
        [
            _row("20220224120000", "u1", "TAX_FNCACT", "", "Boeing shares fall on delivery delays"),
            _row(
                "20220224120000",
                "u2",
                "ARMEDCONFLICT;SANCTIONS",
                "",
                "Russia invades Ukraine; sanctions loom",
            ),
            _row("20220224120000", "u3", "TAX_FNCACT", "", "Local bakery wins award"),
            _row(
                "20220224120000", "u4", "ECON_STOCKMARKET", "jpmorgan chase", "Bank stocks wobble"
            ),
            _row("20220224120000", "u5", "TAX_FNCACT", "", None),
        ]
    )
    df = parse_gkg(raw)
    assert df.attrs["n_raw"] == 5 and len(df) == 4
    rows = {r["url"]: r for r in filter_records(df, EntityLinker(), PREFIXES)}
    assert set(rows) == {"u1", "u2", "u4"}
    assert rows["u1"]["tickers"] == ["BA"]
    assert rows["u2"]["is_mkt"] and "RUSSIA_UKRAINE" in rows["u2"]["regions"]
    assert rows["u4"]["tickers"] == [] and rows["u4"]["org_only_tickers"] == ["JPM"]
    assert rows["u1"]["tone"] == -3.2 and rows["u1"]["countries"] == ["US"]


def test_timestamps_follow_sampling_config() -> None:
    sampling = {"minute": 0, "weekday_hours_utc": [14, 15], "weekend_hours_utc": [12]}
    ts = gkg_timestamps(datetime(2022, 2, 25), datetime(2022, 2, 27), sampling)  # Fri + Sat
    assert ts == [
        datetime(2022, 2, 25, 14, tzinfo=UTC),
        datetime(2022, 2, 25, 15, tzinfo=UTC),
        datetime(2022, 2, 26, 12, tzinfo=UTC),
    ]
