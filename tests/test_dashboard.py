"""Headless render of every dashboard page against cached outputs (skips if none exist)."""

from pathlib import Path

import pytest

from riskpulse.common.config import data_path, repo_root

PAGES = repo_root() / "src" / "riskpulse" / "dashboard"
HAVE_OUTPUTS = any(
    p.exists()
    for p in [data_path("processed", "mentions.parquet"), data_path("demo", "mentions.parquet")]
)


@pytest.mark.skipif(not HAVE_OUTPUTS, reason="no processed or demo outputs to render")
@pytest.mark.parametrize("force_demo", ["0", "1"])
@pytest.mark.parametrize(
    "page",
    ["app.py", *sorted(p.name for p in (PAGES / "pages").glob("*.py"))],
)
def test_page_renders_without_exceptions(
    page: str, force_demo: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("RISKPULSE_API", "http://127.0.0.1:9")  # API down: cached-output mode
    monkeypatch.setenv("RISKPULSE_FORCE_DEMO", force_demo)  # "1" = fresh-clone snapshot only
    path = PAGES / page if page == "app.py" else PAGES / "pages" / page
    at = AppTest.from_file(str(Path(path)), default_timeout=180).run()
    assert not at.exception, [e.value for e in at.exception]
