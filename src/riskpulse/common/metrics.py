"""Single writer for reports/metrics.json: every number quoted in README/deck comes from here."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from typing import Any

from riskpulse.common.config import repo_root, reports_path


def _git_commit() -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], cwd=repo_root(), stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def load_metrics() -> dict[str, Any]:
    path = reports_path("metrics.json")
    return json.loads(path.read_text()) if path.exists() else {}


def update_metrics(section: str, payload: dict[str, Any], script: str) -> dict[str, Any]:
    """Replace one top-level section of metrics.json, stamping provenance."""
    path = reports_path("metrics.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = load_metrics()
    data[section] = {
        **payload,
        "_provenance": {
            "script": script,
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "git_commit": _git_commit(),
        },
    }
    path.write_text(json.dumps(data, indent=2, sort_keys=False, default=float))
    return data
