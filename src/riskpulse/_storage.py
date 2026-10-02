"""Optional external storage for large local caches (D-056: relieves the internal disk).

If `RISKPULSE_EXTERNAL_ROOT` is set (environment or the gitignored `.env` at the repo root):
- the Hugging Face hub cache moves there (`HF_HUB_CACHE`), set before any Hugging Face import;
- symlinked folders under `data/processed/` (cache, models) are expected to resolve there.
If the root is configured but missing (drive unplugged), import stops with a clear error instead
of silently re-downloading models or recomputing caches. A fresh clone has no `.env`: no change.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_dotenv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out


def configure() -> None:
    for key, value in _read_dotenv(REPO_ROOT / ".env").items():
        os.environ.setdefault(key, value)
    root = os.environ.get("RISKPULSE_EXTERNAL_ROOT")
    if not root:
        return
    ext = Path(root)
    if not ext.is_dir():
        raise RuntimeError(
            f"External storage not found: RISKPULSE_EXTERNAL_ROOT={root!r} (set in .env). "
            "Mount the drive, or remove the setting to use the internal disk; refusing to "
            "re-download models or rebuild caches silently."
        )
    os.environ.setdefault("HF_HUB_CACHE", str(ext / "hf_hub_cache"))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # exFAT has no symlinks
    processed = REPO_ROOT / "data" / "processed"
    if processed.is_dir():
        for link in processed.iterdir():
            if link.is_symlink() and not link.exists():
                raise RuntimeError(
                    f"{link} points to {os.readlink(link)!r}, which is missing (external drive "
                    "not mounted?)."
                )
