"""YAML config loading.

All tunable parameters live in ``configs/*.yaml``. Code reads them through
:func:`load_config` so nothing numeric is hard-coded in modules.
"""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path
from typing import Any

import yaml

CONFIG_NAMES: tuple[str, ...] = (
    "app",
    "universe",
    "taxonomy",
    "impact",
    "scenarios",
    "moduleA",
    "moduleB",
    "mcc_sector_map",
)


def repo_root() -> Path:
    """Return the repository root (the directory that contains ``configs/``)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "configs").is_dir() and (parent / "src").is_dir():
            return parent
    return Path.cwd()


def config_dir() -> Path:
    """Directory holding the YAML configs; overridable via ``RISKPULSE_CONFIG_DIR``."""
    override = os.environ.get("RISKPULSE_CONFIG_DIR")
    return Path(override) if override else repo_root() / "configs"


@cache
def load_config(name: str) -> dict[str, Any]:
    """Load ``configs/<name>.yaml`` as a dict.

    Raises:
        FileNotFoundError: if the config file does not exist.
    """
    path = config_dir() / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must be a mapping at top level")
    return data


def data_path(*parts: str) -> Path:
    """Path under the repo ``data/`` directory."""
    return repo_root().joinpath("data", *parts)


def reports_path(*parts: str) -> Path:
    """Path under the repo ``reports/`` directory."""
    return repo_root().joinpath("reports", *parts)
