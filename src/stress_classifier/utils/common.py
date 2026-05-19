from __future__ import annotations

from pathlib import Path
import yaml


def read_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)
