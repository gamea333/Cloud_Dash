"""Load YAML configuration files for agents and routing."""

from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def load_agents_config() -> dict[str, Any]:
    path = CONFIG_DIR / "agents.yaml"
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["agents"]


def load_routing_rules() -> dict[str, Any]:
    path = CONFIG_DIR / "routing_rules.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
