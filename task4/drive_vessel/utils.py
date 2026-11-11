from __future__ import annotations

import json
import os
import random
from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def deep_update(base: dict, update: dict) -> dict:
    result = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_yaml(path: str | Path) -> dict:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    parent = config.pop("extends", None)
    if parent:
        base = load_yaml(path.parent / parent)
        config = deep_update(base, config)
    config["_config_path"] = str(path)
    return config


def parse_overrides(values: Iterable[str]) -> dict:
    overrides: dict = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got {item!r}")
        dotted, raw_value = item.split("=", 1)
        value = yaml.safe_load(raw_value)
        target = overrides
        parts = dotted.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return overrides


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def get_device(config: dict) -> torch.device:
    requested = str(config.get("device", "cuda"))
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)
