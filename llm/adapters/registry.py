# -*- coding: utf-8 -*-
"""Registry of dataset adapters used by the Agent."""
from __future__ import annotations

from llm.adapters.l1 import L1Adapter
from llm.adapters.phm2019 import PHM2019Adapter

ADAPTERS = {
    "L1": L1Adapter,
    "PHM2019": PHM2019Adapter,
}


def get_adapter(name: str):
    if name not in ADAPTERS:
        raise KeyError(f"unknown dataset {name}; available: {sorted(ADAPTERS)}")
    return ADAPTERS[name]()


def list_datasets() -> list[str]:
    return sorted(ADAPTERS)
