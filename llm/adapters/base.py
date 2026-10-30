# -*- coding: utf-8 -*-
"""Dataset adapter interface.

The Agent should depend on this interface, not on a particular CSV schema.
A new dataset is added by implementing `capabilities`, `lookup`, and `compare`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class DatasetAdapter(ABC):
    name: str = "dataset"

    @abstractmethod
    def capabilities(self) -> dict[str, Any]:
        """Return entities, metrics, units, and validity notes."""

    @abstractmethod
    def lookup(self, entity: str, metric: str, **kwargs: Any) -> dict:
        """Return {tool_call, evidence, result} for one metric."""

    @abstractmethod
    def compare(self, entity: str, **kwargs: Any) -> dict:
        """Return {tool_call, evidence, result} for a comparison."""

    @abstractmethod
    def list_entities(self) -> list[str]:
        """Return known entity IDs."""
