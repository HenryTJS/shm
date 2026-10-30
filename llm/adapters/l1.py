# -*- coding: utf-8 -*-
"""L1/L2 benchmark adapter.

This is a facade over the existing ToolBox. It exposes the same methods the
Agent already uses, but behind the DatasetAdapter interface so that other
datasets can be registered the same way.
"""
from __future__ import annotations

from llm.adapters.base import DatasetAdapter
from llm.tools import ToolBox


class L1Adapter(DatasetAdapter):
    name = "L1"

    def __init__(self) -> None:
        self._tb = ToolBox()

    def capabilities(self) -> dict:
        return {
            "dataset": "L1/L2 benchmark",
            "entities": self.list_entities(),
            "metrics": {
                "t50_pct": {"unit": "percent", "definition": "lifetime fraction at 50% cumulative AE"},
                "t85_pct": {"unit": "percent", "definition": "lifetime fraction at 85% cumulative AE"},
                "delta_t85_pct": {"unit": "pp", "definition": "specimen minus campaign median"},
                "delta_campaign_t85_pct": {"unit": "pp", "definition": "campaign B minus campaign A"},
                "n_specimens": {"unit": "count"},
            },
            "modalities": ["AE", "FBG", "DFOS", "PZT"],
            "validity": "L1 composites + L2 structural panels; L2 is same lab and not an external institution.",
        }

    def list_entities(self) -> list[str]:
        return self._tb.list_specimens()

    # --- L1 AE ---------------------------------------------------------
    def lookup(self, gid: str, metric: str = "t85", variant: str = "cum_hits") -> dict:
        return self._tb.lookup(gid, metric, variant)

    def campaign_stats(self, campaign: str, metric: str = "t85",
                       variant: str = "cum_hits") -> dict:
        return self._tb.campaign_stats(campaign, metric, variant)

    def compare_to_campaign(self, gid: str, metric: str = "t85",
                            variant: str = "cum_hits") -> dict:
        return self._tb.compare_to_campaign(gid, metric, variant)

    def compare_campaigns(self, campaign_a: str, campaign_b: str,
                          metric: str = "t85", variant: str = "cum_hits") -> dict:
        return self._tb.compare_campaigns(campaign_a, campaign_b, metric, variant)

    # --- L1 modalities -------------------------------------------------
    def modality_lookup(self, gid: str, modality: str, metric: str = "t85") -> dict:
        return self._tb.modality_lookup(gid, modality, metric)

    def modality_compare(self, gid: str, modality_a: str, modality_b: str,
                         metric: str = "t85") -> dict:
        return self._tb.modality_compare(gid, modality_a, modality_b, metric)

    # --- L2 (kept here until split into L2Adapter) ---------------------
    def l2_lookup(self, gid: str, metric: str = "t85_all") -> dict:
        return self._tb.l2_lookup(gid, metric)

    def l2_pzt_lookup(self, gid: str) -> dict:
        return self._tb.l2_pzt_lookup(gid)

    def dataset_overview(self) -> dict:
        return self._tb.dataset_overview()

    def compare(self, entity: str, **kwargs) -> dict:
        return self.compare_to_campaign(entity, **kwargs)
