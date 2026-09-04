"""Data models for Structure Studio handoff payloads."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class FocusResidue:
    chain: str
    residue_number: int
    label: str = ""

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "chain": self.chain,
            "residue_number": self.residue_number,
        }
        if self.label:
            payload["label"] = self.label
        return payload


@dataclass(frozen=True)
class StructureCandidate:
    pdb_id: str
    method: str = ""
    resolution: float | None = None
    chains: list[str] = field(default_factory=list)
    source_url: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StructureView:
    viewer_url: str
    payload: dict[str, Any]
    selected_pdb_id: str
    selected_structure: StructureCandidate
    uncertainty: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "viewer_url": self.viewer_url,
            "payload": self.payload,
            "selected_pdb_id": self.selected_pdb_id,
            "selected_structure": self.selected_structure.to_dict(),
            "uncertainty": self.uncertainty,
        }
