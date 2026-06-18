"""
core/portfolio.py
Portfolio data model. Stores tickers + weights + value, validates inputs,
and provides a serialisable dict for persistence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np


@dataclass
class Portfolio:
    name: str
    holdings: dict[str, float]   # ticker -> weight (0-1)
    value: float = 100_000.0     # base currency value

    def __post_init__(self):
        self.validate()

    # ------------------------------------------------------------------

    def validate(self) -> None:
        if not self.holdings:
            raise ValueError("Portfolio must have at least one holding.")
        total = sum(self.holdings.values())
        if not np.isclose(total, 1.0, atol=1e-4):
            raise ValueError(f"Weights must sum to 1.0, got {total:.6f}")
        if self.value <= 0:
            raise ValueError("Portfolio value must be positive.")

    @property
    def tickers(self) -> list[str]:
        return list(self.holdings.keys())

    @property
    def weights(self) -> np.ndarray:
        return np.array(list(self.holdings.values()), dtype=float)

    def normalise_weights(self) -> None:
        """Re-scale weights to sum to exactly 1.0 in-place."""
        total = sum(self.holdings.values())
        self.holdings = {k: v / total for k, v in self.holdings.items()}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {"name": self.name, "holdings": self.holdings, "value": self.value}

    @classmethod
    def from_dict(cls, d: dict) -> "Portfolio":
        return cls(name=d["name"], holdings=d["holdings"], value=d["value"])

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> "Portfolio":
        return cls.from_dict(json.loads(path.read_text()))

    # ------------------------------------------------------------------
    # Presets
    # ------------------------------------------------------------------

    @classmethod
    def default(cls) -> "Portfolio":
        """Your two main ETFs as a 60/40 starting point."""
        return cls(
            name="My Portfolio",
            holdings={"VWRP.L": 0.60, "XMWX.L": 0.40},
            value=100_000.0,
        )
