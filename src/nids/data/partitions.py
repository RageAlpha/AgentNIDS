from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Partition:
    domain: str
    scheme: str
    name: str
    rank_start: int
    rank_end_exclusive: int

    @property
    def rows(self) -> int:
        return self.rank_end_exclusive - self.rank_start


class PreparedData:
    """Read the immutable rank/split/window contracts produced by data preparation."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.splits = self._read("split_manifest.csv")
        self.windows = self._read("window_manifest.csv")
        self._partitions = {
            (r["domain"], r["scheme"], r["partition"]): Partition(
                r["domain"], r["scheme"], r["partition"],
                int(r["rank_start"]), int(r["rank_end_exclusive"]),
            )
            for r in self.splits
        }
        self._validate()

    def _read(self, name: str) -> list[dict[str, str]]:
        with (self.root / name).open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def partition(self, domain: str, scheme: str, name: str) -> Partition:
        return self._partitions[(domain, scheme, name)]

    def sorted_rows(self, domain: str) -> np.ndarray:
        return np.load(self.root / f"{domain}_sorted_original_rows.npy", mmap_mode="r")

    def sorted_end_ms(self, domain: str) -> np.ndarray:
        return np.load(self.root / f"{domain}_sorted_end_ms.npy", mmap_mode="r")

    def _validate(self) -> None:
        for domain in ("UNSW", "CIC", "BoT", "ToN"):
            order = self.sorted_rows(domain)
            ends = self.sorted_end_ms(domain)
            assert order.shape == ends.shape
            assert np.all(ends[1:] >= ends[:-1])
        source = [r for r in self.splits if r["domain"] == "UNSW" and r["scheme"] == "source"]
        assert [r["partition"] for r in source] == [
            "S_train", "S_prob_cal", "S_competence", "S_source_test"
        ]
        for domain in ("CIC", "BoT", "ToN"):
            development = [r for r in self.splits if r["domain"] == domain and r["scheme"] == "development"]
            assert [r["partition"] for r in development] == ["R_fit", "R_cal", "R_select"]
        for row in self.windows:
            if row["predecessor_window_id"]:
                assert int(row["predecessor_last_available_ms"]) <= int(row["first_available_ms"])

