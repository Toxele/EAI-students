from __future__ import annotations

import random
from collections import defaultdict
from typing import Any


class GroupedStratifiedSplitter:
    def __init__(
        self,
        val_fraction: float = 0.2,
        group_column: str = "content_hash",
        label_column: str = "label",
        seed: int = 42,
    ) -> None:
        self.val_fraction = val_fraction
        self.group_column = group_column
        self.label_column = label_column
        self.seed = seed

    def split(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[str(row[self.group_column])].append(row)

        by_label: dict[str, list[str]] = defaultdict(list)
        for group_id, group_rows in groups.items():
            labels = {row[self.label_column] for row in group_rows}
            label = sorted(labels)[0] if len(labels) == 1 else "conflict"
            by_label[label].append(group_id)

        rng = random.Random(self.seed)
        val_groups: set[str] = set()
        for group_ids in by_label.values():
            rng.shuffle(group_ids)
            n_val = max(1, round(len(group_ids) * self.val_fraction)) if len(group_ids) > 1 else 0
            val_groups.update(group_ids[:n_val])

        output: list[dict[str, Any]] = []
        for row in rows:
            row = dict(row)
            row["subset"] = "val" if str(row[self.group_column]) in val_groups else "train"
            output.append(row)
        return output

