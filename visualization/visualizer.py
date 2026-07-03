from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from PIL import Image, ImageOps


class DatasetVisualizer:
    def __init__(self, manifest_csv: str | Path) -> None:
        self.manifest_csv = Path(manifest_csv)
        with self.manifest_csv.open("r", newline="", encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))

    def save_label_grid(
        self,
        output_path: str | Path,
        labels: list[str] | None = None,
        n_per_label: int = 6,
        seed: int = 42,
    ) -> None:
        labels = labels or sorted({row["label"] for row in self.rows})
        rng = random.Random(seed)
        chosen: list[dict[str, Any]] = []
        for label in labels:
            rows = [row for row in self.rows if row["label"] == label and row["source"] == "classification"]
            rng.shuffle(rows)
            chosen.extend(rows[:n_per_label])

        cols = n_per_label
        rows_count = max(len(labels), 1)
        fig, axes = plt.subplots(rows_count, cols, figsize=(cols * 3, rows_count * 3))
        if rows_count == 1:
            axes = [axes]
        for ax_row in axes:
            for ax in ax_row:
                ax.axis("off")
        for idx, row in enumerate(chosen):
            r = idx // cols
            c = idx % cols
            with Image.open(row["path"]) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                image.thumbnail((512, 512))
                axes[r][c].imshow(image)
                title = f"{row['label']}\n{Path(row['rel_path']).name}"
                if row.get("label_conflict", "").lower() == "true":
                    title += "\nCONFLICT"
                axes[r][c].set_title(title, fontsize=8)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(output, dpi=160)
        plt.close(fig)

    def save_audit_bars(self, output_path: str | Path) -> None:
        labels = sorted({row["label"] for row in self.rows})
        counts = [sum(1 for row in self.rows if row["label"] == label) for label in labels]
        conflicts = [
            sum(1 for row in self.rows if row["label"] == label and row.get("label_conflict", "").lower() == "true")
            for label in labels
        ]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(labels, counts, label="all")
        ax.bar(labels, conflicts, label="conflicts")
        ax.set_ylabel("images")
        ax.legend()
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(output, dpi=160)
        plt.close(fig)

