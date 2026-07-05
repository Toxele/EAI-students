from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


class TalcReviewVisualizer:
    def __init__(self, report_csv: str | Path) -> None:
        with Path(report_csv).open("r", newline="", encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))

    def save_grid(self, output_path: str | Path, n: int = 24, sort_by_fraction: bool = True) -> None:
        rows = list(self.rows)
        if sort_by_fraction and "talc_fraction" in rows[0]:
            rows.sort(key=lambda row: float(row.get("talc_fraction", 0)), reverse=True)
        rows = rows[:n]
        cols = 4
        rows_n = max(1, (len(rows) + cols - 1) // cols)
        fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 4, rows_n * 3))
        axes = axes.reshape(rows_n, cols) if rows_n > 1 else [axes]
        flat_axes = [ax for row_axes in axes for ax in row_axes]
        for ax in flat_axes:
            ax.axis("off")
        for ax, row in zip(flat_axes, rows):
            image_path = self._resolve_image_path(row)
            image = Image.open(image_path).convert("RGB")
            ax.imshow(image)
            title = Path(row.get("rel_path", image_path)).name
            if "talc_fraction" in row:
                title += f"\nfrac={float(row['talc_fraction']):.3f}"
            ax.set_title(title, fontsize=8)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(output, dpi=160)
        plt.close(fig)

    @staticmethod
    def _resolve_image_path(row: dict[str, str]) -> str:
        candidates = [
            row.get("overlay_path", ""),
            row.get("overlay", ""),
            row.get("image_path", ""),
        ]
        for candidate in candidates:
            if candidate and Path(candidate).exists():
                return candidate

        overlay = row.get("overlay_path") or row.get("overlay") or ""
        if overlay:
            path = Path(overlay)
            for suffix in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]:
                candidate = path.with_suffix(suffix)
                if candidate.exists():
                    return str(candidate)

        raise FileNotFoundError(f"No review image found for row: {row}")
