from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from data.manifest import read_manifest

Image.MAX_IMAGE_PIXELS = None


@dataclass(frozen=True)
class ColorDomainStats:
    rel_path: str
    path: str
    label: str
    source: str
    domain_group: str
    width: int
    height: int
    mean_r: float
    mean_g: float
    mean_b: float
    std_r: float
    std_g: float
    std_b: float
    mean_h: float
    mean_s: float
    mean_v: float
    mean_l: float
    mean_a: float
    mean_lab_b: float
    green_score: float
    gray_score: float
    brightness: float
    contrast: float


class ColorDomainAnalyzer:
    def __init__(
        self,
        manifest_csv: str | Path,
        output_dir: str | Path,
        thumbnail_size: int = 256,
        max_examples_per_group: int = 24,
    ) -> None:
        self.manifest_csv = Path(manifest_csv)
        self.output_dir = Path(output_dir)
        self.thumbnail_size = thumbnail_size
        self.max_examples_per_group = max_examples_per_group

    def run(self) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        rows = read_manifest(self.manifest_csv)
        stats = [self._analyze_row(row) for row in rows]
        self._write_stats(stats)
        summary = self._summary(stats)
        self._write_summary(summary)
        self._plot_distributions(stats)
        self._save_contact_sheets(stats)
        return summary

    def _analyze_row(self, row: dict[str, str]) -> ColorDomainStats:
        path = Path(row["path"])
        with Image.open(path) as img:
            image = ImageOps.exif_transpose(img).convert("RGB")
            width, height = image.size
            image.thumbnail((self.thumbnail_size, self.thumbnail_size))
            rgb = np.asarray(image, dtype=np.uint8)

        rgb_float = rgb.astype(np.float32)
        mean_rgb = rgb_float.reshape(-1, 3).mean(axis=0)
        std_rgb = rgb_float.reshape(-1, 3).std(axis=0)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        mean_hsv = hsv.reshape(-1, 3).mean(axis=0)
        mean_lab = lab.reshape(-1, 3).mean(axis=0)
        brightness = float(mean_rgb.mean())
        contrast = float(std_rgb.mean())
        green_score = float((mean_rgb[1] - mean_rgb[2]) + 0.5 * (mean_rgb[1] - mean_rgb[0]))
        gray_score = float(255 - mean_hsv[1])

        return ColorDomainStats(
            rel_path=row["rel_path"],
            path=row["path"],
            label=row.get("label", ""),
            source=row.get("source", ""),
            domain_group=self._domain_group(row),
            width=width,
            height=height,
            mean_r=round(float(mean_rgb[0]), 4),
            mean_g=round(float(mean_rgb[1]), 4),
            mean_b=round(float(mean_rgb[2]), 4),
            std_r=round(float(std_rgb[0]), 4),
            std_g=round(float(std_rgb[1]), 4),
            std_b=round(float(std_rgb[2]), 4),
            mean_h=round(float(mean_hsv[0]), 4),
            mean_s=round(float(mean_hsv[1]), 4),
            mean_v=round(float(mean_hsv[2]), 4),
            mean_l=round(float(mean_lab[0]), 4),
            mean_a=round(float(mean_lab[1]), 4),
            mean_lab_b=round(float(mean_lab[2]), 4),
            green_score=round(green_score, 4),
            gray_score=round(gray_score, 4),
            brightness=round(brightness, 4),
            contrast=round(contrast, 4),
        )

    @staticmethod
    def _domain_group(row: dict[str, str]) -> str:
        rel = row["rel_path"].lower()
        source = row.get("source", "")
        if source == "panorama":
            return "panorama"
        if source == "weak_talc":
            return "manual_talc_markup"
        if "ч1" in rel or "ch1" in rel:
            return "classification_ch1"
        if "ч2" in rel or "ch2" in rel:
            return "classification_ch2"
        return source or "unknown"

    def _write_stats(self, stats: list[ColorDomainStats]) -> None:
        path = self.output_dir / "color_domain_stats.csv"
        fieldnames = list(ColorDomainStats.__dataclass_fields__)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for item in stats:
                writer.writerow(item.__dict__)

    def _summary(self, stats: list[ColorDomainStats]) -> dict[str, Any]:
        groups = sorted({s.domain_group for s in stats})
        metrics = [
            "mean_r",
            "mean_g",
            "mean_b",
            "mean_h",
            "mean_s",
            "mean_v",
            "mean_l",
            "mean_a",
            "mean_lab_b",
            "green_score",
            "gray_score",
            "brightness",
            "contrast",
        ]
        by_group: dict[str, Any] = {}
        for group in groups:
            values = [s for s in stats if s.domain_group == group]
            by_group[group] = {"count": len(values)}
            for metric in metrics:
                arr = np.array([getattr(s, metric) for s in values], dtype=np.float32)
                by_group[group][metric] = {
                    "mean": round(float(arr.mean()), 4),
                    "std": round(float(arr.std()), 4),
                    "p10": round(float(np.percentile(arr, 10)), 4),
                    "p50": round(float(np.percentile(arr, 50)), 4),
                    "p90": round(float(np.percentile(arr, 90)), 4),
                }
        return {
            "total": len(stats),
            "by_group": by_group,
            "by_group_label": {
                f"{group}|{label}": count
                for (group, label), count in Counter((s.domain_group, s.label) for s in stats).items()
            },
        }

    def _write_summary(self, summary: dict[str, Any]) -> None:
        (self.output_dir / "color_domain_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _plot_distributions(self, stats: list[ColorDomainStats]) -> None:
        import matplotlib.pyplot as plt

        groups = sorted({s.domain_group for s in stats})
        colors = {
            "classification_ch1": "#4b8bbe",
            "classification_ch2": "#d95f02",
            "manual_talc_markup": "#1b9e77",
            "panorama": "#7570b3",
        }
        metrics = [
            ("green_score", "Green score"),
            ("mean_s", "HSV saturation"),
            ("brightness", "RGB brightness"),
            ("contrast", "RGB contrast"),
            ("mean_a", "LAB a"),
            ("mean_lab_b", "LAB b"),
        ]
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        for ax, (metric, title) in zip(axes.ravel(), metrics):
            for group in groups:
                values = [getattr(s, metric) for s in stats if s.domain_group == group]
                if values:
                    ax.hist(values, bins=35, alpha=0.45, label=group, color=colors.get(group))
            ax.set_title(title)
            ax.grid(alpha=0.2)
        axes[0, 0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(self.output_dir / "color_histograms.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 7))
        for group in groups:
            values = [s for s in stats if s.domain_group == group]
            ax.scatter(
                [s.mean_a for s in values],
                [s.mean_lab_b for s in values],
                s=18,
                alpha=0.6,
                label=group,
                color=colors.get(group),
            )
        ax.set_xlabel("LAB a")
        ax.set_ylabel("LAB b")
        ax.set_title("Color domain scatter")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(self.output_dir / "lab_scatter.png", dpi=160)
        plt.close(fig)

    def _save_contact_sheets(self, stats: list[ColorDomainStats]) -> None:
        groups = sorted({s.domain_group for s in stats})
        for group in groups:
            group_stats = [s for s in stats if s.domain_group == group]
            group_stats = sorted(group_stats, key=lambda item: item.green_score)
            if not group_stats:
                continue
            selected = self._spread(group_stats, self.max_examples_per_group)
            self._contact_sheet(selected, self.output_dir / f"examples_{group}.jpg")

    @staticmethod
    def _spread(items: list[ColorDomainStats], n: int) -> list[ColorDomainStats]:
        if len(items) <= n:
            return items
        idxs = np.linspace(0, len(items) - 1, n).round().astype(int)
        return [items[int(idx)] for idx in idxs]

    @staticmethod
    def _contact_sheet(items: list[ColorDomainStats], output_path: Path, thumb: int = 180) -> None:
        cols = 6
        rows = int(np.ceil(len(items) / cols))
        sheet = Image.new("RGB", (cols * thumb, rows * (thumb + 34)), (25, 28, 32))
        draw = ImageDraw.Draw(sheet)
        for idx, item in enumerate(items):
            x = (idx % cols) * thumb
            y = (idx // cols) * (thumb + 34)
            with Image.open(item.path) as img:
                image = ImageOps.exif_transpose(img).convert("RGB")
                image.thumbnail((thumb, thumb))
                sheet.paste(image, (x, y))
            label = f"{item.label} gs={item.green_score:.1f}"
            draw.text((x + 3, y + thumb + 3), label[:28], fill=(230, 235, 240))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(output_path, quality=92)
