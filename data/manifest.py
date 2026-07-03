from __future__ import annotations

import csv
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class ManifestRecord:
    path: str
    rel_path: str
    filename: str
    label: str
    source: str
    width: int
    height: int
    bytes: int
    content_hash: str
    duplicate_count: int = 1
    label_conflict: bool = False
    duplicate_group: str = ""
    subset: str = ""


class NornikelManifestBuilder:
    def __init__(
        self,
        dataset_root: str | Path,
        image_extensions: list[str],
        class_markers: dict[str, list[str]],
        sources: dict[str, str],
        hash_algorithm: str = "md5",
        max_hash_mb: int = 512,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.image_extensions = {ext.lower() for ext in image_extensions}
        self.class_markers = {
            label: [marker.lower() for marker in markers]
            for label, markers in class_markers.items()
        }
        self.sources = {key: value.lower() for key, value in sources.items()}
        self.hash_algorithm = hash_algorithm
        self.max_hash_bytes = max_hash_mb * 1024 * 1024

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "NornikelManifestBuilder":
        return cls(
            dataset_root=cfg["dataset_root"],
            image_extensions=cfg["image_extensions"],
            class_markers=cfg["class_markers"],
            sources=cfg["sources"],
            hash_algorithm=cfg.get("hash_algorithm", "md5"),
            max_hash_mb=cfg.get("max_hash_mb", 512),
        )

    def build(self) -> list[ManifestRecord]:
        rows: list[ManifestRecord] = []
        for path in self._iter_images():
            rel_path = path.relative_to(self.dataset_root).as_posix()
            source = self._infer_source(rel_path)
            label = self._infer_label(rel_path)
            width, height = self._read_size(path)
            rows.append(
                ManifestRecord(
                    path=str(path),
                    rel_path=rel_path,
                    filename=path.name,
                    label=label,
                    source=source,
                    width=width,
                    height=height,
                    bytes=path.stat().st_size,
                    content_hash=self._hash_file(path),
                )
            )
        return self._mark_duplicates(rows)

    def write_csv(self, output_csv: str | Path) -> list[ManifestRecord]:
        rows = self.build()
        output = Path(output_csv)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(ManifestRecord.__dataclass_fields__))
            writer.writeheader()
            for row in rows:
                writer.writerow(row.__dict__)
        return rows

    def _iter_images(self):
        for dirpath, _, filenames in os.walk(self.dataset_root):
            for filename in filenames:
                path = Path(dirpath) / filename
                if path.suffix.lower() in self.image_extensions:
                    yield path

    def _infer_source(self, rel_path: str) -> str:
        lowered = rel_path.lower()
        if self.sources.get("weak_talc_part", "") in lowered:
            return "weak_talc"
        if self.sources.get("panorama_part", "") in lowered:
            return "panorama"
        if self.sources.get("classification_part", "") in lowered:
            return "classification"
        return "auxiliary"

    def _infer_label(self, rel_path: str) -> str:
        lowered = rel_path.lower()
        for label, markers in self.class_markers.items():
            if any(marker in lowered for marker in markers):
                return label
        return "unknown"

    def _read_size(self, path: Path) -> tuple[int, int]:
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(path) as img:
            return img.size

    def _hash_file(self, path: Path) -> str:
        if path.stat().st_size > self.max_hash_bytes:
            return f"too_large:{path.stat().st_size}:{path.name}"
        digest = hashlib.new(self.hash_algorithm)
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _mark_duplicates(rows: list[ManifestRecord]) -> list[ManifestRecord]:
        by_hash: dict[str, list[ManifestRecord]] = {}
        for row in rows:
            by_hash.setdefault(row.content_hash, []).append(row)

        marked: list[ManifestRecord] = []
        for idx, group in enumerate(by_hash.values()):
            labels = {row.label for row in group}
            conflict = len(labels) > 1
            group_id = f"dup_{idx:05d}" if len(group) > 1 else ""
            for row in group:
                marked.append(
                    ManifestRecord(
                        **{
                            **row.__dict__,
                            "duplicate_count": len(group),
                            "label_conflict": conflict,
                            "duplicate_group": group_id,
                        }
                    )
                )
        return sorted(marked, key=lambda item: item.rel_path)


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def write_manifest(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

