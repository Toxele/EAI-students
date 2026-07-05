from __future__ import annotations

import csv
from pathlib import Path


class CsvAnnotationStore:
    """Stores manual label fixes without moving source images."""

    FIELDNAMES = ["rel_path", "label", "comment", "annotator"]

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> dict[str, dict[str, str]]:
        if not self.path.exists():
            return {}
        with self.path.open("r", newline="", encoding="utf-8") as f:
            return {row["rel_path"]: row for row in csv.DictReader(f)}

    def save_template(self, rel_paths: list[str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        existing = self.load()
        with self.path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            writer.writeheader()
            for rel_path in rel_paths:
                writer.writerow(existing.get(rel_path, {"rel_path": rel_path, "label": "", "comment": "", "annotator": ""}))

    def apply_to_manifest(self, manifest_rows: list[dict[str, str]]) -> list[dict[str, str]]:
        annotations = self.load()
        output: list[dict[str, str]] = []
        for row in manifest_rows:
            row = dict(row)
            annotation = annotations.get(row["rel_path"])
            if annotation and annotation.get("label"):
                row["original_label"] = row.get("label", "")
                row["label"] = annotation["label"]
                row["manual_label"] = "true"
                row["manual_comment"] = annotation.get("comment", "")
            else:
                row.setdefault("manual_label", "false")
            output.append(row)
        return output

