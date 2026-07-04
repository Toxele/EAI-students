"""
Train/val manifest для multi-label классификатора из dataset/index/manifest.csv.

Берёт только detail-кадры с хотя бы одним тегом (talc/coarse/fine).
Сплит по MD5 (все дубликаты в одном subset).

Запуск: py scripts/build_multilabel_manifest.py
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.splits import GroupedStratifiedSplitter

INDEX_MANIFEST = ROOT / "dataset" / "index" / "manifest.csv"
DEFAULT_OUT = ROOT / "artifacts" / "manifests" / "multilabel_manifest.csv"

TAGS = ("tag_talc", "tag_coarse", "tag_fine")


def load_detail_rows(manifest_csv: Path) -> list[dict[str, str]]:
    """Строки detail с хотя бы одним тегом."""
    rows: list[dict[str, str]] = []
    with manifest_csv.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("kind") != "detail":
                continue
            if not any(row.get(tag) == "1" for tag in TAGS):
                continue
            rows.append(
                {
                    "md5": row["md5"],
                    "path": row["data_path"],
                    "tags": row["tags"],
                    "tag_talc": row["tag_talc"],
                    "tag_coarse": row["tag_coarse"],
                    "tag_fine": row["tag_fine"],
                    "canonical_path": row["canonical_path"],
                }
            )
    return rows


def split_rows(rows: list[dict[str, str]], val_fraction: float, seed: int) -> list[dict[str, str]]:
    """Grouped split: md5 → train/val, стратификация по tags."""
    splitter = GroupedStratifiedSplitter(
        val_fraction=val_fraction,
        group_column="md5",
        label_column="tags",
        seed=seed,
    )
    keyed = [{**row, "content_hash": row["md5"], "label": row["tags"]} for row in rows]
    split = splitter.split(keyed)
    return [{k: v for k, v in row.items() if k not in ("content_hash", "label")} for row in split]


def write_manifest(rows: list[dict[str, str]], output_csv: Path) -> None:
    """Пишет CSV с колонкой subset."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["md5", "path", "tags", "tag_talc", "tag_coarse", "tag_fine", "canonical_path", "subset"]
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INDEX_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = load_detail_rows(args.input)
    rows = split_rows(rows, val_fraction=args.val_fraction, seed=args.seed)
    write_manifest(rows, args.output)

    n_train = sum(r["subset"] == "train" for r in rows)
    n_val = sum(r["subset"] == "val" for r in rows)
    print(f"wrote {args.output}: total={len(rows)} train={n_train} val={n_val}")


if __name__ == "__main__":
    main()
