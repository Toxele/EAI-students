"""
Train/val manifest для multi-label coarse/fine из dataset/index/manifest.csv.

Правила:
  - только detail-кадры;
  - ровно один тег coarse или fine (оба сразу — ошибка разметки, исключаем);
  - talc в разметке не мешает: если есть coarse или fine — используем;
  - split по MD5, стратификация по ig-классу (coarse vs fine).

Запуск: py scripts/build_coarse_fine_manifest.py
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
DEFAULT_OUT = ROOT / "artifacts" / "manifests" / "coarse_fine_multilabel_manifest.csv"


def ig_label(row: dict[str, str]) -> str | None:
    """Единственный ig-тег или None для конфликта/пусто."""
    coarse = row.get("tag_coarse") == "1"
    fine = row.get("tag_fine") == "1"
    if coarse and fine:
        return None
    if coarse:
        return "coarse"
    if fine:
        return "fine"
    return None


def load_rows(manifest_csv: Path) -> list[dict[str, str]]:
    """Detail-кадры с ровно одним ig-тегом (coarse или fine)."""
    rows: list[dict[str, str]] = []
    skipped_both = 0
    skipped_none = 0
    with manifest_csv.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("kind") != "detail":
                continue
            label = ig_label(row)
            if label is None:
                if row.get("tag_coarse") == "1" and row.get("tag_fine") == "1":
                    skipped_both += 1
                else:
                    skipped_none += 1
                continue
            rows.append(
                {
                    "md5": row["md5"],
                    "path": row["data_path"],
                    "tags": row["tags"],
                    "tag_talc": row["tag_talc"],
                    "tag_coarse": row["tag_coarse"],
                    "tag_fine": row["tag_fine"],
                    "ig_label": label,
                    "canonical_path": row["canonical_path"],
                }
            )
    print(f"skipped coarse+fine both: {skipped_both}")
    print(f"skipped no ig tag: {skipped_none}")
    return rows


def split_rows(rows: list[dict[str, str]], val_fraction: float, seed: int) -> list[dict[str, str]]:
    """Grouped split по md5, стратификация coarse vs fine."""
    splitter = GroupedStratifiedSplitter(
        val_fraction=val_fraction,
        group_column="md5",
        label_column="label",
        seed=seed,
    )
    keyed = [{**row, "content_hash": row["md5"], "label": row["ig_label"]} for row in rows]
    split = splitter.split(keyed)
    return [{k: v for k, v in row.items() if k not in ("content_hash", "label")} for row in split]


def write_manifest(rows: list[dict[str, str]], output_csv: Path) -> None:
    """Пишет CSV с колонкой subset."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "md5",
        "path",
        "tags",
        "tag_talc",
        "tag_coarse",
        "tag_fine",
        "ig_label",
        "canonical_path",
        "subset",
    ]
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def print_stats(rows: list[dict[str, str]]) -> None:
    """Краткая статистика по subset и talc."""
    for subset in ("train", "val"):
        part = [r for r in rows if r["subset"] == subset]
        coarse = sum(r["ig_label"] == "coarse" for r in part)
        fine = sum(r["ig_label"] == "fine" for r in part)
        with_talc = sum(r["tag_talc"] == "1" for r in part)
        print(f"  {subset}: total={len(part)} coarse={coarse} fine={fine} with_talc_tag={with_talc}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INDEX_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = load_rows(args.input)
    rows = split_rows(rows, val_fraction=args.val_fraction, seed=args.seed)
    write_manifest(rows, args.output)

    print(f"wrote {args.output}: total={len(rows)}")
    print_stats(rows)


if __name__ == "__main__":
    main()
