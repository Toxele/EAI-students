"""
Сборка dataset/ из data/: полное зеркало + classification по MD5 + индекс.

Запуск: py scripts/build_dataset.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "dataset"

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def file_md5(path: Path, block: int = 65536) -> str:
    """MD5 содержимого файла."""
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while chunk := handle.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def tags_from_path(rel: str) -> set[str]:
    """
    Теги из пути (English, unified ch1/ch2).

    coarse = row ore / coarse intergrowth; fine = difficult-to-dress / fine intergrowth.
    """
    tags: set[str] = set()
    parts = rel.replace("\\", "/").split("/")

    if parts and "Панорамы" in parts[0]:
        tags.add("panorama")
        return tags

    if "Области оталькования" in rel:
        tags.add("talc_annotation")

    if "ч1" in parts[0]:
        if len(parts) > 1:
            folder = parts[1]
            if folder == "Оталькованные руды":
                tags.add("talc_bearing")
            elif folder == "Рядовые руды":
                tags.add("non_talc_bearing")
                tags.add("coarse")
            elif folder == "Труднообогатимые руды":
                tags.add("non_talc_bearing")
                tags.add("fine")

    if "ч2" in parts[0] and len(parts) > 1:
        folder = parts[1]
        if folder == "оталькованные":
            tags.add("talc_bearing")
        elif folder == "рядовые":
            tags.add("coarse")
        elif folder == "тонкие":
            tags.add("fine")

    return tags


def folder_name(tags: set[str]) -> str:
    """Имя compound-папки из множества тегов."""
    if "panorama" in tags:
        return "panoramas"
    if "talc_annotation" in tags and tags <= {"talc_annotation", "talc_bearing"}:
        return "_talc_annotation_only"

    has_talc = "talc_bearing" in tags
    has_non = "non_talc_bearing" in tags
    if has_talc and has_non:
        talc_part = "talc_mixed"
    elif has_talc:
        talc_part = "talc_bearing"
    elif has_non:
        talc_part = "non_talc_bearing"
    else:
        talc_part = "unknown_talc"

    has_coarse = "coarse" in tags
    has_fine = "fine" in tags
    if has_coarse and has_fine:
        ig_part = "coarse_and_fine"
    elif has_coarse:
        ig_part = "coarse"
    elif has_fine:
        ig_part = "fine"
    else:
        ig_part = "unknown_intergrowth"

    return f"{talc_part}__{ig_part}"


def pick_canonical(paths: list[str]) -> str:
    """
    Канонический путь для classification: ch1 без annotated > ch2 > annotated.
    """
    def score(rel: str) -> tuple[int, str]:
        is_ann = "Области оталькования" in rel
        is_ch1 = "ч1" in rel
        is_ch2 = "ч2" in rel
        if is_ch1 and not is_ann:
            return (0, rel)
        if is_ch2:
            return (1, rel)
        if is_ann:
            return (2, rel)
        return (3, rel)

    return min(paths, key=score)


@dataclass
class Md5Group:
    """Группа файлов с одинаковым MD5."""

    paths: list[str] = field(default_factory=list)
    tags: set[str] = field(default_factory=set)


def scan_groups() -> dict[str, Md5Group]:
    """Обходит data/ и строит словарь md5 → paths + tags."""
    groups: dict[str, Md5Group] = {}
    for root, _, files in os.walk(DATA):
        for name in sorted(files):
            if Path(name).suffix.lower() not in IMAGE_EXT:
                continue
            path = Path(root) / name
            rel = path.relative_to(DATA).as_posix()
            digest = file_md5(path)
            if digest not in groups:
                groups[digest] = Md5Group()
            groups[digest].paths.append(rel)
            groups[digest].tags |= tags_from_path(rel)
    return groups


def copy_file(src_rel: str, dst: Path) -> None:
    """Копирует файл из data/, создаёт родительские папки."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DATA / src_rel.replace("/", os.sep), dst)


def build_talc_pairs(groups: dict[str, Md5Group]) -> list[dict]:
    """Пары original / annotated по имени файла в ch1."""
    by_name: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for digest, group in groups.items():
        for rel in group.paths:
            if "Области оталькования" in rel:
                by_name[Path(rel).name].append(("annotated", rel))
            elif "ч1" in rel and "Оталькованные руды" in rel:
                by_name[Path(rel).name].append(("original", rel))

    pairs: list[dict] = []
    for filename, items in sorted(by_name.items()):
        roles = dict(items)
        if "original" in roles and "annotated" in roles:
            pairs.append(
                {
                    "filename": filename,
                    "original": roles["original"],
                    "annotated": roles["annotated"],
                }
            )
    return pairs


def main() -> None:
    """Главная функция сборки dataset/."""
    if not DATA.is_dir():
        raise SystemExit(f"Not found: {DATA}")

    if OUT.exists():
        shutil.rmtree(OUT)

    source_dir = OUT / "source"
    class_dir = OUT / "classification"
    index_dir = OUT / "index"
    talc_dir = OUT / "talc_segmentation"
    for d in (source_dir, class_dir, index_dir, talc_dir):
        d.mkdir(parents=True)

    groups = scan_groups()
    talc_pairs = build_talc_pairs(groups)

    # --- A. Полное зеркало source/ ---
    all_paths_rows: list[dict] = []
    for digest, group in groups.items():
        for rel in group.paths:
            copy_file(rel, source_dir / rel.replace("/", os.sep))
            all_paths_rows.append({"md5": digest, "relative_path": rel, "tags": "|".join(sorted(group.tags))})

    # --- B. classification/ — один canonical на MD5 (кроме annotation-only дубликатов) ---
    manifest_rows: list[dict] = []
    by_md5: dict[str, dict] = {}

    for digest, group in sorted(groups.items()):
        folder = folder_name(group.tags)
        canonical = pick_canonical(group.paths)
        by_md5[digest] = {
            "paths": group.paths,
            "tags": sorted(group.tags),
            "folder": folder,
            "canonical_path": canonical,
        }

        # annotated-only MD5 без «фото» классификации — только talc_segmentation
        if folder == "_talc_annotation_only":
            continue
        if "panorama" in group.tags:
            folder = "panoramas"

        stem = Path(canonical).name
        dst_name = f"{digest[:12]}_{stem}"
        copy_file(canonical, class_dir / folder / dst_name)

        manifest_rows.append(
            {
                "md5": digest,
                "folder": folder,
                "tags": "|".join(sorted(group.tags)),
                "canonical_path": canonical,
                "duplicate_paths": "|".join(p for p in group.paths if p != canonical),
                "copy_name": dst_name,
            }
        )

    # --- C. talc_segmentation/ ---
    for pair in talc_pairs:
        stem = Path(pair["filename"]).stem
        copy_file(pair["original"], talc_dir / "images" / pair["filename"])
        copy_file(pair["annotated"], talc_dir / "annotated" / pair["filename"])
        pair["image_path"] = f"talc_segmentation/images/{pair['filename']}"
        pair["annotated_path"] = f"talc_segmentation/annotated/{pair['filename']}"

    # --- D. index/ ---
    with (index_dir / "paths.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["md5", "relative_path", "tags"])
        writer.writeheader()
        writer.writerows(all_paths_rows)

    with (index_dir / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["md5", "folder", "tags", "canonical_path", "duplicate_paths", "copy_name"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest_rows)

    with (index_dir / "by_md5.json").open("w", encoding="utf-8") as handle:
        json.dump(by_md5, handle, ensure_ascii=False, indent=2)

    with (index_dir / "talc_pairs.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "original", "annotated", "image_path", "annotated_path"])
        writer.writeheader()
        writer.writerows(talc_pairs)

    folder_counts = Counter(r["folder"] for r in manifest_rows)
    summary = {
        "total_files_in_source": len(all_paths_rows),
        "unique_md5": len(groups),
        "classification_copies": len(manifest_rows),
        "talc_pairs": len(talc_pairs),
        "folders": dict(folder_counts),
    }
    with (index_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nDone: {OUT}")
    _write_readme()


def _write_readme() -> None:
    """Короткая подпись к dataset/."""
    text = """# dataset — Nornickel ore microscopy

Clean copy of `data/` with English folder labels for ML.

## Layout

| Path | Contents |
|------|----------|
| `source/` | Full mirror of `data/` (all files, all paths) |
| `classification/` | One canonical image per MD5, sorted by compound folder |
| `talc_segmentation/` | 41 pairs: `images/`, `annotated/`, `masks/`, `validation/` |
| `index/` | `paths.csv`, `manifest.csv`, `by_md5.json`, `talc_pairs.csv`, `summary.json` |

## Folder names (`classification/`)

Two axes joined with `__`:

- **Talc:** `talc_bearing` | `non_talc_bearing` | `talc_mixed`
- **Intergrowth:** `coarse` | `fine` | `coarse_and_fine` | `unknown_intergrowth`

Mapping: ch1 row ore ↔ ch2 intergrowth (`рядовые`→`coarse`, `тонкие`→`fine`).

## Rebuild

```bash
py scripts/build_dataset.py
py scripts/validate_talc_masks.py
```

After rebuild you can remove `data/`.

## Notes

- `non_talc_bearing` = folder label, not proof of zero talc.
- Talc masks: 41 annotated ch1 images only.
- Validation: `talc_segmentation/validation/*_validation.jpg` (3 panels in one file).
"""
    (OUT / "README.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
