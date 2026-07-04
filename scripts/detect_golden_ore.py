from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from app.models.golden_ore_detector import GoldenOreDetector


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for golden ore detection."""
    parser = argparse.ArgumentParser(description="Detect golden ore inclusions algorithmically.")
    parser.add_argument("--input", help="Image file or directory.")
    parser.add_argument("--name", help="Find image by filename under dataset/, e.g. -21.jpg.")
    parser.add_argument("--names", help="Comma-separated filenames under dataset/, e.g. -1.jpg,-10.jpg.")
    parser.add_argument("--output", default="notebooks/ore_detection_yolo/outputs/golden_ore_detector")
    parser.add_argument("--recursive", action="store_true", help="Scan input directory recursively.")
    parser.add_argument("--limit", type=int, default=0, help="Optional maximum number of images.")
    parser.add_argument("--hue-min", type=int, default=12)
    parser.add_argument("--hue-max", type=int, default=48)
    parser.add_argument("--min-saturation", type=int, default=35)
    parser.add_argument("--value-percentile", type=float, default=78.0)
    parser.add_argument("--local-value-percentile", type=float, default=88.0)
    parser.add_argument("--min-box-gold-ratio", type=float, default=0.018)
    parser.add_argument("--min-knn-box-ratio", type=float, default=0.004)
    parser.add_argument("--min-knn-gold-ratio", type=float, default=0.25)
    parser.add_argument("--max-knn-distance", type=float, default=0.36)
    parser.add_argument("--component-close-size", type=int, default=3)
    parser.add_argument("--no-merge", action="store_true", help="Disable merging intersecting boxes.")
    parser.add_argument("--box-merge-gap", type=int, default=0)
    return parser.parse_args(_normalize_dash_prefixed_values(sys.argv[1:]))


def main() -> None:
    """Run the detector and save masks, overlays, YOLO-like labels, and a report."""
    args = parse_args()
    if not args.input and not args.name and not args.names:
        raise ValueError("Pass --input with a path, --name with one filename, or --names with filenames.")

    output_dir = Path(args.output)
    mask_dir = output_dir / "masks"
    overlay_dir = output_dir / "overlays"
    label_dir = output_dir / "labels"
    for directory in (mask_dir, overlay_dir, label_dir):
        directory.mkdir(parents=True, exist_ok=True)

    detector = GoldenOreDetector(
        hue_min=args.hue_min,
        hue_max=args.hue_max,
        min_saturation=args.min_saturation,
        value_percentile=args.value_percentile,
        local_value_percentile=args.local_value_percentile,
        min_box_gold_ratio=args.min_box_gold_ratio,
        min_knn_box_ratio=args.min_knn_box_ratio,
        min_knn_gold_ratio=args.min_knn_gold_ratio,
        max_knn_distance=args.max_knn_distance,
        component_close_size=args.component_close_size,
        merge_intersecting_boxes=not args.no_merge,
        box_merge_gap=args.box_merge_gap,
    )

    if args.names:
        image_paths = _collect_named_images([name.strip() for name in args.names.split(",") if name.strip()])
        source_for_error = args.names
    else:
        input_path = Path(args.name or args.input)
        image_paths = _collect_images(input_path, recursive=args.recursive)
        source_for_error = str(input_path)
    if args.limit > 0:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise FileNotFoundError(f"No images found at {source_for_error}")

    report_rows = []
    for image_path in tqdm(image_paths, desc="golden ore"):
        image_rgb = _read_rgb(image_path)
        result = detector.detect(image_rgb)

        safe_name = _safe_name(image_path)
        mask_path = mask_dir / f"{safe_name}.png"
        overlay_path = overlay_dir / f"{safe_name}.jpg"
        label_path = label_dir / f"{safe_name}.txt"

        _write_image(mask_path, result.mask)
        _write_image(overlay_path, cv2.cvtColor(result.overlay_rgb, cv2.COLOR_RGB2BGR))
        _write_yolo_labels(label_path, result.inclusions, image_rgb.shape[1], image_rgb.shape[0])

        report_rows.append(
            {
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "overlay_path": str(overlay_path),
                "label_path": str(label_path),
                "boxes": len(result.inclusions),
                "ore_percent": result.ore_percent,
                "width": image_rgb.shape[1],
                "height": image_rgb.shape[0],
            }
        )

    _write_report(output_dir / "report.csv", report_rows)
    print(f"Saved {len(report_rows)} detections to {output_dir}")


def _collect_images(path: Path, recursive: bool) -> list[Path]:
    """Collect image paths from a file or directory."""
    if path.is_file():
        return [path]
    if not path.exists() and path.name:
        matches = sorted(Path("dataset").rglob(path.name))
        if matches:
            return matches
    pattern = "**/*" if recursive else "*"
    return sorted(p for p in path.glob(pattern) if p.suffix.lower() in IMAGE_EXTENSIONS)


def _collect_named_images(names: list[str]) -> list[Path]:
    """Collect images by filename under dataset/ in the requested order."""
    output: list[Path] = []
    for name in names:
        matches = sorted(Path("dataset").rglob(name))
        output.extend(matches)
    return output


def _normalize_dash_prefixed_values(argv: list[str]) -> list[str]:
    """Allow --input -21.jpg and --name -21.jpg values that start with a dash."""
    normalized: list[str] = []
    idx = 0
    while idx < len(argv):
        item = argv[idx]
        if item in {"--input", "--name"} and idx + 1 < len(argv):
            normalized.append(f"{item}={argv[idx + 1]}")
            idx += 2
            continue
        normalized.append(item)
        idx += 1
    return normalized


def _read_rgb(path: Path) -> np.ndarray:
    """Read an RGB image from a unicode-safe filesystem path."""
    data = np.fromfile(str(path), dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _write_image(path: Path, image: np.ndarray) -> None:
    """Write an image to a unicode-safe filesystem path."""
    extension = path.suffix or ".png"
    ok, buffer = cv2.imencode(extension, image)
    if not ok:
        raise ValueError(f"Could not encode image: {path}")
    buffer.tofile(str(path))


def _write_yolo_labels(
    path: Path,
    inclusions: list,
    image_width: int,
    image_height: int,
) -> None:
    """Write detected boxes in YOLO txt format for optional downstream reuse."""
    lines = []
    for inclusion in inclusions:
        x, y, width, height = inclusion.bbox
        x_center = (x + width / 2) / image_width
        y_center = (y + height / 2) / image_height
        norm_width = width / image_width
        norm_height = height / image_height
        lines.append(f"0 {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _write_report(path: Path, rows: list[dict]) -> None:
    """Write a CSV report with one row per processed image."""
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _safe_name(path: Path) -> str:
    """Make a stable output filename stem from a source path."""
    parts = [part.replace(" ", "_") for part in path.with_suffix("").parts[-4:]]
    return "__".join(parts)


if __name__ == "__main__":
    main()
