from __future__ import annotations

import argparse

from training.data.annotations import CsvAnnotationStore
from training.data.manifest import read_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="artifacts/manifests/nornikel_manifest.csv")
    parser.add_argument("--output", default="artifacts/annotations/manual_labels.csv")
    parser.add_argument("--conflicts-only", action="store_true")
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    if args.conflicts_only:
        rows = [row for row in rows if row.get("label_conflict", "").lower() == "true"]
    CsvAnnotationStore(args.output).save_template([row["rel_path"] for row in rows])
    print(f"wrote annotation template: {args.output} rows={len(rows)}")


if __name__ == "__main__":
    main()
