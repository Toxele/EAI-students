from __future__ import annotations

import argparse

from data.annotations import CsvAnnotationStore
from data.manifest import read_manifest, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="artifacts/manifests/nornikel_manifest.csv")
    parser.add_argument("--annotations", default="artifacts/annotations/manual_labels.csv")
    parser.add_argument("--output", default="artifacts/manifests/nornikel_manifest_annotated.csv")
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    annotated = CsvAnnotationStore(args.annotations).apply_to_manifest(rows)
    write_manifest(args.output, annotated)
    changed = sum(row.get("manual_label") == "true" for row in annotated)
    print(f"wrote {args.output}: manual_labels={changed}")


if __name__ == "__main__":
    main()

