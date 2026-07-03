from __future__ import annotations

import argparse
import json

from data.color_domains import ColorDomainAnalyzer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="artifacts/manifests/nornikel_manifest.csv")
    parser.add_argument("--output-dir", default="artifacts/eda/color_domains")
    parser.add_argument("--thumbnail-size", type=int, default=256)
    parser.add_argument("--max-examples-per-group", type=int, default=24)
    args = parser.parse_args()

    summary = ColorDomainAnalyzer(
        manifest_csv=args.manifest,
        output_dir=args.output_dir,
        thumbnail_size=args.thumbnail_size,
        max_examples_per_group=args.max_examples_per_group,
    ).run()
    print(json.dumps(summary["by_group"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
