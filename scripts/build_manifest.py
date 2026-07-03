from __future__ import annotations

import argparse

from data.manifest import NornikelManifestBuilder, read_manifest, write_manifest
from data.splits import GroupedStratifiedSplitter
from hydra.json_config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data/nornikel_manifest.json")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    builder = NornikelManifestBuilder.from_config(cfg)
    rows = builder.write_csv(cfg["output_csv"])
    dict_rows = read_manifest(cfg["output_csv"])
    split_rows = GroupedStratifiedSplitter(seed=42).split(dict_rows)
    write_manifest(cfg["output_csv"], split_rows)
    conflicts = sum(str(row.label_conflict).lower() == "true" for row in rows)
    duplicates = sum(row.duplicate_count > 1 for row in rows)
    print(f"wrote {cfg['output_csv']}: rows={len(rows)} duplicates={duplicates} conflicts={conflicts}")


if __name__ == "__main__":
    main()

