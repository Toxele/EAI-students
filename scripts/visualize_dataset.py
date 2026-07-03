from __future__ import annotations

import argparse

from visualization.visualizer import DatasetVisualizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="artifacts/manifests/nornikel_manifest.csv")
    parser.add_argument("--output-dir", default="artifacts/visualizations")
    args = parser.parse_args()

    visualizer = DatasetVisualizer(args.manifest)
    visualizer.save_audit_bars(f"{args.output_dir}/audit_bars.png")
    visualizer.save_label_grid(f"{args.output_dir}/label_grid.png")
    print(f"saved visualizations to {args.output_dir}")


if __name__ == "__main__":
    main()

