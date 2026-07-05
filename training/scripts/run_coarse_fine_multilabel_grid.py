"""
Grid search для coarse/fine multi-label классификатора.

Запуск: py scripts/run_coarse_fine_multilabel_grid.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE_CONFIG = ROOT / "configs" / "classifier" / "coarse_fine_multilabel.json"
GRID_OUT = ROOT / "artifacts" / "runs" / "coarse_fine_multilabel" / "grid_summary.json"

EXPERIMENTS = [
    {"model.name": "resnet18", "optimizer.lr": 0.0003, "scheduler.name": "cosine", "run_suffix": "resnet18_lr3e4_cosine"},
    {"model.name": "resnet18", "optimizer.lr": 0.0001, "scheduler.name": "cosine", "run_suffix": "resnet18_lr1e4_cosine"},
    {"model.name": "resnet18", "optimizer.lr": 0.0003, "scheduler.name": "onecycle", "run_suffix": "resnet18_lr3e4_onecycle"},
    {"model.name": "resnet34", "optimizer.lr": 0.0002, "scheduler.name": "cosine", "run_suffix": "resnet34_lr2e4_cosine"},
    {"model.name": "resnet34", "optimizer.lr": 0.0001, "scheduler.name": "cosine", "run_suffix": "resnet34_lr1e4_cosine"},
    {"model.name": "efficientnet_b0", "optimizer.lr": 0.00018, "scheduler.name": "cosine", "run_suffix": "effnet_b0_lr18e4_cosine"},
    {"model.name": "efficientnet_b0", "optimizer.lr": 0.0001, "scheduler.name": "onecycle", "run_suffix": "effnet_b0_lr1e4_onecycle"},
    {"model.name": "convnext_tiny", "optimizer.lr": 0.00015, "scheduler.name": "cosine", "run_suffix": "convnext_tiny_lr15e4_cosine"},
]


def run_experiment(overrides: dict[str, object]) -> dict:
    """Один прогон train_multilabel.py с overrides."""
    suffix = str(overrides.pop("run_suffix"))
    run_dir = ROOT / "artifacts" / "runs" / "coarse_fine_multilabel" / suffix
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "train_multilabel.py"),
        "--config",
        str(BASE_CONFIG),
        f"run_dir={run_dir.as_posix()}",
        f"run_name={suffix}",
    ]
    for key, value in overrides.items():
        if isinstance(value, bool):
            rendered = "true" if value else "false"
        else:
            rendered = str(value)
        cmd.append(f"{key}={rendered}")

    print("\n" + "=" * 60)
    print("RUN:", suffix)
    print(" ".join(cmd))
    print("=" * 60)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    result = subprocess.run(cmd, cwd=ROOT, check=False, env=env)
    summary_path = run_dir / "summary.json"
    payload: dict = {"run": suffix, "run_dir": str(run_dir), "exit_code": result.returncode}
    if summary_path.exists():
        payload.update(json.loads(summary_path.read_text(encoding="utf-8")))
        best_metrics_path = run_dir / "best_metrics.json"
        if best_metrics_path.exists():
            payload["best_metrics"] = json.loads(best_metrics_path.read_text(encoding="utf-8"))
    return payload


def main() -> None:
    results = []
    for spec in EXPERIMENTS:
        results.append(run_experiment(dict(spec)))

    ranked = sorted(
        results,
        key=lambda item: item.get("best_val_ig_macro_f1", item.get("best_val_macro_f1", -1)),
        reverse=True,
    )
    GRID_OUT.parent.mkdir(parents=True, exist_ok=True)
    GRID_OUT.write_text(json.dumps({"results": ranked}, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 60)
    print("GRID SUMMARY (by ig_macro_f1)")
    print("=" * 60)
    for row in ranked:
        ig_f1 = row.get("best_val_ig_macro_f1", "n/a")
        macro_f1 = row.get("best_val_macro_f1", "n/a")
        print(f"  {row['run']}: ig_macro_f1={ig_f1} macro_f1={macro_f1} exit={row.get('exit_code')}")
    print(f"\nSaved: {GRID_OUT}")


if __name__ == "__main__":
    main()
