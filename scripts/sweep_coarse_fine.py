"""
Hyperparameter sweep для coarse vs fine soft binary.

Запуск: py scripts/sweep_coarse_fine.py
"""
from __future__ import annotations

import copy
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hydra.json_config import JsonConfig
from trainers.coarse_fine_trainer import CoarseFineTrainer

SWEEP_ROOT = ROOT / "artifacts" / "runs" / "coarse_fine_sweep"
BASE_CONFIG = ROOT / "configs" / "classifier" / "coarse_fine_resnet18.json"

VARIANTS: list[dict] = [
    {"run_name": "resnet18_lr3e4_cosine", "model.name": "resnet18", "optimizer.lr": 0.0003, "scheduler.name": "cosine"},
    {"run_name": "resnet18_lr1e4_cosine", "model.name": "resnet18", "optimizer.lr": 0.0001, "scheduler.name": "cosine"},
    {"run_name": "resnet18_lr3e4_onecycle", "model.name": "resnet18", "optimizer.lr": 0.0003, "scheduler.name": "onecycle"},
    {"run_name": "resnet34_lr3e4_cosine", "model.name": "resnet34", "optimizer.lr": 0.0003, "scheduler.name": "cosine"},
    {"run_name": "resnet34_lr1e4_cosine", "model.name": "resnet34", "optimizer.lr": 0.0001, "scheduler.name": "cosine"},
    {"run_name": "resnet50_lr3e4_cosine", "model.name": "resnet50", "optimizer.lr": 0.0003, "scheduler.name": "cosine"},
    {"run_name": "resnet50_lr1e4_step", "model.name": "resnet50", "optimizer.lr": 0.0001, "scheduler.name": "step", "scheduler.step_size": 10},
    {"run_name": "effnet_b0_lr18e4_cosine", "model.name": "efficientnet_b0", "optimizer.lr": 0.00018, "scheduler.name": "cosine"},
    {"run_name": "convnext_tiny_lr3e4_cosine", "model.name": "convnext_tiny", "optimizer.lr": 0.0003, "scheduler.name": "cosine"},
    {"run_name": "resnet18_lr3e4_cosine_448", "model.name": "resnet18", "optimizer.lr": 0.0002, "scheduler.name": "cosine", "data.image_size": 448, "data.batch_size": 12},
]


def flatten_overrides(variant: dict) -> list[str]:
    overrides: list[str] = []
    for key, value in variant.items():
        if key == "run_name":
            continue
        overrides.append(f"{key}={json.dumps(value)}")
    return overrides


def main() -> None:
    SWEEP_ROOT.mkdir(parents=True, exist_ok=True)
    leaderboard: list[dict] = []

    for variant in VARIANTS:
        run_name = variant["run_name"]
        run_dir = SWEEP_ROOT / run_name
        overrides = flatten_overrides(variant) + [
            f"run_dir={json.dumps(str(run_dir))}",
            f"run_name={json.dumps(run_name)}",
            "trainer.epochs=18",
            "trainer.early_stopping_patience=5",
        ]
        config = JsonConfig.load(BASE_CONFIG).merged(overrides)
        cfg = config.to_dict()
        run_dir.mkdir(parents=True, exist_ok=True)
        config.save_resolved(run_dir / "resolved_config.json")

        print(f"\n{'=' * 60}\nSWEEP: {run_name}\n{'=' * 60}")
        try:
            summary = CoarseFineTrainer(cfg).fit()
            best_metrics_path = run_dir / "best_metrics.json"
            metrics = json.loads(best_metrics_path.read_text(encoding="utf-8")) if best_metrics_path.exists() else {}
            row = {
                "run_name": run_name,
                "composite_score": summary.get("best_composite_score", 0.0),
                "clean_f1": metrics.get("clean_f1", float("nan")),
                "clean_auc": metrics.get("clean_auc", float("nan")),
                "clean_accuracy": metrics.get("clean_accuracy", float("nan")),
                "ambiguous_mae": metrics.get("ambiguous_mae", float("nan")),
                "epochs": summary.get("epochs", 0),
                "status": "ok",
            }
        except Exception as exc:
            print(f"FAILED {run_name}: {exc}")
            row = {"run_name": run_name, "status": f"error: {exc}"}

        leaderboard.append(row)
        _write_leaderboard(leaderboard)

    best = max(
        (r for r in leaderboard if r.get("status") == "ok"),
        key=lambda r: r.get("composite_score", -1),
        default=None,
    )
    if best:
        print(f"\nBEST: {best['run_name']} composite={best['composite_score']:.4f} clean_f1={best.get('clean_f1')}")
        _copy_best(best["run_name"])


def _write_leaderboard(rows: list[dict]) -> None:
    path = SWEEP_ROOT / "leaderboard.csv"
    fields = ["run_name", "composite_score", "clean_f1", "clean_auc", "clean_accuracy", "ambiguous_mae", "epochs", "status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _copy_best(run_name: str) -> None:
    import shutil

    src = SWEEP_ROOT / run_name
    dst = ROOT / "artifacts" / "runs" / "coarse_fine_best"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    (dst / "best_run_name.txt").write_text(run_name, encoding="utf-8")


if __name__ == "__main__":
    main()
