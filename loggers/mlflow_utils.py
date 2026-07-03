from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import quote
from typing import Any


class MlflowRun:
    def __init__(self, cfg: dict[str, Any] | None, run_name: str | None = None) -> None:
        self.cfg = cfg or {}
        self.run_name = run_name
        self.enabled = bool(self.cfg.get("enabled", False))
        self.mlflow = None
        self.active = False

    def __enter__(self) -> "MlflowRun":
        if not self.enabled:
            return self
        try:
            import mlflow
        except ImportError as exc:  # pragma: no cover
            raise ImportError("MLflow logging requested, but `mlflow` is not installed.") from exc
        self.mlflow = mlflow
        if self.cfg.get("tracking_uri"):
            tracking_uri = self._resolve_tracking_uri(str(self.cfg["tracking_uri"]))
            if tracking_uri.startswith("file:"):
                os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
            mlflow.set_tracking_uri(tracking_uri)
        if self.cfg.get("experiment"):
            mlflow.set_experiment(self.cfg["experiment"])
        mlflow.start_run(run_name=self.run_name or self.cfg.get("run_name"))
        self.active = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.active and self.mlflow is not None:
            self.mlflow.end_run()
        self.active = False

    def log_params_flat(self, params: dict[str, Any], prefix: str = "") -> None:
        if not self.active or self.mlflow is None:
            return
        flat = self._flatten(params, prefix=prefix)
        for key, value in flat.items():
            if isinstance(value, (dict, list, tuple)):
                value = json.dumps(value, ensure_ascii=False)
            self.mlflow.log_param(key[:250], value)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None, prefix: str = "") -> None:
        if not self.active or self.mlflow is None:
            return
        clean = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                clean[f"{prefix}{key}"] = float(value)
        if clean:
            self.mlflow.log_metrics(clean, step=step)

    def log_artifact(self, path: str | Path, artifact_path: str | None = None) -> None:
        if self.active and self.mlflow is not None and Path(path).exists():
            self.mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def log_artifacts(self, path: str | Path, artifact_path: str | None = None) -> None:
        if self.active and self.mlflow is not None and Path(path).exists():
            self.mlflow.log_artifacts(str(path), artifact_path=artifact_path)

    @classmethod
    def _flatten(cls, params: dict[str, Any], prefix: str = "") -> dict[str, Any]:
        output = {}
        for key, value in params.items():
            full_key = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, dict):
                output.update(cls._flatten(value, full_key))
            else:
                output[full_key] = value
        return output

    @staticmethod
    def _resolve_tracking_uri(uri: str) -> str:
        if uri.startswith("file:") and not uri.startswith("file:/"):
            path = Path(uri.removeprefix("file:")).resolve()
            return f"file:///{quote(path.as_posix(), safe='/:')}"
        return uri
