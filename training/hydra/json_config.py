from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class JsonConfig:
    data: dict[str, Any]
    path: Path | None = None

    @classmethod
    def load(cls, path: str | Path) -> "JsonConfig":
        config_path = cls._resolve_path(Path(path))
        with config_path.open("r", encoding="utf-8") as f:
            return cls(json.load(f), config_path)

    def merged(self, overrides: Iterable[str] | None = None) -> "JsonConfig":
        data = copy.deepcopy(self.data)
        for item in overrides or []:
            if "=" not in item:
                raise ValueError(f"Override must be key=value, got: {item}")
            key, raw_value = item.split("=", 1)
            self._set_nested(data, key.split("."), self._parse_value(raw_value))
        return JsonConfig(data, self.path)

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.data)

    def save_resolved(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _resolve_path(path: Path) -> Path:
        if path.exists() or path.is_absolute():
            return path
        for parent in [Path.cwd(), *Path.cwd().parents]:
            candidate = parent / path
            if candidate.exists():
                return candidate
        return path

    @staticmethod
    def _set_nested(data: dict[str, Any], keys: list[str], value: Any) -> None:
        cursor = data
        for key in keys[:-1]:
            if key not in cursor or not isinstance(cursor[key], dict):
                cursor[key] = {}
            cursor = cursor[key]
        cursor[keys[-1]] = value

    @staticmethod
    def _parse_value(value: str) -> Any:
        lowered = value.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        if lowered == "null":
            return None
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value


def load_config(path: str | Path, overrides: Iterable[str] | None = None) -> dict[str, Any]:
    return JsonConfig.load(path).merged(overrides).to_dict()
