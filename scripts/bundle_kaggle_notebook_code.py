"""
Устарело: Kaggle-notebooks self-contained (см. kaggle/train_coarse_fine_*.ipynb).

Раньше встраивал модули в multilabel notebook. Оставлено для совместимости;
предпочтительно править ipynb напрямую.

Запуск: py scripts/bundle_kaggle_notebook_code.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "kaggle" / "train_coarse_fine_multilabel.ipynb"

MODULES = [
    "hydra/__init__.py",
    "hydra/json_config.py",
    "loggers/mlflow_utils.py",
    "data/datasets.py",
    "data/multilabel_dataset.py",
    "models/classifiers.py",
    "trainers/multilabel_trainer.py",
]

CONFIG = "configs/classifier/coarse_fine_multilabel_kaggle_fast.json"

SETUP_CELL = '''import json
import sys
from pathlib import Path

_BUNDLED = json.loads(r"""__BUNDLED_JSON__""")


def setup_project_code(work_dir: Path) -> Path:
    """Распаковывает модули проекта в work_dir (Kaggle без code zip)."""
    root = work_dir / "nornickel_code"
    for rel, content in _BUNDLED["modules"].items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    cfg_dir = root / "configs" / "classifier"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "coarse_fine_multilabel_kaggle_fast.json").write_text(
        _BUNDLED["config"], encoding="utf-8"
    )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


WORK_DIR = Path("/kaggle/working") if Path("/kaggle/input").is_dir() else Path("../dataset/kaggle/runs")
WORK_DIR.mkdir(parents=True, exist_ok=True)
CODE_ROOT = setup_project_code(WORK_DIR)
BASE_CONFIG_PATH = CODE_ROOT / "configs/classifier/coarse_fine_multilabel_kaggle_fast.json"
print("CODE_ROOT:", CODE_ROOT)
'''


def load_notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_notebook(path: Path, nb: dict) -> None:
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")


def build_bundle() -> dict:
    modules: dict[str, str] = {}
    for rel in MODULES:
        src = ROOT / rel
        if not src.is_file():
            raise FileNotFoundError(src)
        modules[rel.replace("\\", "/")] = src.read_text(encoding="utf-8")
    config = (ROOT / CONFIG).read_text(encoding="utf-8")
    return {"modules": modules, "config": config}


def main() -> None:
    bundle = build_bundle()
    bundled_json = json.dumps(bundle, ensure_ascii=False)
    setup_source = SETUP_CELL.replace("__BUNDLED_JSON__", bundled_json.replace("\\", "\\\\").replace('"""', '\\"\\"\\"'))

    nb = load_notebook(NOTEBOOK)
    setup_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in setup_source.splitlines()],
    }

    # replace cell after hyperparams (index 3) or insert setup cell
    inserted = False
    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") == "code" and any("setup_project_code" in line for line in cell.get("source", [])):
            nb["cells"][i] = setup_cell
            inserted = True
            break
    if not inserted:
        nb["cells"].insert(3, setup_cell)

    # strip find_code_root from following cells — patch imports cell
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "find_code_root" in src:
            new_src = src.replace(
                "CODE_ROOT = find_code_root()\nDATA_ROOT = find_under_input(\"coarse_fine\", \"manifest.csv\")\nif str(CODE_ROOT) not in sys.path:\n    sys.path.insert(0, str(CODE_ROOT))\n\n",
                "DATA_ROOT = find_under_input(\"coarse_fine\", \"manifest.csv\")\n\n",
            )
            cell["source"] = [line + "\n" for line in new_src.splitlines()]

        if "BASE_CONFIG = " in src and "run_one" in src:
            new_src = src.replace(
                'BASE_CONFIG = "configs/classifier/coarse_fine_multilabel_kaggle_fast.json"\n',
                "",
            ).replace(
                "cfg_path = CODE_ROOT / BASE_CONFIG\n    if not cfg_path.is_file():\n        cfg_path = Path(BASE_CONFIG)\n",
                "cfg_path = BASE_CONFIG_PATH\n",
            )
            cell["source"] = [line + "\n" for line in new_src.splitlines()]

    save_notebook(NOTEBOOK, nb)
    print(f"Updated {NOTEBOOK} ({len(bundle['modules'])} modules, config embedded)")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
