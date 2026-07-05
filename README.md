# Head (disguise from another participant)

AI pipeline that analyzes microphotographs of polished ore sections and
classifies each sample by expert rule (not a black-box classifier). It
segments sulfide inclusions into **ordinary** and **fine** intergrowths plus
**talc**, computes area percentages, and applies:

```text
if talc_percent > 10%:
    ore = "Talc ore"
else:
    if ordinary_percent > fine_percent:
        ore = "Ordinary ore"
    else:
        ore = "Hard-to-beneficiate ore"
```

## Architecture

```text
app/            FastAPI backend — pipeline, models, HTTP API
web/            React + OpenSeadragon frontend (zoom/pan, layer toggles, corrections)
scripts/        Product CLI tools (weight download, smoke test)
weights/        Model weight files (not in git, see weights/README.md)
docs/           User guide and app documentation
training/       Model training / research pipeline (see training/README.md)
```

Request flow: upload → `mode_detector` (panorama vs. detail) → grain/talc
detection → rule engine → metrics + overlay + PDF/CSV report.

## Requirements

- Python 3.11+
- Node.js 18+ (for the React frontend)
- CPU works; a CUDA GPU speeds up inference but is not required

## Installation

```bash
pip install -r requirements.txt
cd web && npm install && cd ..
```

## Model weights

The app needs two trained checkpoints in `weights/`:

```text
weights/segmentator.pt   # Unet++ talc segmentation
weights/classifier.pt    # coarse/fine (ordinary vs. thin) classifier
```

Download both from Google Drive:

```bash
pip install gdown
python scripts/download_weights.py
```

Without weights, the app still runs — it falls back to a fixed placeholder
layout for grains/talc and a heuristic for the classifier, so the UI never
crashes, but results are not meaningful. See `weights/README.md` and
`training/README.md` if you need to retrain them.

## Running the app

```bash
# Backend (FastAPI)
uvicorn app.main:app --reload --port 8000

# Frontend (React), in a second terminal
cd web && npm run dev
# -> http://127.0.0.1:5173
```

On Windows, `.\scripts\start.ps1` launches both.

Verify the backend works end-to-end without a browser:

```bash
python scripts/smoke_test.py     # runs the pipeline directly on synthetic images
python scripts/test_api.py       # hits a running API over HTTP
```

## Using the app

1. Open `http://127.0.0.1:5173` and upload a TIFF/PNG/JPEG (up to 10000x10000 px).
2. The backend auto-detects panorama vs. detail mode and returns the ore
   classification, per-class percentages, and colored overlay layers
   (green = ordinary, red = fine, blue = talc).
3. Zoom/pan the image, correct misclassified grains or the talc mask if
   needed (metrics recompute automatically), and export CSV, PDF, or the
   overlay PNG.

See [docs/USER_GUIDE.md](docs/USER_GUIDE.md) for a walkthrough with worked
examples, including edge cases (borderline talc %, low-confidence
detections, panorama-only limitations).

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/analyze` | Upload (`file`, `mode=panorama\|detail`) |
| POST | `/result/{id}/corrections` | Apply grain edits, recompute metrics |
| POST | `/result/{id}/talc-mask` | Apply a manually edited talc mask |
| GET | `/overlay/{id}` | Combined overlay image |
| GET | `/result/{id}/image/original` | Original image |
| GET | `/result/{id}/layer/talc-colored` | Talc layer |
| GET | `/result/{id}/layer/type` | Intergrowth type layer |
| GET | `/result/{id}/labels.json` | Grain labels |
| GET | `/result/{id}/csv` | Metrics CSV |
| GET | `/result/{id}/pdf` | PDF report |

## Batch mode

Analyze every image in a folder via the CLI. Each file gets the same outputs
(overlay JPG, CSV, labels.json) that the API produces, written to `results/`,
plus a combined summary:

```bash
python scripts/batch_analyze.py path/to/folder
# -> results/batch_summary.csv
```

## Logging

Each analysis is persisted as a JSON state file in `results/{id}.json`,
recording the filename, processing mode, model readiness (whether trained
weights were loaded or the heuristic fallback was used), computed metrics,
and the final classification.
