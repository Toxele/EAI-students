# User Guide

A walkthrough of the Nornickel Ore Analyzer for geologists and lab
technicians, with worked examples of typical and edge-case classifications.
For installation and API reference, see the root [README.md](../README.md).

## 1. Uploading an image

Open the web UI (`http://127.0.0.1:5173`) and upload a TIFF, PNG, or JPEG —
either a full panorama of a polished section or a single detail (close-up)
optical microscope (OM) shot.

The backend picks a processing **mode** automatically from pixel count:

- **panorama** — more than 50 megapixels (`app/config.py:PANORAMA_PIXEL_THRESHOLD`).
  Sulfide grains are found tile by tile (`PANORAMA_TILE_SIZE` = 2500 px, with
  a 256 px context margin to avoid seam artifacts at tile borders).
- **detail** — everything else. The whole frame is processed at once.

You can also force a mode explicitly (`mode=panorama|detail` on `/analyze`)
if the automatic guess is wrong for an unusual image.

## 2. Reading the output

- **Overlay** — green boxes = ordinary intergrowths, red boxes = fine (thin)
  intergrowths, blue mask = talc.
- **Metrics table** — sulfide / ordinary / fine / talc area percentages, plus
  grain counts (`k` = total grains, `l` = ordinary, `j` = thin).
- **Conclusion** — one sentence in Russian stating the ore grade and the
  measurements behind it, e.g. *"Руда классифицирована как **оталькованная**:
  содержание талька — 14.2%, преобладание тонких срастаний — 62%."*

## 3. The classification rule

```text
if talc_percent > 10%:
    ore = "Talc ore"
else:
    if ordinary_percent > fine_percent:
        ore = "Ordinary ore"
    else:
        ore = "Hard-to-beneficiate ore"
```

This is a fixed expert rule (`app/pipeline/rule_engine.py`), not a black-box
model decision — the same numbers always produce the same grade, and the
"why" is always in the conclusion text.

## 4. Typical cases

### Ordinary ore

Talc 3%, ordinary intergrowths 72%, fine intergrowths 28%.

Talc is below the 10% threshold, and ordinary (72%) exceeds fine (28%) →
**Ordinary ore**.

### Hard-to-beneficiate ore

Talc 4%, ordinary intergrowths 35%, fine intergrowths 65%.

Talc is below threshold, but fine (65%) exceeds ordinary (35%) →
**Hard-to-beneficiate ore**.

### Talc ore

Talc 22%, ordinary intergrowths 40%, fine intergrowths 60%.

Talc exceeds 10%, so the sample is **Talc ore** regardless of the
intergrowth split — the ordinary/fine numbers are still reported for
context, but they don't affect the grade once talc dominates.

## 5. Edge cases

### Talc percentage right at the threshold

The rule is a **strict** `>`, not `>=`. A sample measured at exactly
**10.0% talc** falls through to the intergrowth rule (not classified as talc
ore); **10.1% talc** is classified as talc ore. If your measured value is
within a fraction of a percent of 10%, treat the grade as borderline and
double-check the talc mask visually — segmentation noise of a few tenths of
a percent can flip the result. The talc layer can be corrected manually
(see §7) if the mask is off.

### Ordinary and fine intergrowths tied

If `ordinary_percent == thin_percent` exactly, the rule (`ordinary_percent
>= thin_percent`) resolves the tie in favor of **Ordinary ore**. This also
covers the "no grains at all" case below, since it produces a 50/50 split.

### No sulfide grains detected

When the detector finds zero grains, the pipeline reports
`sulfide_percent = 0%` and an ordinary/fine split of **50%/50%** (a
deliberate default, not a measurement) — which the tie-break above turns
into **Ordinary ore**. This is a real edge case worth knowing: an empty or
non-ore image does not raise an error or return "unknown," it silently
reports Ordinary ore at 0% sulfides. Always check the sulfide percentage
alongside the grade — 0% sulfides with an "Ordinary ore" verdict likely means
no grains were found, not that the sample is a clean ordinary ore.

### Model weights not loaded

Both the talc segmenter and the coarse/fine classifier fall back to a
non-ML heuristic when their weight files (`weights/segmentator.pt`,
`weights/classifier.pt`) are missing, so the app never crashes for lack of
weights. The API response's `classifier_match` field says which one ran,
e.g. *"эвристика (веса не загружены)"* ("heuristic, weights not loaded")
instead of *"классификатор"* ("classifier"). Treat results produced under
the heuristic as indicative only — download the real weights
(`python scripts/download_weights.py`) before relying on the output.

### Panorama talc layer without a trained segmenter

If the talc segmenter isn't ready, panorama mode falls back to a fixed
placeholder talc shape scaled to the image (not a real measurement). This
exists so the UI and report pipeline stay testable without weights — it is
not meant to be read as a talc estimate. Always confirm `classifier_match`
and the segmenter's ready state before trusting a panorama talc percentage.

### Corrupted or unsupported file

Uploading a file that isn't a decodable image returns an HTTP 400 with
`"Некорректный файл изображения"` ("invalid image file") rather than a
silent misclassification.

## 6. Very large panoramas

Panoramas up to 10000x10000 px are supported by tiling — the tradeoff is
that each tile only sees a limited context margin (256 px) around it. Very
small or very elongated grains that straddle several tile borders can be
undercounted; if grain counts look low on a huge panorama, inspect the
overlay near tile boundaries.

## 7. Correcting results

Both grain classifications and the talc mask can be corrected in the UI
without re-running the model:

- **Grain edits** (`POST /result/{id}/corrections`) — change a grain's
  status to `ordinary`, `thin`, `uncertain`, or `false_positive`, or adjust
  its bounding box. Metrics and the grade recompute immediately from the
  edited grain list.
- **Talc mask edits** (`POST /result/{id}/talc-mask`) — draw over the talc
  layer directly; the edited mask is treated as ground truth (100%
  confidence) and the grade recomputes from the new talc percentage.

Use this when you disagree with a specific detection rather than the
overall grade — e.g. marking a scratch or artifact that was picked up as a
grain as `false_positive` removes it from all percentage calculations.

## 8. Exporting

- **CSV** (`/result/{id}/csv`) — one row of metrics for spreadsheets.
- **PDF** (`/result/{id}/pdf`) — a one-page report with the conclusion,
  metrics table, and overlay images.
- **Overlay PNG** (`/overlay/{id}`) — the combined image layer.
- **Batch mode** (`python scripts/batch_analyze.py <folder>`) — runs every
  image in a folder and writes a `batch_summary.csv` alongside each file's
  usual outputs. Useful for processing an entire sample set overnight.
