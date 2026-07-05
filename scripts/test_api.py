"""Quick API test: health + analyze detail + analyze panorama.

Generates synthetic JPEGs in a temp directory so it does not depend on any
local dataset. Requires the API to be running (uvicorn app.main:app).

Run: py scripts/test_api.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import urllib.request
from pathlib import Path

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")  # sort_label_ru is Cyrillic; default Windows console codepage can't print it

BASE = "http://127.0.0.1:8000"


def synthetic_jpeg(path: Path, width: int, height: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    cv2.imwrite(str(path), image)


def post_analyze(path: Path) -> dict:
    """POST /analyze with a multipart file upload."""
    data = path.read_bytes()
    boundary = "----boundary"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
        f"Content-Type: image/jpeg\r\n\r\n"
    ).encode() + data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        f"{BASE}/analyze",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    resp = urllib.request.urlopen(req)
    return json.loads(resp.read().decode())


def main() -> None:
    r = urllib.request.urlopen(f"{BASE}/health")
    print("health:", r.read().decode())

    with tempfile.TemporaryDirectory() as tmp:
        detail = Path(tmp) / "detail.jpg"
        panorama = Path(tmp) / "panorama.jpg"
        synthetic_jpeg(detail, 1200, 900, seed=1)
        synthetic_jpeg(panorama, 9000, 6000, seed=2)

        for label, path in [("detail", detail), ("panorama", panorama)]:
            result = post_analyze(path)
            print(f"{label}: mode={result['mode']} sort={result['sort_label_ru']} id={result['result_id']}")
            oid = result["result_id"]
            ov = urllib.request.urlopen(f"{BASE}/overlay/{oid}")
            print(f"  overlay bytes: {len(ov.read())}")
            csv = urllib.request.urlopen(f"{BASE}/result/{oid}/csv")
            print(f"  csv lines: {len(csv.read().decode().splitlines())}")

    print("API OK")


if __name__ == "__main__":
    main()
