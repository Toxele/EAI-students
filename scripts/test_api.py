"""Быстрый тест API: health + analyze detail + panorama."""
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import CH1_DATA_DIR, DATA_DIR


BASE = "http://127.0.0.1:8000"


def post_analyze(path: Path) -> dict:
    """POST /analyze с multipart upload."""
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

    detail = CH1_DATA_DIR / "Рядовые руды" / "2539590-1.JPG"
    pan = DATA_DIR / "Панорамы" / "4.jpg"

    for label, path in [("detail", detail), ("panorama", pan)]:
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
