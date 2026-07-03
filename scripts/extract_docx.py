from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


def extract_text(xml_bytes: bytes) -> str:
    root = ET.fromstring(xml_bytes)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs: list[str] = []
    for paragraph in root.findall(".//w:p", ns):
        parts: list[str] = []
        for node in paragraph.iter():
            if node.tag == f"{{{ns['w']}}}t" and node.text:
                parts.append(node.text)
            elif node.tag == f"{{{ns['w']}}}tab":
                parts.append("\t")
            elif node.tag == f"{{{ns['w']}}}br":
                parts.append("\n")
        text = "".join(parts).strip()
        if text:
            paragraphs.append(text)
    return "\n".join(paragraphs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("docx")
    parser.add_argument("--output-dir", default="artifacts/docx_extract")
    args = parser.parse_args()

    docx = Path(args.docx)
    out = Path(args.output_dir)
    media = out / "media"
    media.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(docx) as archive:
        text = extract_text(archive.read("word/document.xml"))
        (out / "document.txt").write_text(text, encoding="utf-8")
        for name in archive.namelist():
            if name.startswith("word/media/"):
                target = media / Path(name).name
                with archive.open(name) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
