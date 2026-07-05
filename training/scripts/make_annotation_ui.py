from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from urllib.parse import quote


def file_url(path: str) -> str:
    resolved = Path(path).resolve()
    return "file:///" + quote(str(resolved).replace("\\", "/"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="artifacts/manifests/nornikel_manifest.csv")
    parser.add_argument("--output", default="artifacts/annotation_ui/index.html")
    parser.add_argument("--conflicts-only", action="store_true")
    parser.add_argument("--source", default="classification")
    args = parser.parse_args()

    with Path(args.manifest).open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows = [row for row in rows if row.get("source") == args.source]
    if args.conflicts_only:
        rows = [row for row in rows if row.get("label_conflict", "").lower() == "true"]

    items = [
        {
            "rel_path": row["rel_path"],
            "path": row["path"],
            "url": file_url(row["path"]),
            "label": row["label"],
            "conflict": row.get("label_conflict", "").lower() == "true",
            "duplicate_group": row.get("duplicate_group", ""),
            "size": f"{row.get('width')}x{row.get('height')}",
        }
        for row in rows
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    data_json = json.dumps(items, ensure_ascii=False)
    output.write_text(
        f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Nornickel Annotation UI</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #101418; color: #e8edf2; }}
    .app {{ display: grid; grid-template-columns: 320px 1fr; min-height: 100vh; }}
    aside {{ padding: 16px; background: #171d24; border-right: 1px solid #2a333d; overflow: auto; }}
    main {{ display: grid; grid-template-rows: auto 1fr auto; min-width: 0; }}
    .toolbar {{ display: flex; gap: 8px; align-items: center; padding: 12px; border-bottom: 1px solid #2a333d; flex-wrap: wrap; }}
    button, select {{ background: #26313d; color: #e8edf2; border: 1px solid #3b4856; border-radius: 6px; padding: 8px 10px; cursor: pointer; }}
    button:hover {{ background: #334252; }}
    button.active {{ outline: 2px solid #6aa6ff; }}
    .image-wrap {{ display: flex; align-items: center; justify-content: center; overflow: auto; padding: 16px; }}
    img {{ max-width: 100%; max-height: calc(100vh - 170px); object-fit: contain; background: #050608; }}
    .meta {{ padding: 12px; border-top: 1px solid #2a333d; font-size: 14px; word-break: break-word; }}
    .list {{ display: grid; gap: 6px; }}
    .item {{ padding: 8px; border: 1px solid #2a333d; border-radius: 6px; cursor: pointer; }}
    .item.current {{ border-color: #6aa6ff; }}
    .item.done {{ background: #183326; }}
    .badge {{ color: #ffcc66; }}
    .muted {{ color: #9aa7b4; }}
    textarea {{ width: 100%; height: 80px; background: #0d1116; color: #e8edf2; border: 1px solid #3b4856; border-radius: 6px; }}
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <h3>Annotation UI</h3>
      <p class="muted">Классы сохраняются в localStorage. В конце нажми Export CSV.</p>
      <div id="stats"></div>
      <hr />
      <div class="list" id="list"></div>
    </aside>
    <main>
      <div class="toolbar">
        <button onclick="prevItem()">← Prev</button>
        <button onclick="nextItem()">Next →</button>
        <button onclick="setLabel('ordinary')">ordinary</button>
        <button onclick="setLabel('thin')">thin</button>
        <button onclick="setLabel('talc')">talc</button>
        <button onclick="setLabel('skip')">skip</button>
        <button onclick="exportCsv()">Export CSV</button>
      </div>
      <div class="image-wrap"><img id="image" alt=""></div>
      <div class="meta">
        <div id="meta"></div>
        <textarea id="comment" placeholder="Комментарий к разметке" oninput="saveComment()"></textarea>
      </div>
    </main>
  </div>
  <script>
    const items = {data_json};
    const key = "nornickel_annotations_v1";
    let annotations = JSON.parse(localStorage.getItem(key) || "{{}}");
    let index = 0;

    function save() {{ localStorage.setItem(key, JSON.stringify(annotations)); }}
    function current() {{ return items[index]; }}
    function ann(item) {{ return annotations[item.rel_path] || {{ label: "", comment: "" }}; }}
    function setLabel(label) {{
      const item = current();
      annotations[item.rel_path] = {{ ...ann(item), label }};
      save();
      render();
      nextItem();
    }}
    function saveComment() {{
      const item = current();
      annotations[item.rel_path] = {{ ...ann(item), comment: document.getElementById("comment").value }};
      save();
      renderList();
      renderStats();
    }}
    function prevItem() {{ index = Math.max(0, index - 1); render(); }}
    function nextItem() {{ index = Math.min(items.length - 1, index + 1); render(); }}
    function render() {{
      const item = current();
      const a = ann(item);
      document.getElementById("image").src = item.url;
      document.getElementById("meta").innerHTML =
        `<div><b>${{index + 1}} / ${{items.length}}</b></div>` +
        `<div>folder label: <b>${{item.label}}</b> ${{item.conflict ? '<span class="badge">CONFLICT</span>' : ''}}</div>` +
        `<div>manual label: <b>${{a.label || '-'}}</b></div>` +
        `<div>size: ${{item.size}}, duplicate group: ${{item.duplicate_group || '-'}}</div>` +
        `<div class="muted">${{item.rel_path}}</div>`;
      document.getElementById("comment").value = a.comment || "";
      renderList();
      renderStats();
    }}
    function renderList() {{
      const list = document.getElementById("list");
      list.innerHTML = "";
      items.forEach((item, i) => {{
        const a = ann(item);
        const div = document.createElement("div");
        div.className = `item ${{i === index ? 'current' : ''}} ${{a.label ? 'done' : ''}}`;
        div.onclick = () => {{ index = i; render(); }};
        div.innerHTML = `<b>${{i + 1}}.</b> ${{a.label || item.label}} ${{item.conflict ? '<span class="badge">!</span>' : ''}}<br><span class="muted">${{item.rel_path.split('/').pop()}}</span>`;
        list.appendChild(div);
      }});
    }}
    function renderStats() {{
      const done = Object.values(annotations).filter(x => x.label && x.label !== 'skip').length;
      const skipped = Object.values(annotations).filter(x => x.label === 'skip').length;
      document.getElementById("stats").innerHTML = `<b>${{done}}</b> labeled, <b>${{skipped}}</b> skipped, total <b>${{items.length}}</b>`;
    }}
    function exportCsv() {{
      const rows = [["rel_path","label","comment","annotator"]];
      items.forEach(item => {{
        const a = ann(item);
        if (a.label && a.label !== "skip") rows.push([item.rel_path, a.label, a.comment || "", "manual_ui"]);
      }});
      const csv = rows.map(row => row.map(v => '"' + String(v).replaceAll('"', '""') + '"').join(",")).join("\\n");
      const blob = new Blob([csv], {{ type: "text/csv;charset=utf-8" }});
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = "manual_labels.csv";
      link.click();
    }}
    document.addEventListener("keydown", (e) => {{
      if (e.key === "ArrowLeft") prevItem();
      if (e.key === "ArrowRight") nextItem();
      if (e.key === "1") setLabel("ordinary");
      if (e.key === "2") setLabel("thin");
      if (e.key === "3") setLabel("talc");
      if (e.key === "0") setLabel("skip");
    }});
    render();
  </script>
</body>
</html>
""",
        encoding="utf-8",
    )
    print(f"wrote {output}: items={len(items)}")


if __name__ == "__main__":
    main()
