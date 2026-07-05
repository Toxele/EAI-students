from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from urllib.parse import quote


def file_url(path: str | Path) -> str:
    resolved = Path(path).resolve()
    return "file:///" + quote(str(resolved).replace("\\", "/"))


def web_url(path: str | Path, root: str | Path) -> str:
    resolved = Path(path).resolve()
    root_path = Path(root).resolve()
    try:
        rel = resolved.relative_to(root_path)
    except ValueError:
        return file_url(resolved)
    return "/" + quote(str(rel).replace("\\", "/"))


def safe_stem(path: str) -> str:
    stem = Path(path).stem.replace(" ", "_").replace("/", "__").replace("\\", "__")
    digest = hashlib.md5(path.encode("utf-8")).hexdigest()[:8]
    return f"{stem}_{digest}"


def read_items(report_csv: str | Path, prefer_overlay: bool, root: str | Path) -> list[dict[str, str]]:
    with Path(report_csv).open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    items = []
    for row in rows:
        image_path = row.get("overlay_path") if prefer_overlay else row.get("image_path")
        if not image_path or not Path(image_path).exists():
            image_path = row.get("image_path") or row.get("overlay_path")
        if not image_path or not Path(image_path).exists():
            continue

        source_path = row.get("image_path", image_path)
        items.append(
            {
                "image_path": image_path,
                "file_url": file_url(image_path),
                "web_url": web_url(image_path, root),
                "source_image_path": source_path,
                "source_file_url": file_url(source_path),
                "source_web_url": web_url(source_path, root),
                "rel_path": row.get("rel_path", source_path),
                "suggested_name": f"{safe_stem(source_path)}__manual_mask.png",
                "foreground_fraction": row.get("foreground_fraction", ""),
                "contours": row.get("contours", ""),
            }
        )
    return items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="artifacts/weak_masks/talc/weak_mask_report.csv")
    parser.add_argument("--output", default="artifacts/mask_editor/index.html")
    parser.add_argument("--root", default=".")
    parser.add_argument("--prefer-overlay", action="store_true")
    args = parser.parse_args()

    items = read_items(args.report, prefer_overlay=args.prefer_overlay, root=args.root)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    data_json = json.dumps(items, ensure_ascii=False)

    output.write_text(
        f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Manual Talc Mask Editor</title>
  <style>
    :root {{ color-scheme: dark; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Arial, sans-serif; background: #11161c; color: #e9eef5; }}
    .app {{ display: grid; grid-template-columns: 320px 1fr; min-height: 100vh; }}
    aside {{ background: #171f28; border-right: 1px solid #2b3642; padding: 14px; overflow: auto; }}
    main {{ display: grid; grid-template-rows: auto 1fr auto; min-width: 0; }}
    .toolbar {{ display: flex; gap: 8px; flex-wrap: wrap; align-items: center; padding: 10px; border-bottom: 1px solid #2b3642; }}
    button, input[type=range] {{ accent-color: #ff2a2a; }}
    button {{ background: #253242; color: #e9eef5; border: 1px solid #3b4b5d; border-radius: 6px; padding: 8px 10px; cursor: pointer; }}
    button:hover {{ background: #314257; }}
    button.active {{ outline: 2px solid #ff4d4d; }}
    .stage-wrap {{ display: flex; justify-content: center; align-items: center; overflow: auto; padding: 12px; }}
    .stage {{ position: relative; display: inline-block; background: #05070a; }}
    canvas {{ display: block; max-width: calc(100vw - 360px); max-height: calc(100vh - 130px); }}
    #fillCanvas, #drawCanvas {{ position: absolute; left: 0; top: 0; }}
    #drawCanvas {{ cursor: crosshair; touch-action: none; }}
    .meta {{ padding: 10px; border-top: 1px solid #2b3642; font-size: 13px; word-break: break-word; }}
    .list {{ display: grid; gap: 6px; }}
    .item {{ padding: 8px; border: 1px solid #2b3642; border-radius: 6px; cursor: pointer; }}
    .item.current {{ border-color: #ff4d4d; }}
    .muted {{ color: #9aabbc; }}
    .ok {{ color: #8fd694; }}
    .warn {{ color: #ffcb6b; }}
    .hint {{ font-size: 13px; color: #9aabbc; line-height: 1.35; }}
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <h3>Manual Mask Editor</h3>
      <p class="hint">
        Brush closes contours, Fill creates area masks. Blue overlay lines and
        red manual lines are treated as fill boundaries.
      </p>
      <p class="hint">
        Recommended folder: artifacts/manual_masks/talc_masks.
      </p>
      <div id="stats" class="muted"></div>
      <hr>
      <div id="list" class="list"></div>
    </aside>
    <main>
      <div class="toolbar">
        <button onclick="prevItem()">Prev</button>
        <button onclick="nextItem()">Next</button>
        <button id="brushBtn" class="active" onclick="setTool('brush')">Brush</button>
        <button id="fillBtn" onclick="setTool('fill')">Fill</button>
        <button id="eraserBtn" onclick="setTool('eraser')">Eraser</button>
        <label>Size <input id="size" type="range" min="2" max="80" value="18"></label>
        <button onclick="undo()">Undo</button>
        <button onclick="clearMask()">Clear</button>
        <button onclick="chooseAutosaveDir()">Choose autosave folder</button>
        <button onclick="saveCurrentMask()">Save now</button>
        <button onclick="downloadMask()">Download mask</button>
        <button onclick="downloadOverlay()">Download overlay</button>
      </div>
      <div class="stage-wrap">
        <div class="stage">
          <canvas id="imageCanvas"></canvas>
          <canvas id="boundaryCanvas" style="display:none"></canvas>
          <canvas id="fillCanvas"></canvas>
          <canvas id="drawCanvas"></canvas>
        </div>
      </div>
      <div class="meta" id="meta"></div>
    </main>
  </div>
  <script>
    const items = {data_json};
    let index = 0;
    let tool = "brush";
    let drawing = false;
    let last = null;
    let strokes = [];
    let autosaveDir = null;
    let dirty = false;

    const imageCanvas = document.getElementById("imageCanvas");
    const boundaryCanvas = document.getElementById("boundaryCanvas");
    const fillCanvas = document.getElementById("fillCanvas");
    const drawCanvas = document.getElementById("drawCanvas");
    const imageCtx = imageCanvas.getContext("2d");
    const boundaryCtx = boundaryCanvas.getContext("2d");
    const fillCtx = fillCanvas.getContext("2d");
    const drawCtx = drawCanvas.getContext("2d");

    function current() {{ return items[index]; }}
    function imageUrl(item) {{
      return window.location.protocol === "file:" ? item.file_url : item.web_url;
    }}
    function boundaryUrl(item) {{
      return window.location.protocol === "file:" ? item.source_file_url : item.source_web_url;
    }}

    function setTool(name) {{
      tool = name;
      document.getElementById("brushBtn").classList.toggle("active", tool === "brush");
      document.getElementById("fillBtn").classList.toggle("active", tool === "fill");
      document.getElementById("eraserBtn").classList.toggle("active", tool === "eraser");
    }}

    async function prevItem() {{ await goToItem(Math.max(0, index - 1)); }}
    async function nextItem() {{ await goToItem(Math.min(items.length - 1, index + 1)); }}

    async function goToItem(newIndex) {{
      if (newIndex === index) return;
      const saved = await saveCurrentMask();
      if (autosaveDir && !saved) return;
      index = newIndex;
      await loadItem();
    }}

    function canvasPoint(evt) {{
      const rect = drawCanvas.getBoundingClientRect();
      return {{
        x: (evt.clientX - rect.left) * (drawCanvas.width / rect.width),
        y: (evt.clientY - rect.top) * (drawCanvas.height / rect.height)
      }};
    }}

    function pushSnapshot() {{
      strokes.push({{
        draw: drawCtx.getImageData(0, 0, drawCanvas.width, drawCanvas.height),
        fill: fillCtx.getImageData(0, 0, fillCanvas.width, fillCanvas.height)
      }});
      if (strokes.length > 50) strokes.shift();
    }}

    function markDirty() {{
      dirty = true;
      renderMeta();
    }}

    function startDraw(evt) {{
      const pt = canvasPoint(evt);
      if (tool === "fill") {{
        floodFill(pt);
        return;
      }}
      drawing = true;
      pushSnapshot();
      last = pt;
      drawDot(last);
      markDirty();
    }}

    function moveDraw(evt) {{
      if (!drawing) return;
      const pt = canvasPoint(evt);
      drawCtx.save();
      drawCtx.lineCap = "round";
      drawCtx.lineJoin = "round";
      drawCtx.lineWidth = Number(document.getElementById("size").value);
      if (tool === "eraser") {{
        drawCtx.globalCompositeOperation = "destination-out";
        drawCtx.strokeStyle = "rgba(0,0,0,1)";
      }} else {{
        drawCtx.globalCompositeOperation = "source-over";
        drawCtx.strokeStyle = "rgba(255, 32, 32, 0.95)";
      }}
      drawCtx.beginPath();
      drawCtx.moveTo(last.x, last.y);
      drawCtx.lineTo(pt.x, pt.y);
      drawCtx.stroke();
      if (tool === "eraser") {{
        fillCtx.save();
        fillCtx.lineCap = "round";
        fillCtx.lineJoin = "round";
        fillCtx.lineWidth = Number(document.getElementById("size").value);
        fillCtx.globalCompositeOperation = "destination-out";
        fillCtx.beginPath();
        fillCtx.moveTo(last.x, last.y);
        fillCtx.lineTo(pt.x, pt.y);
        fillCtx.stroke();
        fillCtx.restore();
      }}
      drawCtx.restore();
      last = pt;
      dirty = true;
    }}

    function endDraw() {{
      if (drawing) renderMeta();
      drawing = false;
      last = null;
    }}

    function drawDot(pt) {{
      drawCtx.save();
      drawCtx.beginPath();
      drawCtx.arc(pt.x, pt.y, Number(document.getElementById("size").value) / 2, 0, Math.PI * 2);
      if (tool === "eraser") {{
        drawCtx.globalCompositeOperation = "destination-out";
        drawCtx.fillStyle = "rgba(0,0,0,1)";
      }} else {{
        drawCtx.globalCompositeOperation = "source-over";
        drawCtx.fillStyle = "rgba(255, 32, 32, 0.95)";
      }}
      drawCtx.fill();
      if (tool === "eraser") {{
        fillCtx.save();
        fillCtx.beginPath();
        fillCtx.arc(pt.x, pt.y, Number(document.getElementById("size").value) / 2, 0, Math.PI * 2);
        fillCtx.globalCompositeOperation = "destination-out";
        fillCtx.fillStyle = "rgba(0,0,0,1)";
        fillCtx.fill();
        fillCtx.restore();
      }}
      drawCtx.restore();
    }}

    function undo() {{
      const img = strokes.pop();
      if (img) {{
        drawCtx.putImageData(img.draw, 0, 0);
        fillCtx.putImageData(img.fill, 0, 0);
        markDirty();
      }}
    }}

    function clearMask() {{
      pushSnapshot();
      drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
      fillCtx.clearRect(0, 0, fillCanvas.width, fillCanvas.height);
      markDirty();
    }}

    async function loadItem() {{
      const item = current();
      const img = new Image();
      const boundaryImg = new Image();
      await new Promise((resolve, reject) => {{
        img.onload = resolve;
        img.onerror = reject;
        img.src = imageUrl(item);
      }});
      await new Promise((resolve, reject) => {{
        boundaryImg.onload = resolve;
        boundaryImg.onerror = reject;
        boundaryImg.src = boundaryUrl(item);
      }});
      imageCanvas.width = img.naturalWidth;
      imageCanvas.height = img.naturalHeight;
      boundaryCanvas.width = img.naturalWidth;
      boundaryCanvas.height = img.naturalHeight;
      fillCanvas.width = img.naturalWidth;
      fillCanvas.height = img.naturalHeight;
      drawCanvas.width = img.naturalWidth;
      drawCanvas.height = img.naturalHeight;
      imageCtx.drawImage(img, 0, 0);
      boundaryCtx.drawImage(boundaryImg, 0, 0, boundaryCanvas.width, boundaryCanvas.height);
      fillCtx.clearRect(0, 0, fillCanvas.width, fillCanvas.height);
      drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
      strokes = [];
      dirty = false;
      await loadSavedMask();
      renderList();
      renderMeta();
    }}

    function renderMeta() {{
      const item = current();
      const saveState = autosaveDir ? "enabled" : "choose folder first";
      const saveClass = autosaveDir ? "ok" : "warn";
      document.getElementById("stats").className = "muted";
      document.getElementById("stats").textContent = `${{index + 1}} / ${{items.length}}`;
      document.getElementById("meta").innerHTML =
        `<b>${{index + 1}} / ${{items.length}}</b><br>` +
        `<span class="muted">${{item.source_image_path}}</span><br>` +
        `mask file: <b>${{item.suggested_name}}</b><br>` +
        `autosave: <b class="${{saveClass}}">${{saveState}}</b>, unsaved changes: ${{dirty ? "yes" : "no"}}<br>` +
        `auto fraction: ${{item.foreground_fraction || "-"}}, contours: ${{item.contours || "-"}}`;
    }}

    function setStatus(text, cls = "muted") {{
      const stats = document.getElementById("stats");
      stats.className = cls;
      stats.textContent = `${{index + 1}} / ${{items.length}} - ${{text}}`;
    }}

    function renderList() {{
      const list = document.getElementById("list");
      list.innerHTML = "";
      items.forEach((item, i) => {{
        const div = document.createElement("div");
        div.className = `item ${{i === index ? "current" : ""}}`;
        div.onclick = () => {{ goToItem(i); }};
        div.innerHTML = `<b>${{i + 1}}.</b> ${{item.suggested_name}}`;
        list.appendChild(div);
      }});
    }}

    function buildBinaryMaskCanvas() {{
      const out = document.createElement("canvas");
      out.width = drawCanvas.width;
      out.height = drawCanvas.height;
      const outCtx = out.getContext("2d");
      const data = drawCtx.getImageData(0, 0, drawCanvas.width, drawCanvas.height);
      const fill = fillCtx.getImageData(0, 0, fillCanvas.width, fillCanvas.height);
      const outData = outCtx.createImageData(drawCanvas.width, drawCanvas.height);
      for (let i = 0; i < data.data.length; i += 4) {{
        const alpha = Math.max(data.data[i + 3], fill.data[i + 3]);
        const v = alpha > 0 ? 255 : 0;
        outData.data[i] = v;
        outData.data[i + 1] = v;
        outData.data[i + 2] = v;
        outData.data[i + 3] = 255;
      }}
      outCtx.putImageData(outData, 0, 0);
      return out;
    }}

    function isBlueBoundary(r, g, b, a) {{
      return a > 0 && b > 130 && b > r * 1.35 && b > g * 1.2 && (b - Math.max(r, g)) > 35;
    }}

    function findFillStart(boundary, x, y, w, h) {{
      const start = y * w + x;
      if (!boundary[start]) return start;
      for (let radius = 1; radius <= 12; radius++) {{
        for (let dy = -radius; dy <= radius; dy++) {{
          for (let dx = -radius; dx <= radius; dx++) {{
            if (Math.abs(dx) !== radius && Math.abs(dy) !== radius) continue;
            const nx = x + dx;
            const ny = y + dy;
            if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
            const p = ny * w + nx;
            if (!boundary[p]) return p;
          }}
        }}
      }}
      return -1;
    }}

    function dilateBoundary(boundary, w, h) {{
      const out = new Uint8Array(boundary);
      for (let y = 0; y < h; y++) {{
        for (let x = 0; x < w; x++) {{
          const p = y * w + x;
          if (!boundary[p]) continue;
          for (let dy = -1; dy <= 1; dy++) {{
            for (let dx = -1; dx <= 1; dx++) {{
              const nx = x + dx;
              const ny = y + dy;
              if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
              out[ny * w + nx] = 1;
            }}
          }}
        }}
      }}
      return out;
    }}

    function floodFill(pt) {{
      const w = imageCanvas.width;
      const h = imageCanvas.height;
      const startX = Math.max(0, Math.min(w - 1, Math.round(pt.x)));
      const startY = Math.max(0, Math.min(h - 1, Math.round(pt.y)));
      const n = w * h;
      const boundaryImage = boundaryCtx.getImageData(0, 0, w, h);
      const lines = drawCtx.getImageData(0, 0, w, h);
      const boundary = new Uint8Array(n);

      for (let p = 0, i = 0; p < n; p++, i += 4) {{
        boundary[p] = isBlueBoundary(boundaryImage.data[i], boundaryImage.data[i + 1], boundaryImage.data[i + 2], boundaryImage.data[i + 3]) || lines.data[i + 3] > 0 ? 1 : 0;
      }}

      const walls = dilateBoundary(boundary, w, h);
      const start = findFillStart(walls, startX, startY, w, h);
      if (start < 0) {{
        setStatus("fill clicked on a dense boundary", "warn");
        return;
      }}

      pushSnapshot();
      const visited = new Uint8Array(n);
      const stack = new Int32Array(n);
      let top = 0;
      let count = 0;
      stack[top++] = start;
      visited[start] = 1;

      while (top > 0) {{
        const p = stack[--top];
        count++;
        const x = p % w;
        const y = Math.floor(p / w);
        if (x > 0) {{
          const q = p - 1;
          if (!visited[q] && !walls[q]) {{ visited[q] = 1; stack[top++] = q; }}
        }}
        if (x < w - 1) {{
          const q = p + 1;
          if (!visited[q] && !walls[q]) {{ visited[q] = 1; stack[top++] = q; }}
        }}
        if (y > 0) {{
          const q = p - w;
          if (!visited[q] && !walls[q]) {{ visited[q] = 1; stack[top++] = q; }}
        }}
        if (y < h - 1) {{
          const q = p + w;
          if (!visited[q] && !walls[q]) {{ visited[q] = 1; stack[top++] = q; }}
        }}
      }}

      const fill = fillCtx.getImageData(0, 0, w, h);
      for (let p = 0, i = 0; p < n; p++, i += 4) {{
        if (visited[p]) {{
          fill.data[i] = 255;
          fill.data[i + 1] = 32;
          fill.data[i + 2] = 32;
          fill.data[i + 3] = 90;
        }}
      }}
      fillCtx.putImageData(fill, 0, 0);
      markDirty();
      setStatus(`filled ${{count}} px`, "ok");
    }}

    function canvasToBlob(canvas) {{
      return new Promise((resolve, reject) => {{
        canvas.toBlob(blob => {{
          if (blob) resolve(blob);
          else reject(new Error("Canvas export failed. Open the editor through localhost, not as a file."));
        }}, "image/png");
      }});
    }}

    async function chooseAutosaveDir() {{
      try {{
        if (!window.showDirectoryPicker) {{
          throw new Error("Folder autosave requires Chrome/Edge and http://localhost. Use Download mask as a fallback.");
        }}
        autosaveDir = await window.showDirectoryPicker({{ mode: "readwrite" }});
        await loadSavedMask();
        renderMeta();
        setStatus("autosave folder selected", "ok");
      }} catch (err) {{
        setStatus(`autosave setup failed: ${{err.message}}`, "warn");
        alert(err.message);
      }}
    }}

    async function saveCurrentMask() {{
      if (!autosaveDir) return false;
      try {{
        const handle = await autosaveDir.getFileHandle(current().suggested_name, {{ create: true }});
        const writable = await handle.createWritable();
        await writable.write(await canvasToBlob(buildBinaryMaskCanvas()));
        await writable.close();
        dirty = false;
        renderMeta();
        setStatus(`saved ${{current().suggested_name}}`, "ok");
        return true;
      }} catch (err) {{
        setStatus(`save failed: ${{err.message}}`, "warn");
        alert(`Mask was not saved: ${{err.message}}`);
        return false;
      }}
    }}

    async function loadSavedMask() {{
      if (!autosaveDir) return;
      try {{
        const handle = await autosaveDir.getFileHandle(current().suggested_name, {{ create: false }});
        const file = await handle.getFile();
        const img = new Image();
        img.src = URL.createObjectURL(file);
        await new Promise((resolve, reject) => {{
          img.onload = resolve;
          img.onerror = reject;
        }});

        const tmp = document.createElement("canvas");
        tmp.width = drawCanvas.width;
        tmp.height = drawCanvas.height;
        const tmpCtx = tmp.getContext("2d");
        tmpCtx.drawImage(img, 0, 0, drawCanvas.width, drawCanvas.height);
        URL.revokeObjectURL(img.src);

        const data = tmpCtx.getImageData(0, 0, tmp.width, tmp.height);
        const red = fillCtx.createImageData(fillCanvas.width, fillCanvas.height);
        for (let i = 0; i < data.data.length; i += 4) {{
          const v = Math.max(data.data[i], data.data[i + 1], data.data[i + 2]);
          if (v > 0) {{
            red.data[i] = 255;
            red.data[i + 1] = 32;
            red.data[i + 2] = 32;
            red.data[i + 3] = 90;
          }}
        }}
        fillCtx.putImageData(red, 0, 0);
        dirty = false;
        setStatus("loaded saved mask", "ok");
      }} catch (err) {{
        dirty = false;
      }}
    }}

    function downloadCanvas(canvas, filename) {{
      canvas.toBlob(blob => {{
        const link = document.createElement("a");
        link.href = URL.createObjectURL(blob);
        link.download = filename;
        link.click();
        URL.revokeObjectURL(link.href);
      }}, "image/png");
    }}

    function downloadMask() {{
      downloadCanvas(buildBinaryMaskCanvas(), current().suggested_name);
    }}

    function downloadOverlay() {{
      const out = document.createElement("canvas");
      out.width = imageCanvas.width;
      out.height = imageCanvas.height;
      const outCtx = out.getContext("2d");
      outCtx.drawImage(imageCanvas, 0, 0);
      outCtx.drawImage(fillCanvas, 0, 0);
      outCtx.drawImage(drawCanvas, 0, 0);
      downloadCanvas(out, current().suggested_name.replace("__manual_mask", "__manual_overlay"));
    }}

    drawCanvas.addEventListener("pointerdown", startDraw);
    drawCanvas.addEventListener("pointermove", moveDraw);
    drawCanvas.addEventListener("pointerup", endDraw);
    drawCanvas.addEventListener("pointerleave", endDraw);

    document.addEventListener("keydown", e => {{
      if (e.ctrlKey && e.key.toLowerCase() === "z") undo();
      if (e.key === "b") setTool("brush");
      if (e.key === "f") setTool("fill");
      if (e.key === "e") setTool("eraser");
      if (e.key === "ArrowLeft") {{ e.preventDefault(); prevItem(); }}
      if (e.key === "ArrowRight") {{ e.preventDefault(); nextItem(); }}
    }});

    window.addEventListener("beforeunload", e => {{
      if (dirty) {{
        e.preventDefault();
        e.returnValue = "";
      }}
    }});

    loadItem();
  </script>
</body>
</html>
""",
        encoding="utf-8",
    )
    print(f"wrote {output}: items={len(items)}")


if __name__ == "__main__":
    main()
