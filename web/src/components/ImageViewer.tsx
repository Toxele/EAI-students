import { useEffect, useRef, useCallback, useState, forwardRef, useImperativeHandle } from "react";
import OpenSeadragon from "openseadragon";
import { Badge, Paper, ActionIcon, Tooltip, Group, Text, Loader } from "@mantine/core";
import { IconPlus, IconMinus, IconZoomScan } from "@tabler/icons-react";
import type { AnalysisResult, Grain, LayerMode, TalcTool, TalcViewMode } from "../types";
import { absUrl, applyTalcMask, grainColor } from "../api";
import { CONFIDENCE_DISPLAY_ALPHA, confidenceByteToT, confidenceColor } from "../confidenceColormap";

const MAX_SVG_GRAINS = 800;
// Тальк редактируется на уменьшенном растре (иначе canvas 10000x10000 px
// слишком тяжёлый для перерисовки на каждый pointermove); маска
// апскейлится обратно на бэкенде при сохранении (nearest-neighbor).
const TALC_EDIT_MAX_SIDE = 2048;
const TALC_COLOR_RGB = "40, 120, 255";
const TALC_HISTORY_LIMIT = 20;

type Corner = "tl" | "tr" | "bl" | "br";
type EdgeMid = "t" | "b" | "l" | "r";
type DragKind = Corner | EdgeMid | "move";

const HANDLE_CURSOR: Record<Corner | EdgeMid, string> = {
  tl: "nwse-resize",
  br: "nwse-resize",
  tr: "nesw-resize",
  bl: "nesw-resize",
  t: "ns-resize",
  b: "ns-resize",
  l: "ew-resize",
  r: "ew-resize",
};

const clamp = (v: number, lo: number, hi: number) => Math.min(Math.max(v, lo), hi);

interface DragState {
  grainId: number;
  kind: DragKind;
  startBboxImg: [number, number, number, number];
  startPointerImg: { x: number; y: number };
}

interface Props {
  imageUrl: string;
  talcMaskUrl: string | null;
  talcConfidenceUrl: string | null;
  typeLayerUrl: string | null;
  grains: Grain[];
  layer: LayerMode;
  imageWidth: number;
  imageHeight: number;
  selectedId: number | null;
  onSelectGrain: (id: number | null) => void;
  onGrainBboxChange?: (id: number, bbox: [number, number, number, number]) => void;
  onGrainDragStart?: () => void;
  resultId: string;
  talcTool: TalcTool | null;
  talcBrushSize: number;
  talcViewMode: TalcViewMode;
  onTalcDirtyChange?: (dirty: boolean) => void;
  onTalcHistoryChange?: (canUndo: boolean, canRedo: boolean) => void;
  onTalcSaved?: (updated: AnalysisResult) => void;
  onTalcSaveError?: (message: string) => void;
}

export interface ImageViewerHandle {
  saveTalcMask: () => Promise<void>;
  undoTalc: () => void;
  redoTalc: () => void;
}

const ImageViewer = forwardRef<ImageViewerHandle, Props>(function ImageViewer(
  {
    imageUrl,
    talcMaskUrl,
    talcConfidenceUrl,
    typeLayerUrl,
    grains,
    layer,
    imageWidth,
    imageHeight,
    selectedId,
    onSelectGrain,
    onGrainBboxChange,
    onGrainDragStart,
    resultId,
    talcTool,
    talcBrushSize,
    talcViewMode,
    onTalcDirtyChange,
    onTalcHistoryChange,
    onTalcSaved,
    onTalcSaveError,
  },
  ref
) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const osdRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const viewerRef = useRef<OpenSeadragon.Viewer | null>(null);
  const typeItemRef = useRef<OpenSeadragon.TiledImage | null>(null);
  const [drawStats, setDrawStats] = useState({ drawn: 0, total: 0 });

  const layerRef = useRef(layer);
  const grainsRef = useRef(grains);
  const selectedIdRef = useRef(selectedId);
  const onSelectRef = useRef(onSelectGrain);
  const onBboxChangeRef = useRef(onGrainBboxChange);
  const onGrainDragStartRef = useRef(onGrainDragStart);
  const dragStateRef = useRef<DragState | null>(null);
  const pendingBboxRef = useRef<{ id: number; bbox: [number, number, number, number] } | null>(null);
  const rafRef = useRef<number | null>(null);
  layerRef.current = layer;
  grainsRef.current = grains;
  selectedIdRef.current = selectedId;
  onSelectRef.current = onSelectGrain;
  onBboxChangeRef.current = onGrainBboxChange;
  onGrainDragStartRef.current = onGrainDragStart;

  // --- Редактирование маски талька (карандаш/ластик/заливка) ---
  const talcScreenCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const talcOffscreenRef = useRef<HTMLCanvasElement | null>(null);
  // Карта уверенности модели — не редактируется тулзами напрямую (кроме
  // stampTalc/floodFillTalc, которые проставляют туда максимальную
  // уверенность синхронно с правкой маски), используется только для
  // отображения в режиме "Уверенность".
  const talcConfidenceOffscreenRef = useRef<HTMLCanvasElement | null>(null);
  const talcReadyRef = useRef(false);
  const talcStrokeActiveRef = useRef(false);
  const talcLastPointRef = useRef<{ x: number; y: number } | null>(null);
  const talcToolRef = useRef(talcTool);
  const talcBrushSizeRef = useRef(talcBrushSize);
  const talcViewModeRef = useRef(talcViewMode);
  const onTalcSavedRef = useRef(onTalcSaved);
  const onTalcSaveErrorRef = useRef(onTalcSaveError);
  const onTalcDirtyChangeRef = useRef(onTalcDirtyChange);
  const onTalcHistoryChangeRef = useRef(onTalcHistoryChange);
  const talcUndoStackRef = useRef<ImageData[]>([]);
  const talcRedoStackRef = useRef<ImageData[]>([]);
  const [talcLoading, setTalcLoading] = useState(false);
  const [talcDirty, setTalcDirty] = useState(false);
  talcToolRef.current = talcTool;
  talcBrushSizeRef.current = talcBrushSize;
  talcViewModeRef.current = talcViewMode;
  onTalcSavedRef.current = onTalcSaved;
  onTalcSaveErrorRef.current = onTalcSaveError;
  onTalcDirtyChangeRef.current = onTalcDirtyChange;
  onTalcHistoryChangeRef.current = onTalcHistoryChange;

  useEffect(() => {
    onTalcDirtyChangeRef.current?.(talcDirty);
  }, [talcDirty]);

  const notifyTalcHistory = useCallback(() => {
    onTalcHistoryChangeRef.current?.(talcUndoStackRef.current.length > 0, talcRedoStackRef.current.length > 0);
  }, []);

  // Снимок маски перед мутирующим действием (начало мазка/заливка) —
  // основа для undo. Новое действие обнуляет redo-историю.
  const pushTalcHistory = useCallback(() => {
    const off = talcOffscreenRef.current;
    const ctx = off?.getContext("2d");
    if (!ctx || !off) return;
    const snapshot = ctx.getImageData(0, 0, off.width, off.height);
    const stack = talcUndoStackRef.current;
    stack.push(snapshot);
    if (stack.length > TALC_HISTORY_LIMIT) stack.shift();
    talcRedoStackRef.current = [];
    notifyTalcHistory();
  }, [notifyTalcHistory]);

  const clientToImagePoint = useCallback((clientX: number, clientY: number) => {
    const viewer = viewerRef.current;
    const osdEl = osdRef.current;
    if (!viewer || !osdEl) return null;
    const bounds = osdEl.getBoundingClientRect();
    const localX = clientX - bounds.left;
    const localY = clientY - bounds.top;
    return viewer.viewport.viewerElementToImageCoordinates(new OpenSeadragon.Point(localX, localY));
  }, []);

  const syncOpacities = useCallback(() => {
    const viewer = viewerRef.current;
    if (!viewer || viewer.world.getItemCount() === 0) return;
    viewer.world.getItemAt(0).setOpacity(1);
    typeItemRef.current?.setOpacity(0);
  }, []);

  const drawTalcCanvas = useCallback(() => {
    const canvas = talcScreenCanvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap) return;

    const cw = wrap.clientWidth;
    const ch = wrap.clientHeight;
    if (canvas.width !== cw) canvas.width = cw;
    if (canvas.height !== ch) canvas.height = ch;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const viewer = viewerRef.current;
    const off = talcOffscreenRef.current;
    if (layerRef.current !== "talc" || !viewer || !off || !talcReadyRef.current) return;

    const viewport = viewer.viewport;
    const p1 = viewport.imageToViewerElementCoordinates(new OpenSeadragon.Point(0, 0));
    const p2 = viewport.imageToViewerElementCoordinates(new OpenSeadragon.Point(imageWidth, imageHeight));
    const dw = p2.x - p1.x;
    const dh = p2.y - p1.y;

    const confOff = talcConfidenceOffscreenRef.current;
    if (talcViewModeRef.current === "confidence" && confOff) {
      // Силуэт маски (форма — что вообще тальк).
      ctx.drawImage(off, p1.x, p1.y, dw, dh);
      // confOff уже хранит цвет по шкале уверенности и постоянную альфу
      // (см. confidenceColormap.ts) — source-in оставляет только пересечение
      // с силуэтом маски, итоговая альфа = альфа маски × альфа confOff, цвет
      // берётся из confOff. Показывается только внутри маски, цвет
      // градуируется уверенностью.
      ctx.globalCompositeOperation = "source-in";
      ctx.drawImage(confOff, p1.x, p1.y, dw, dh);
      ctx.globalCompositeOperation = "source-over";
      return;
    }

    // off хранит маску как непрозрачно-белую фигуру на прозрачном фоне —
    // перекрашиваем в полупрозрачный синий через source-atop (красит только
    // там, где уже есть alpha, форму не меняет).
    ctx.drawImage(off, p1.x, p1.y, dw, dh);
    ctx.globalCompositeOperation = "source-atop";
    ctx.fillStyle = `rgba(${TALC_COLOR_RGB}, 0.55)`;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.globalCompositeOperation = "source-over";
  }, [imageWidth, imageHeight]);

  const drawSvgBboxes = useCallback(() => {
    const viewer = viewerRef.current;
    const svg = svgRef.current;
    if (!svg) return;

    svg.innerHTML = "";

    if (layerRef.current !== "type" || !viewer) {
      setDrawStats({ drawn: 0, total: 0 });
      return;
    }

    const viewport = viewer.viewport;
    const bounds = viewport.getBounds(true);
    const tl = viewport.viewportToImageCoordinates(bounds.getTopLeft());
    const br = viewport.viewportToImageCoordinates(bounds.getBottomRight());
    const x0 = Math.min(tl.x, br.x);
    const y0 = Math.min(tl.y, br.y);
    const x1 = Math.max(tl.x, br.x);
    const y1 = Math.max(tl.y, br.y);

    const candidates: { g: Grain; area: number }[] = [];

    for (const g of grainsRef.current) {
      if (g.status === "false_positive") continue;
      const [bx, by, bw, bh] = g.bbox;
      if (bx + bw < x0 || bx > x1 || by + bh < y0 || by > y1) continue;

      const p1 = viewport.imageToViewerElementCoordinates(new OpenSeadragon.Point(bx, by));
      const p2 = viewport.imageToViewerElementCoordinates(
        new OpenSeadragon.Point(bx + bw, by + bh)
      );
      const rw = Math.abs(p2.x - p1.x);
      const rh = Math.abs(p2.y - p1.y);
      candidates.push({ g, area: rw * rh });
    }

    candidates.sort((a, b) => a.area - b.area);
    const toDraw = candidates.slice(0, MAX_SVG_GRAINS);
    setDrawStats({ drawn: toDraw.length, total: candidates.length });

    for (const { g } of toDraw) {
      const [bx, by, bw, bh] = g.bbox;
      const p1 = viewport.imageToViewerElementCoordinates(new OpenSeadragon.Point(bx, by));
      const p2 = viewport.imageToViewerElementCoordinates(
        new OpenSeadragon.Point(bx + bw, by + bh)
      );
      const x = Math.min(p1.x, p2.x);
      const y = Math.min(p1.y, p2.y);
      const w = Math.max(Math.abs(p2.x - p1.x), 1);
      const h = Math.max(Math.abs(p2.y - p1.y), 1);

      const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      rect.setAttribute("x", String(x));
      rect.setAttribute("y", String(y));
      rect.setAttribute("width", String(w));
      rect.setAttribute("height", String(h));
      rect.setAttribute("stroke", grainColor(g).replace("0.7", "1"));
      rect.setAttribute("fill", grainColor(g));
      const isSelected = g.id === selectedIdRef.current;
      if (isSelected) rect.classList.add("selected");
      rect.addEventListener("click", (e) => {
        e.stopPropagation();
        onSelectRef.current(g.id);
      });

      if (isSelected && onBboxChangeRef.current) {
        rect.classList.add("grain-rect-movable");
        rect.addEventListener("pointerdown", (e) => {
          e.stopPropagation();
          e.preventDefault();
          const startPt = clientToImagePoint(e.clientX, e.clientY);
          if (!startPt) return;
          onGrainDragStartRef.current?.();
          viewer.setMouseNavEnabled(false);
          dragStateRef.current = {
            grainId: g.id,
            kind: "move",
            startBboxImg: [bx, by, bw, bh],
            startPointerImg: startPt,
          };
          svg.setPointerCapture(e.pointerId);
        });
      }

      svg.appendChild(rect);

      if (isSelected && onBboxChangeRef.current) {
        const addHandle = (key: Corner | EdgeMid, cx: number, cy: number) => {
          const handle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
          handle.setAttribute("cx", String(cx));
          handle.setAttribute("cy", String(cy));
          handle.setAttribute("r", "6");
          handle.setAttribute("class", "grain-handle");
          handle.style.cursor = HANDLE_CURSOR[key];
          handle.addEventListener("pointerdown", (e) => {
            e.stopPropagation();
            e.preventDefault();
            onGrainDragStartRef.current?.();
            viewer.setMouseNavEnabled(false);
            dragStateRef.current = {
              grainId: g.id,
              kind: key,
              startBboxImg: [bx, by, bw, bh],
              startPointerImg: { x: bx, y: by },
            };
            svg.setPointerCapture(e.pointerId);
          });
          svg.appendChild(handle);
        };

        addHandle("tl", x, y);
        addHandle("tr", x + w, y);
        addHandle("bl", x, y + h);
        addHandle("br", x + w, y + h);
        addHandle("t", x + w / 2, y);
        addHandle("b", x + w / 2, y + h);
        addHandle("l", x, y + h / 2);
        addHandle("r", x + w, y + h / 2);
      }
    }
  }, [clientToImagePoint]);

  const handleSvgPointerMove = useCallback((e: React.PointerEvent<SVGSVGElement>) => {
    const drag = dragStateRef.current;
    if (!drag) return;
    const imgPt = clientToImagePoint(e.clientX, e.clientY);
    if (!imgPt) return;

    const [ox, oy, ow, oh] = drag.startBboxImg;
    let nx: number;
    let ny: number;
    let nw: number;
    let nh: number;

    if (drag.kind === "move") {
      const dx = imgPt.x - drag.startPointerImg.x;
      const dy = imgPt.y - drag.startPointerImg.y;
      nw = ow;
      nh = oh;
      nx = clamp(ox + dx, 0, Math.max(0, imageWidth - nw));
      ny = clamp(oy + dy, 0, Math.max(0, imageHeight - nh));
    } else {
      let x0 = ox;
      let y0 = oy;
      let x1 = ox + ow;
      let y1 = oy + oh;
      switch (drag.kind) {
        case "tl":
          x0 = imgPt.x;
          y0 = imgPt.y;
          break;
        case "tr":
          x1 = imgPt.x;
          y0 = imgPt.y;
          break;
        case "bl":
          x0 = imgPt.x;
          y1 = imgPt.y;
          break;
        case "br":
          x1 = imgPt.x;
          y1 = imgPt.y;
          break;
        case "t":
          y0 = imgPt.y;
          break;
        case "b":
          y1 = imgPt.y;
          break;
        case "l":
          x0 = imgPt.x;
          break;
        case "r":
          x1 = imgPt.x;
          break;
      }
      nx = Math.min(x0, x1);
      ny = Math.min(y0, y1);
      nw = Math.max(4, Math.abs(x1 - x0));
      nh = Math.max(4, Math.abs(y1 - y0));
    }

    pendingBboxRef.current = {
      id: drag.grainId,
      bbox: [Math.round(nx), Math.round(ny), Math.round(nw), Math.round(nh)],
    };

    // Coalesce updates to one per animation frame — pointermove can fire
    // far more often than the SVG (up to MAX_SVG_GRAINS boxes) can redraw.
    if (rafRef.current == null) {
      rafRef.current = requestAnimationFrame(() => {
        rafRef.current = null;
        const pending = pendingBboxRef.current;
        if (pending) onBboxChangeRef.current?.(pending.id, pending.bbox);
      });
    }
  }, [clientToImagePoint, imageWidth, imageHeight]);

  const handleSvgPointerUp = useCallback((e: React.PointerEvent<SVGSVGElement>) => {
    const drag = dragStateRef.current;
    if (!drag) return;
    dragStateRef.current = null;
    viewerRef.current?.setMouseNavEnabled(true);
    if (svgRef.current?.hasPointerCapture(e.pointerId)) {
      svgRef.current.releasePointerCapture(e.pointerId);
    }
    if (rafRef.current != null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    const pending = pendingBboxRef.current;
    pendingBboxRef.current = null;
    if (pending && pending.id === drag.grainId) {
      onBboxChangeRef.current?.(pending.id, pending.bbox);
    }
  }, []);

  const refresh = useCallback(() => {
    syncOpacities();
    drawSvgBboxes();
    drawTalcCanvas();
  }, [syncOpacities, drawSvgBboxes, drawTalcCanvas]);

  const addAlignedImage = useCallback(
    (
      viewer: OpenSeadragon.Viewer,
      url: string,
      preload = false
    ): Promise<OpenSeadragon.TiledImage | null> => {
      const base = viewer.world.getItemAt(0);
      if (!base) return Promise.resolve(null);
      const b = base.getBounds();

      return new Promise((resolve) => {
        try {
          // OpenSeadragon never loads tiles for opacity:0 images unless
          // preload is set — without it, a hidden layer can never reach
          // getFullyLoaded() and so can never be revealed.
          //
          // Only width is passed: OpenSeadragon derives height from the
          // loaded image's own aspect ratio and logs an error if both are
          // given (dropping height silently). Талька/type-слои используют
          // тот же aspect ratio, что и base — ширины достаточно для align.
          const ret = viewer.addSimpleImage({
            url: absUrl(url),
            x: b.x,
            y: b.y,
            width: b.width,
            opacity: 0,
            preload,
          }) as unknown;

          if (ret && typeof (ret as Promise<OpenSeadragon.TiledImage>).then === "function") {
            (ret as Promise<OpenSeadragon.TiledImage>).then(resolve).catch(() => resolve(null));
            return;
          }
        } catch {
          /* ignore */
        }
        setTimeout(() => {
          const idx = viewer.world.getItemCount() - 1;
          resolve(idx > 0 ? viewer.world.getItemAt(idx) : null);
        }, 800);
      });
    },
    []
  );

  useEffect(() => {
    if (!osdRef.current) return;

    const viewer = OpenSeadragon({
      element: osdRef.current,
      tileSources: { type: "image", url: absUrl(imageUrl) },
      showNavigationControl: false,
      showZoomControl: false,
      showHomeControl: false,
      showFullPageControl: false,
      minZoomImageRatio: 0.5,
      maxZoomPixelRatio: 3,
      gestureSettingsMouse: { clickToZoom: false },
      animationTime: 0.35,
      crossOriginPolicy: "Anonymous",
    });

    viewerRef.current = viewer;

    const onMove = () => refresh();

    viewer.addHandler("open", async () => {
      if (typeLayerUrl && !typeItemRef.current) {
        typeItemRef.current = await addAlignedImage(viewer, typeLayerUrl);
      }
      refresh();
    });
    viewer.addHandler("animation", onMove);
    viewer.addHandler("animation-finish", onMove);
    viewer.addHandler("resize", onMove);
    viewer.addHandler("viewport-change", onMove);

    return () => {
      viewer.destroy();
      viewerRef.current = null;
      typeItemRef.current = null;
      if (rafRef.current != null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
    };
  }, [imageUrl, typeLayerUrl, addAlignedImage, refresh]);

  // Загрузка/инициализация редактируемых растров талька (маска + карта
  // уверенности) — при смене результата анализа (новый result_id => новые
  // URL). Правки, сделанные на клиенте, не перезагружаются повторно с
  // сервера. Готовность (talcReadyRef) выставляется только после того, как
  // оба растра загружены (или их URL отсутствует).
  useEffect(() => {
    talcReadyRef.current = false;
    talcStrokeActiveRef.current = false;
    talcLastPointRef.current = null;
    setTalcDirty(false);
    talcUndoStackRef.current = [];
    talcRedoStackRef.current = [];
    notifyTalcHistory();

    const longSide = Math.max(imageWidth, imageHeight, 1);
    const scale = Math.min(1, TALC_EDIT_MAX_SIDE / longSide);
    const workingW = Math.max(1, Math.round(imageWidth * scale));
    const workingH = Math.max(1, Math.round(imageHeight * scale));

    const off = document.createElement("canvas");
    off.width = workingW;
    off.height = workingH;
    talcOffscreenRef.current = off;

    const confOff = document.createElement("canvas");
    confOff.width = workingW;
    confOff.height = workingH;
    talcConfidenceOffscreenRef.current = confOff;

    let cancelled = false;
    let maskDone = false;
    let confDone = false;
    const maybeFinish = () => {
      if (cancelled || !maskDone || !confDone) return;
      talcReadyRef.current = true;
      setTalcLoading(false);
      drawTalcCanvas();
    };

    if (!talcMaskUrl) {
      maskDone = true;
    } else {
      setTalcLoading(true);
      const img = new Image();
      img.crossOrigin = "anonymous";
      img.onload = () => {
        if (cancelled) return;
        const tmp = document.createElement("canvas");
        tmp.width = workingW;
        tmp.height = workingH;
        const tctx = tmp.getContext("2d");
        if (tctx) {
          // Без сглаживания: исходная маска бинарна (0/255), а бикубическое/
          // билинейное масштабирование создаёт промежуточные alpha-значения
          // на границах, из-за которых заливка после порога >127 могла
          // оставлять тонкий незакрашенный "шов" на стыке соседних областей.
          tctx.imageSmoothingEnabled = false;
          tctx.drawImage(img, 0, 0, workingW, workingH);
          const imgData = tctx.getImageData(0, 0, workingW, workingH);
          const d = imgData.data;
          for (let i = 0; i < d.length; i += 4) {
            const on = d[i] > 127;
            d[i] = 255;
            d[i + 1] = 255;
            d[i + 2] = 255;
            d[i + 3] = on ? 255 : 0;
          }
          off.getContext("2d")?.putImageData(imgData, 0, 0);
        }
        maskDone = true;
        maybeFinish();
      };
      img.onerror = () => {
        maskDone = true;
        maybeFinish();
      };
      img.src = absUrl(talcMaskUrl);
    }

    if (!talcConfidenceUrl) {
      confDone = true;
    } else {
      const cimg = new Image();
      cimg.crossOrigin = "anonymous";
      cimg.onload = () => {
        if (cancelled) return;
        const tmp = document.createElement("canvas");
        tmp.width = workingW;
        tmp.height = workingH;
        const tctx = tmp.getContext("2d");
        if (tctx) {
          tctx.imageSmoothingEnabled = false;
          tctx.drawImage(cimg, 0, 0, workingW, workingH);
          const imgData = tctx.getImageData(0, 0, workingW, workingH);
          const d = imgData.data;
          // Перекрашиваем байт уверенности (0..255) в цвет по шкале "холодный
          // → горячий" (см. confidenceColormap.ts), альфа — постоянная
          // (совпадает по плотности с обычной маской). Итоговый цвет/альфа
          // видны только внутри силуэта маски — обрезка через source-in в
          // drawTalcCanvas.
          for (let i = 0; i < d.length; i += 4) {
            const t = confidenceByteToT(d[i]);
            const [r, g, b] = confidenceColor(t);
            d[i] = r;
            d[i + 1] = g;
            d[i + 2] = b;
            d[i + 3] = CONFIDENCE_DISPLAY_ALPHA;
          }
          confOff.getContext("2d")?.putImageData(imgData, 0, 0);
        }
        confDone = true;
        maybeFinish();
      };
      cimg.onerror = () => {
        confDone = true;
        maybeFinish();
      };
      cimg.src = absUrl(talcConfidenceUrl);
    }

    maybeFinish();

    return () => {
      cancelled = true;
    };
  }, [talcMaskUrl, talcConfidenceUrl, imageWidth, imageHeight, drawTalcCanvas, notifyTalcHistory]);

  useEffect(() => {
    refresh();
    // Доп. перерисовка после смены слоя — OSD иногда не успевает
    const t = window.setTimeout(refresh, 50);
    return () => window.clearTimeout(t);
  }, [layer, grains, selectedId, refresh]);

  useEffect(() => {
    drawTalcCanvas();
  }, [talcViewMode, drawTalcCanvas]);

  const imageToMaskPoint = useCallback(
    (imgPt: { x: number; y: number }) => {
      const off = talcOffscreenRef.current;
      if (!off || imageWidth <= 0 || imageHeight <= 0) return null;
      return { x: (imgPt.x * off.width) / imageWidth, y: (imgPt.y * off.height) / imageHeight };
    },
    [imageWidth, imageHeight]
  );

  const stampTalc = useCallback((x: number, y: number, tool: TalcTool, radius: number) => {
    const off = talcOffscreenRef.current;
    const ctx = off?.getContext("2d");
    if (!ctx) return;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    if (tool === "eraser") {
      // destination-out стирает alpha независимо от цвета — так ластик
      // корректно убирает ранее закрашенные пиксели.
      ctx.globalCompositeOperation = "destination-out";
      ctx.fillStyle = "rgba(0, 0, 0, 1)";
    } else {
      ctx.globalCompositeOperation = "source-over";
      ctx.fillStyle = "#fff";
    }
    ctx.fill();
    ctx.globalCompositeOperation = "source-over";

    // Правки всегда максимальной уверенности — тот же штамп синхронно
    // ставим на карту уверенности (пенсиль/заливка — цвет "горячего" конца
    // шкалы при постоянной альфе, ластик убирает её вместе с маской).
    const confCtx = talcConfidenceOffscreenRef.current?.getContext("2d");
    if (confCtx) {
      confCtx.beginPath();
      confCtx.arc(x, y, radius, 0, Math.PI * 2);
      if (tool === "eraser") {
        confCtx.globalCompositeOperation = "destination-out";
        confCtx.fillStyle = "rgba(0, 0, 0, 1)";
      } else {
        confCtx.globalCompositeOperation = "source-over";
        const [r, g, b] = confidenceColor(1);
        confCtx.fillStyle = `rgba(${r}, ${g}, ${b}, ${CONFIDENCE_DISPLAY_ALPHA / 255})`;
      }
      confCtx.fill();
      confCtx.globalCompositeOperation = "source-over";
    }
  }, []);

  const strokeTalc = useCallback(
    (from: { x: number; y: number } | null, to: { x: number; y: number }, tool: TalcTool, radius: number) => {
      if (!from) {
        stampTalc(to.x, to.y, tool, radius);
        return;
      }
      const dist = Math.hypot(to.x - from.x, to.y - from.y);
      const step = Math.max(1, radius / 2);
      const steps = Math.max(1, Math.ceil(dist / step));
      for (let s = 1; s <= steps; s++) {
        const t = s / steps;
        stampTalc(from.x + (to.x - from.x) * t, from.y + (to.y - from.y) * t, tool, radius);
      }
    },
    [stampTalc]
  );

  const floodFillTalc = useCallback((px: number, py: number) => {
    const off = talcOffscreenRef.current;
    const ctx = off?.getContext("2d");
    if (!ctx || !off) return;
    const w = off.width;
    const h = off.height;
    if (px < 0 || py < 0 || px >= w || py >= h) return;

    const imgData = ctx.getImageData(0, 0, w, h);
    const data = imgData.data;
    const startOn = data[(py * w + px) * 4 + 3] > 0;
    const visited = new Uint8Array(w * h);
    const stack: number[] = [py * w + px];

    // Правки всегда максимальной уверенности — заливка меняет ту же область
    // синхронно и на карте уверенности (цвет "горячего" конца шкалы).
    const confCtx = talcConfidenceOffscreenRef.current?.getContext("2d");
    const confData = confCtx?.getImageData(0, 0, w, h);
    const [maxR, maxG, maxB] = confidenceColor(1);

    // Заливка меняет связную область на противоположное состояние: клик по
    // пустой зоне закрашивает её тальком, клик по закрашенной — снимает.
    while (stack.length) {
      const p = stack.pop() as number;
      if (visited[p]) continue;
      const idx = p * 4;
      if (data[idx + 3] > 0 !== startOn) continue;
      visited[p] = 1;
      data[idx] = 255;
      data[idx + 1] = 255;
      data[idx + 2] = 255;
      data[idx + 3] = startOn ? 0 : 255;
      if (confData) {
        confData.data[idx] = maxR;
        confData.data[idx + 1] = maxG;
        confData.data[idx + 2] = maxB;
        confData.data[idx + 3] = startOn ? 0 : CONFIDENCE_DISPLAY_ALPHA;
      }

      // 8-связность (с диагоналями) — устойчивее к одно-пиксельным
      // диагональным разрывам на границе от сглаживания при загрузке маски.
      const cx = p % w;
      const cy = (p / w) | 0;
      const hasL = cx > 0;
      const hasR = cx < w - 1;
      const hasT = cy > 0;
      const hasB = cy < h - 1;
      if (hasL) stack.push(p - 1);
      if (hasR) stack.push(p + 1);
      if (hasT) stack.push(p - w);
      if (hasB) stack.push(p + w);
      if (hasL && hasT) stack.push(p - w - 1);
      if (hasR && hasT) stack.push(p - w + 1);
      if (hasL && hasB) stack.push(p + w - 1);
      if (hasR && hasB) stack.push(p + w + 1);
    }

    ctx.putImageData(imgData, 0, 0);
    if (confCtx && confData) confCtx.putImageData(confData, 0, 0);
  }, []);

  const handleTalcPointerDown = useCallback(
    (e: React.PointerEvent<HTMLCanvasElement>) => {
      const tool = talcToolRef.current;
      if (!tool || tool === "cursor" || !talcReadyRef.current) return;
      const imgPt = clientToImagePoint(e.clientX, e.clientY);
      if (!imgPt) return;
      const mp = imageToMaskPoint(imgPt);
      if (!mp) return;

      e.stopPropagation();
      e.preventDefault();
      e.currentTarget.setPointerCapture(e.pointerId);
      viewerRef.current?.setMouseNavEnabled(false);

      if (tool === "fill") {
        pushTalcHistory();
        floodFillTalc(Math.round(mp.x), Math.round(mp.y));
        setTalcDirty(true);
        drawTalcCanvas();
        viewerRef.current?.setMouseNavEnabled(true);
        return;
      }

      pushTalcHistory();
      talcStrokeActiveRef.current = true;
      strokeTalc(null, mp, tool, talcBrushSizeRef.current);
      talcLastPointRef.current = mp;
      setTalcDirty(true);
      drawTalcCanvas();
    },
    [clientToImagePoint, imageToMaskPoint, floodFillTalc, strokeTalc, drawTalcCanvas, pushTalcHistory]
  );

  const handleTalcPointerMove = useCallback(
    (e: React.PointerEvent<HTMLCanvasElement>) => {
      if (!talcStrokeActiveRef.current) return;
      const tool = talcToolRef.current;
      if (!tool || tool === "fill" || tool === "cursor") return;
      const imgPt = clientToImagePoint(e.clientX, e.clientY);
      if (!imgPt) return;
      const mp = imageToMaskPoint(imgPt);
      if (!mp) return;
      strokeTalc(talcLastPointRef.current, mp, tool, talcBrushSizeRef.current);
      talcLastPointRef.current = mp;
      drawTalcCanvas();
    },
    [clientToImagePoint, imageToMaskPoint, strokeTalc, drawTalcCanvas]
  );

  const handleTalcPointerUp = useCallback((e: React.PointerEvent<HTMLCanvasElement>) => {
    talcStrokeActiveRef.current = false;
    talcLastPointRef.current = null;
    viewerRef.current?.setMouseNavEnabled(true);
    if (e.currentTarget.hasPointerCapture(e.pointerId)) {
      e.currentTarget.releasePointerCapture(e.pointerId);
    }
  }, []);

  const saveTalcMask = useCallback(async () => {
    const off = talcOffscreenRef.current;
    if (!off) return;
    const blob: Blob | null = await new Promise((resolve) => off.toBlob(resolve, "image/png"));
    if (!blob) return;
    try {
      const updated = await applyTalcMask(resultId, blob);
      setTalcDirty(false);
      onTalcSavedRef.current?.(updated);
    } catch (err) {
      onTalcSaveErrorRef.current?.(err instanceof Error ? err.message : "Ошибка сохранения маски талька");
    }
  }, [resultId]);

  const undoTalc = useCallback(() => {
    const off = talcOffscreenRef.current;
    const ctx = off?.getContext("2d");
    const undoStack = talcUndoStackRef.current;
    if (!ctx || !off || undoStack.length === 0) return;
    const current = ctx.getImageData(0, 0, off.width, off.height);
    const prev = undoStack.pop() as ImageData;
    talcRedoStackRef.current.push(current);
    ctx.putImageData(prev, 0, 0);
    setTalcDirty(true);
    notifyTalcHistory();
    drawTalcCanvas();
  }, [drawTalcCanvas, notifyTalcHistory]);

  const redoTalc = useCallback(() => {
    const off = talcOffscreenRef.current;
    const ctx = off?.getContext("2d");
    const redoStack = talcRedoStackRef.current;
    if (!ctx || !off || redoStack.length === 0) return;
    const current = ctx.getImageData(0, 0, off.width, off.height);
    const next = redoStack.pop() as ImageData;
    talcUndoStackRef.current.push(current);
    ctx.putImageData(next, 0, 0);
    setTalcDirty(true);
    notifyTalcHistory();
    drawTalcCanvas();
  }, [drawTalcCanvas, notifyTalcHistory]);

  useImperativeHandle(ref, () => ({ saveTalcMask, undoTalc, redoTalc }), [saveTalcMask, undoTalc, redoTalc]);

  const zoomBy = (factor: number) => {
    viewerRef.current?.viewport.zoomBy(factor);
    viewerRef.current?.viewport.applyConstraints(true);
  };

  // Колесо мыши над канвасом талька — зум к курсору. Канвас лежит поверх
  // OSD как соседний (не вложенный) элемент, поэтому событие колеса до OSD
  // не доходит само по себе; нужен нативный (не passive) листенер, чтобы
  // preventDefault реально останавливал прокрутку страницы.
  useEffect(() => {
    const canvas = talcScreenCanvasRef.current;
    if (!canvas) return;

    const onWheel = (e: WheelEvent) => {
      const tool = talcToolRef.current;
      if (layerRef.current !== "talc" || !tool || tool === "cursor") return;
      const viewer = viewerRef.current;
      const osdEl = osdRef.current;
      if (!viewer || !osdEl) return;
      e.preventDefault();
      const bounds = osdEl.getBoundingClientRect();
      const point = viewer.viewport.pointFromPixel(
        new OpenSeadragon.Point(e.clientX - bounds.left, e.clientY - bounds.top)
      );
      const factor = e.deltaY < 0 ? 1.2 : 1 / 1.2;
      viewer.viewport.zoomBy(factor, point);
      viewer.viewport.applyConstraints();
    };

    canvas.addEventListener("wheel", onWheel, { passive: false });
    return () => canvas.removeEventListener("wheel", onWheel);
  }, []);

  const showSvg = layer === "type";
  const showTalc = layer === "talc";

  return (
    <div ref={wrapRef} className="viewer-wrap">
      <div ref={osdRef} className="viewer-osd" />
      <svg
        ref={svgRef}
        className={`viewer-overlay-svg${showSvg ? " is-active" : ""}`}
        aria-hidden={!showSvg}
        onPointerMove={handleSvgPointerMove}
        onPointerUp={handleSvgPointerUp}
        onPointerCancel={handleSvgPointerUp}
      />
      <canvas
        ref={talcScreenCanvasRef}
        className={`viewer-talc-canvas${showTalc ? " is-active" : ""}`}
        aria-hidden={!showTalc}
        style={{
          pointerEvents: showTalc && talcTool && talcTool !== "cursor" ? "auto" : "none",
          cursor: talcTool === "cursor" ? "grab" : talcTool ? "crosshair" : "default",
        }}
        onPointerDown={handleTalcPointerDown}
        onPointerMove={handleTalcPointerMove}
        onPointerUp={handleTalcPointerUp}
        onPointerCancel={handleTalcPointerUp}
      />

      <Paper className="viewer-controls" shadow="md" radius="md" p={4}>
        <Group gap={4}>
          <Tooltip label="Приблизить" withArrow position="left">
            <ActionIcon variant="subtle" color="gray" size="lg" onClick={() => zoomBy(1.25)}>
              <IconPlus size={18} />
            </ActionIcon>
          </Tooltip>
          <Tooltip label="Отдалить" withArrow position="left">
            <ActionIcon variant="subtle" color="gray" size="lg" onClick={() => zoomBy(0.8)}>
              <IconMinus size={18} />
            </ActionIcon>
          </Tooltip>
          <Tooltip label="Показать целиком" withArrow position="left">
            <ActionIcon
              variant="subtle"
              color="nornickel"
              size="lg"
              onClick={() => viewerRef.current?.viewport.goHome(true)}
            >
              <IconZoomScan size={18} />
            </ActionIcon>
          </Tooltip>
        </Group>
      </Paper>

      <Group className="viewer-badge" gap={6}>
        <Badge variant="light" color="gray" size="sm">
          {imageWidth}×{imageHeight}
        </Badge>
        {showTalc && talcLoading && (
          <Badge variant="light" color="blue" size="sm" leftSection={<Loader color="blue" size={10} />}>
            слой талька загружается…
          </Badge>
        )}
        {showTalc && !talcLoading && (
          <Badge variant="light" color="blue" size="sm">
            слой: тальк
          </Badge>
        )}
        {showTalc && talcDirty && (
          <Badge variant="light" color="orange" size="sm">
            есть несохранённые правки
          </Badge>
        )}
        {showSvg && drawStats.total > drawStats.drawn && (
          <Text size="xs" c="dimmed">
            bbox {drawStats.drawn}/{drawStats.total}
          </Text>
        )}
      </Group>
    </div>
  );
});

export default ImageViewer;
