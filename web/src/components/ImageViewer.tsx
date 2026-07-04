import { useEffect, useRef, useCallback, useState } from "react";
import OpenSeadragon from "openseadragon";
import { Badge, Paper, ActionIcon, Tooltip, Group, Text, Loader } from "@mantine/core";
import { IconPlus, IconMinus, IconZoomScan } from "@tabler/icons-react";
import type { Grain, LayerMode } from "../types";
import { absUrl, grainColor } from "../api";

const MAX_SVG_GRAINS = 800;

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
  talcDisplayUrl: string | null;
  typeLayerUrl: string | null;
  grains: Grain[];
  layer: LayerMode;
  imageWidth: number;
  imageHeight: number;
  selectedId: number | null;
  onSelectGrain: (id: number | null) => void;
  onGrainBboxChange?: (id: number, bbox: [number, number, number, number]) => void;
}

export default function ImageViewer({
  imageUrl,
  talcDisplayUrl,
  typeLayerUrl,
  grains,
  layer,
  imageWidth,
  imageHeight,
  selectedId,
  onSelectGrain,
  onGrainBboxChange,
}: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const osdRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const viewerRef = useRef<OpenSeadragon.Viewer | null>(null);
  const talcItemRef = useRef<OpenSeadragon.TiledImage | null>(null);
  const typeItemRef = useRef<OpenSeadragon.TiledImage | null>(null);
  const [drawStats, setDrawStats] = useState({ drawn: 0, total: 0 });
  const [talcLoading, setTalcLoading] = useState(false);

  const layerRef = useRef(layer);
  const grainsRef = useRef(grains);
  const selectedIdRef = useRef(selectedId);
  const onSelectRef = useRef(onSelectGrain);
  const onBboxChangeRef = useRef(onGrainBboxChange);
  const dragStateRef = useRef<DragState | null>(null);
  const pendingBboxRef = useRef<{ id: number; bbox: [number, number, number, number] } | null>(null);
  const rafRef = useRef<number | null>(null);
  layerRef.current = layer;
  grainsRef.current = grains;
  selectedIdRef.current = selectedId;
  onSelectRef.current = onSelectGrain;
  onBboxChangeRef.current = onGrainBboxChange;

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

    const base = viewer.world.getItemAt(0);
    const talc = talcItemRef.current;
    const mode = layerRef.current;

    if (mode === "talc" && talc?.getFullyLoaded()) {
      base.setOpacity(0);
      talc.setOpacity(1);
    } else {
      base.setOpacity(1);
      talc?.setOpacity(0);
    }
    typeItemRef.current?.setOpacity(0);
  }, []);

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
  }, [syncOpacities, drawSvgBboxes]);

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
          // given (dropping height silently). Talc/type layers share the
          // base image's aspect ratio, so width alone aligns them exactly.
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
      if (talcDisplayUrl && !talcItemRef.current) {
        setTalcLoading(true);
        const item = await addAlignedImage(viewer, talcDisplayUrl, true);
        if (item) {
          talcItemRef.current = item;
          setTalcLoading(!item.getFullyLoaded());
          item.addHandler("fully-loaded-change", () => {
            setTalcLoading(!item.getFullyLoaded());
            refresh();
          });
        } else {
          setTalcLoading(false);
        }
      }
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
      talcItemRef.current = null;
      typeItemRef.current = null;
      if (rafRef.current != null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
    };
  }, [imageUrl, talcDisplayUrl, typeLayerUrl, addAlignedImage, refresh]);

  useEffect(() => {
    refresh();
    // Доп. перерисовка после смены слоя — OSD иногда не успевает
    const t = window.setTimeout(refresh, 50);
    return () => window.clearTimeout(t);
  }, [layer, grains, selectedId, refresh]);

  const zoomBy = (factor: number) => {
    viewerRef.current?.viewport.zoomBy(factor);
    viewerRef.current?.viewport.applyConstraints(true);
  };

  const showSvg = layer === "type";

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
        {layer === "talc" && talcLoading && (
          <Badge variant="light" color="blue" size="sm" leftSection={<Loader color="blue" size={10} />}>
            слой талька загружается…
          </Badge>
        )}
        {layer === "talc" && !talcLoading && (
          <Badge variant="light" color="blue" size="sm">
            слой: тальк
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
}
