import { useEffect, useRef, useCallback, useState } from "react";
import OpenSeadragon from "openseadragon";
import { Badge, Paper, ActionIcon, Tooltip, Group, Text } from "@mantine/core";
import { IconPlus, IconMinus, IconZoomScan } from "@tabler/icons-react";
import type { Grain, LayerMode } from "../types";
import { absUrl, grainColor } from "../api";

const MAX_SVG_GRAINS = 800;

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
}: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const osdRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const viewerRef = useRef<OpenSeadragon.Viewer | null>(null);
  const talcItemRef = useRef<OpenSeadragon.TiledImage | null>(null);
  const typeItemRef = useRef<OpenSeadragon.TiledImage | null>(null);
  const [drawStats, setDrawStats] = useState({ drawn: 0, total: 0 });

  const layerRef = useRef(layer);
  const grainsRef = useRef(grains);
  const selectedIdRef = useRef(selectedId);
  const onSelectRef = useRef(onSelectGrain);
  layerRef.current = layer;
  grainsRef.current = grains;
  selectedIdRef.current = selectedId;
  onSelectRef.current = onSelectGrain;

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
      if (g.id === selectedIdRef.current) rect.classList.add("selected");
      rect.addEventListener("click", (e) => {
        e.stopPropagation();
        onSelectRef.current(g.id);
      });
      svg.appendChild(rect);
    }
  }, []);

  const refresh = useCallback(() => {
    syncOpacities();
    drawSvgBboxes();
  }, [syncOpacities, drawSvgBboxes]);

  const addAlignedImage = useCallback(
    (viewer: OpenSeadragon.Viewer, url: string): Promise<OpenSeadragon.TiledImage | null> => {
      const base = viewer.world.getItemAt(0);
      if (!base) return Promise.resolve(null);
      const b = base.getBounds();

      return new Promise((resolve) => {
        try {
          const ret = viewer.addSimpleImage({
            url: absUrl(url),
            x: b.x,
            y: b.y,
            width: b.width,
            height: b.height,
            opacity: 0,
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
        const item = await addAlignedImage(viewer, talcDisplayUrl);
        if (item) {
          talcItemRef.current = item;
          item.addHandler("fully-loaded-change", () => refresh());
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
              color="indigo"
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
        {layer === "talc" && (
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
