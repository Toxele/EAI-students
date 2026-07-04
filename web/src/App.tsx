import { useState, useCallback, useRef, useEffect } from "react";
import {
  AppShell,
  Group,
  Title,
  Text,
  Select,
  Button,
  SegmentedControl,
  Alert,
  Stack,
  Paper,
  Badge,
  Anchor,
  Loader,
  Overlay,
  Box,
  ThemeIcon,
} from "@mantine/core";
import {
  IconUpload,
  IconPhoto,
  IconLayersLinked,
  IconFileTypePdf,
  IconFileTypeCsv,
  IconDownload,
  IconAlertCircle,
} from "@tabler/icons-react";
import type { AnalysisResult, Grain, GrainStatus, LayerMode, TalcTool } from "./types";
import { analyzeFile, applyCorrections, absUrl } from "./api";
import ImageViewer, { type ImageViewerHandle } from "./components/ImageViewer";
import MetricsPanel from "./components/MetricsPanel";
import GrainEditor from "./components/GrainEditor";
import TalcMaskEditor from "./components/TalcMaskEditor";

const LAYER_OPTIONS = [
  { label: "Обзор", value: "overview" },
  { label: "Тальк", value: "talc" },
  { label: "Тип", value: "type" },
];

// Nornickel corporate palette (Стандарт «Фирменный стиль», стр. 35):
// синий Pantone 3005 = #0077C8, темно-синий Pantone 2945 = #004C97
const BRAND_BLUE = "#004C97";

// Официальный логотип на белой плашке-подложке — обеспечивает контраст
// знака на цветном/тёмном фоне (правило стр. 18).
function BrandMark() {
  return (
    <div
      style={{
        background: "#fff",
        borderRadius: 12,
        padding: "6px 14px",
        display: "flex",
        alignItems: "center",
        flexShrink: 0,
      }}
    >
      <img src="/nornickel-logo.png" alt="Норникель" height={28} />
    </div>
  );
}

// Декоративный элемент «лента» (стр. 26-30): непрерывная полоса с приподнятым
// закруглённым сегментом, используется как тонкий фирменный акцент.
function NornickelRibbon() {
  return (
    <div className="nn-ribbon">
      <div className="nn-ribbon-bump" style={{ left: "30%", width: "22%" }} />
    </div>
  );
}

export default function App() {
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [layer, setLayer] = useState<LayerMode>("overview");
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [mode, setMode] = useState<string>("auto");
  const fileRef = useRef<HTMLInputElement>(null);
  const imageViewerRef = useRef<ImageViewerHandle>(null);

  const [talcTool, setTalcTool] = useState<TalcTool>("cursor");
  const [talcBrushSize, setTalcBrushSize] = useState(16);
  const [talcDirty, setTalcDirty] = useState(false);
  const [talcSaving, setTalcSaving] = useState(false);
  const [talcUndoAvailable, setTalcUndoAvailable] = useState(false);
  const [talcRedoAvailable, setTalcRedoAvailable] = useState(false);

  const grainUndoStackRef = useRef<Grain[][]>([]);
  const grainRedoStackRef = useRef<Grain[][]>([]);
  const [grainUndoAvailable, setGrainUndoAvailable] = useState(false);
  const [grainRedoAvailable, setGrainRedoAvailable] = useState(false);
  const GRAIN_HISTORY_LIMIT = 50;

  const selectedGrain = result?.grains.find((g) => g.id === selectedId) ?? null;

  // "Тип" доступна только для панорамы: и при явном выборе режима, и при
  // "авто", если снимок определился как панорама (result.mode).
  const layerOptions = result
    ? LAYER_OPTIONS.filter((o) => o.value !== "type" || result.mode === "panorama")
    : LAYER_OPTIONS;

  const handleFile = async (file: File | null) => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setSelectedId(null);
    setTalcDirty(false);
    grainUndoStackRef.current = [];
    grainRedoStackRef.current = [];
    setGrainUndoAvailable(false);
    setGrainRedoAvailable(false);
    try {
      const data = await analyzeFile(
        file,
        mode as "auto" | "panorama" | "detail"
      );
      setResult(data);
      setLayer("overview");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Ошибка анализа");
    } finally {
      setLoading(false);
    }
  };

  const handleSaveTalcMask = useCallback(async () => {
    setTalcSaving(true);
    try {
      await imageViewerRef.current?.saveTalcMask();
    } finally {
      setTalcSaving(false);
    }
  }, []);

  const handleTalcSaved = useCallback((updated: AnalysisResult) => {
    setResult((prev) =>
      prev
        ? {
            ...prev,
            ...updated,
            grains: updated.grains,
            counts: updated.counts,
            metrics: updated.metrics,
            image_url: prev.image_url,
            talc_display_url: prev.talc_display_url,
            talc_layer_url: prev.talc_layer_url,
            type_layer_url: prev.type_layer_url,
            original_width: prev.original_width,
            original_height: prev.original_height,
          }
        : prev
    );
  }, []);

  const handleTalcSaveError = useCallback((message: string) => {
    setError(message);
  }, []);

  const handleTalcHistoryChange = useCallback((canUndo: boolean, canRedo: boolean) => {
    setTalcUndoAvailable(canUndo);
    setTalcRedoAvailable(canRedo);
  }, []);

  // Снимок массива зёрен перед мутирующим действием (начало drag / перед
  // сохранением правки) — основа для undo вкладки «Тип». Сами объекты
  // зёрен не мутируются на месте (везде spread), поэтому неглубокой копии
  // массива достаточно, чтобы сохранить прежнее состояние.
  const pushGrainSnapshot = useCallback(() => {
    if (!result) return;
    const stack = grainUndoStackRef.current;
    stack.push(result.grains);
    if (stack.length > GRAIN_HISTORY_LIMIT) stack.shift();
    grainRedoStackRef.current = [];
    setGrainUndoAvailable(true);
    setGrainRedoAvailable(false);
  }, [result]);

  // Применяет снимок зёрен: сразу локально (для отклика UI), затем
  // синхронизирует с бэкендом (пересчёт метрик/сорта) минимальным диффом.
  const applyGrainSnapshot = useCallback(
    async (target: Grain[]) => {
      if (!result) return;
      const targetById = new Map(target.map((g) => [g.id, g]));
      const updates: { id: number; status?: GrainStatus; bbox?: number[] }[] = [];
      for (const g of result.grains) {
        const was = targetById.get(g.id);
        if (!was) continue;
        const bboxChanged = was.bbox.some((v, i) => v !== g.bbox[i]);
        const statusChanged = was.status !== g.status;
        if (bboxChanged || statusChanged) {
          updates.push({
            id: g.id,
            ...(statusChanged ? { status: g.status } : {}),
            ...(bboxChanged ? { bbox: g.bbox } : {}),
          });
        }
      }

      const resultId = result.result_id;
      setResult((prev) => (prev ? { ...prev, grains: target } : prev));
      if (updates.length === 0) return;

      try {
        const updated = await applyCorrections(resultId, updates);
        setResult((prev) =>
          prev
            ? {
                ...prev,
                ...updated,
                grains: updated.grains,
                counts: updated.counts,
                metrics: updated.metrics,
                image_url: prev.image_url,
                talc_display_url: prev.talc_display_url,
                talc_layer_url: prev.talc_layer_url,
                type_layer_url: prev.type_layer_url,
                original_width: prev.original_width,
                original_height: prev.original_height,
              }
            : prev
        );
      } catch (err) {
        setError(err instanceof Error ? err.message : "Ошибка отмены/повтора");
      }
    },
    [result]
  );

  const undoGrain = useCallback(() => {
    const undoStack = grainUndoStackRef.current;
    if (!result || undoStack.length === 0) return;
    const target = undoStack.pop() as Grain[];
    grainRedoStackRef.current.push(result.grains);
    setGrainUndoAvailable(undoStack.length > 0);
    setGrainRedoAvailable(true);
    void applyGrainSnapshot(target);
  }, [result, applyGrainSnapshot]);

  const redoGrain = useCallback(() => {
    const redoStack = grainRedoStackRef.current;
    if (!result || redoStack.length === 0) return;
    const target = redoStack.pop() as Grain[];
    grainUndoStackRef.current.push(result.grains);
    setGrainRedoAvailable(redoStack.length > 0);
    setGrainUndoAvailable(true);
    void applyGrainSnapshot(target);
  }, [result, applyGrainSnapshot]);

  const handleGrainBboxChange = useCallback(
    (id: number, bbox: [number, number, number, number]) => {
      setResult((prev) => {
        if (!prev) return prev;
        const grains = prev.grains.map((g) =>
          g.id === id ? { ...g, bbox, area: bbox[2] * bbox[3] } : g
        );
        return { ...prev, grains };
      });
    },
    []
  );

  const handleSaveGrain = useCallback(
    async (
      id: number,
      status: GrainStatus,
      bbox: [number, number, number, number]
    ) => {
      if (!result) return;
      pushGrainSnapshot();
      setSaving(true);
      try {
        const updated = await applyCorrections(result.result_id, [
          { id, status, bbox },
        ]);
        setResult((prev) =>
          prev
            ? {
                ...prev,
                ...updated,
                grains: updated.grains,
                counts: updated.counts,
                metrics: updated.metrics,
                image_url: prev.image_url,
                talc_display_url: prev.talc_display_url ?? prev.talc_layer_url,
                talc_layer_url: prev.talc_layer_url,
                original_width: prev.original_width,
                original_height: prev.original_height,
              }
            : prev
        );
      } catch (err) {
        setError(err instanceof Error ? err.message : "Ошибка сохранения");
      } finally {
        setSaving(false);
      }
    },
    [result, pushGrainSnapshot]
  );

  // Диспетчер общей кнопки/сочетания отмены-повтора: смотрит на активную
  // вкладку и вызывает историю талька (ImageViewer) или зёрен (локальную).
  const canUndo = layer === "talc" ? talcUndoAvailable : layer === "type" ? grainUndoAvailable : false;
  const canRedo = layer === "talc" ? talcRedoAvailable : layer === "type" ? grainRedoAvailable : false;

  const handleUndo = useCallback(() => {
    if (layer === "talc") imageViewerRef.current?.undoTalc();
    else if (layer === "type") undoGrain();
  }, [layer, undoGrain]);

  const handleRedo = useCallback(() => {
    if (layer === "talc") imageViewerRef.current?.redoTalc();
    else if (layer === "type") redoGrain();
  }, [layer, redoGrain]);

  useEffect(() => {
    if (result && result.mode !== "panorama" && layer === "type") {
      setLayer("overview");
    }
  }, [result, layer]);

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA") return;
      // e.code — физическая клавиша (не зависит от раскладки, в отличие от
      // e.key: на русской раскладке та же клавиша даёт "я", а не "z").
      if (!(e.ctrlKey || e.metaKey) || e.code !== "KeyZ") return;
      e.preventDefault();
      if (e.shiftKey) handleRedo();
      else handleUndo();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [handleUndo, handleRedo]);

  return (
    <AppShell
      header={{ height: 64 }}
      footer={result ? { height: 56 } : undefined}
      padding="md"
      styles={{
        root: { height: "100vh" },
        main: { background: "#f4f6fa", display: "flex", flexDirection: "column" },
        header: { background: BRAND_BLUE, border: "none" },
        footer: { background: BRAND_BLUE, border: "none" },
      }}
    >
      <AppShell.Header px="lg">
        <Group h="100%" justify="space-between" wrap="nowrap">
          <Group gap="sm">
            <BrandMark />
            <Text size="xs" c="#B9CBEE">
              Ore Analyzer · AI-анализ шлифов руды
            </Text>
          </Group>

          <Group gap="sm" wrap="nowrap">
            <Select
              w={160}
              size="sm"
              radius="xl"
              value={mode}
              onChange={(v) => setMode(v ?? "auto")}
              data={[
                { value: "auto", label: "Режим: авто" },
                { value: "panorama", label: "Панорама" },
                { value: "detail", label: "Близкое фото" },
              ]}
            />
            <input
              ref={fileRef}
              type="file"
              accept="image/*"
              hidden
              onChange={(e) => handleFile(e.target.files?.[0] ?? null)}
            />
            <Button
              radius="xl"
              color="nornickel.5"
              leftSection={<IconUpload size={16} />}
              onClick={() => fileRef.current?.click()}
              loading={loading}
            >
              Загрузить фото
            </Button>
          </Group>
        </Group>
      </AppShell.Header>

      {result && (
        <AppShell.Footer px="lg">
          <Group h="100%" justify="space-between">
            <Group gap="xs">
              <Button
                component="a"
                href={absUrl(result.pdf_url)}
                download
                variant="outline"
                color="#fff"
                radius="xl"
                leftSection={<IconFileTypePdf size={16} />}
                size="sm"
              >
                PDF
              </Button>
              <Button
                component="a"
                href={absUrl(result.labels_url!)}
                download
                variant="outline"
                color="#fff"
                radius="xl"
                leftSection={<IconDownload size={16} />}
                size="sm"
              >
                labels.json
              </Button>
              <Button
                component="a"
                href={absUrl(result.talc_layer_url!)}
                download
                variant="outline"
                color="#fff"
                radius="xl"
                leftSection={<IconLayersLinked size={16} />}
                size="sm"
              >
                Маска талька
              </Button>
              <Button
                component="a"
                href={absUrl(result.csv_url!)}
                download
                variant="outline"
                color="#fff"
                radius="xl"
                leftSection={<IconFileTypeCsv size={16} />}
                size="sm"
              >
                CSV
              </Button>
            </Group>
            <Text size="sm" c="#B9CBEE">
              id: {result.result_id} · {result.mode} · {result.grains.length} зёрен
            </Text>
          </Group>
        </AppShell.Footer>
      )}

      <AppShell.Main style={{ height: "100vh", display: "flex", flexDirection: "column" }}>
        <Stack gap="md" h="100%">
          <NornickelRibbon />

          {error && (
            <Alert
              icon={<IconAlertCircle size={16} />}
              color="red"
              variant="light"
              withCloseButton
              onClose={() => setError(null)}
            >
              {error}
            </Alert>
          )}

          {result && (
            <Paper p="sm" radius="xl" shadow="xs" withBorder>
              <Group justify="space-between">
                <SegmentedControl
                  value={layer}
                  onChange={(v) => setLayer(v as LayerMode)}
                  radius="xl"
                  data={layerOptions}
                />
                <Badge variant="light" color="nornickel" size="lg" radius="sm">
                  {result.original_width}×{result.original_height}
                </Badge>
              </Group>
            </Paper>
          )}

          <Box style={{ flex: 1, minHeight: 0, display: "grid", gridTemplateColumns: result ? "1fr 340px" : "1fr", gap: 16 }}>
            <Box style={{ position: "relative", minHeight: 400 }}>
              {loading && (
                <Overlay color="#fff" backgroundOpacity={0.7} zIndex={10}>
                  <Stack align="center" justify="center" h="100%" gap="sm">
                    <Loader color="nornickel" type="dots" />
                    <Text c="dimmed">Анализ изображения…</Text>
                  </Stack>
                </Overlay>
              )}
              {result?.image_url ? (
                <ImageViewer
                  ref={imageViewerRef}
                  imageUrl={result.image_url}
                  talcMaskUrl={result.talc_layer_url}
                  typeLayerUrl={result.type_layer_url}
                  grains={result.grains}
                  layer={layer}
                  imageWidth={result.original_width}
                  imageHeight={result.original_height}
                  selectedId={selectedId}
                  onSelectGrain={setSelectedId}
                  onGrainBboxChange={handleGrainBboxChange}
                  onGrainDragStart={pushGrainSnapshot}
                  resultId={result.result_id}
                  talcTool={layer === "talc" ? talcTool : null}
                  talcBrushSize={talcBrushSize}
                  onTalcDirtyChange={setTalcDirty}
                  onTalcHistoryChange={handleTalcHistoryChange}
                  onTalcSaved={handleTalcSaved}
                  onTalcSaveError={handleTalcSaveError}
                />
              ) : (
                <Paper
                  h="100%"
                  radius="xl"
                  shadow="xs"
                  withBorder
                  style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    background: "linear-gradient(135deg, #fff 0%, #eaf4fc 100%)",
                  }}
                >
                  {!loading && (
                    <Stack align="center" gap="md" maw={420} ta="center" p="xl">
                      <ThemeIcon size={64} radius="xl" variant="light" color="nornickel">
                        <IconPhoto size={32} />
                      </ThemeIcon>
                      <Title order={3}>Загрузите изображение</Title>
                      <Text c="dimmed" size="sm">
                        Выберите режим (панорама или близкое фото), нажмите «Загрузить фото».
                        После анализа используйте zoom для деталей — на слое «Тип» можно править зёрна.
                      </Text>
                      <Button
                        radius="xl"
                        leftSection={<IconUpload size={16} />}
                        onClick={() => fileRef.current?.click()}
                      >
                        Выбрать файл
                      </Button>
                    </Stack>
                  )}
                </Paper>
              )}
            </Box>

            {result && (
              <Stack gap="md" style={{ overflowY: "auto", maxHeight: "100%" }}>
                <MetricsPanel result={result} />
                {layer === "type" && (
                  <GrainEditor
                    key={selectedGrain?.id ?? "none"}
                    grain={selectedGrain}
                    onBboxChange={handleGrainBboxChange}
                    onSave={handleSaveGrain}
                    saving={saving}
                    canUndo={canUndo}
                    canRedo={canRedo}
                    onUndo={handleUndo}
                    onRedo={handleRedo}
                  />
                )}
                {layer === "talc" && (
                  <TalcMaskEditor
                    tool={talcTool}
                    onToolChange={setTalcTool}
                    brushSize={talcBrushSize}
                    onBrushSizeChange={setTalcBrushSize}
                    dirty={talcDirty}
                    saving={talcSaving}
                    onSave={handleSaveTalcMask}
                    canUndo={canUndo}
                    canRedo={canRedo}
                    onUndo={handleUndo}
                    onRedo={handleRedo}
                  />
                )}
              </Stack>
            )}
          </Box>
        </Stack>
      </AppShell.Main>
    </AppShell>
  );
}
