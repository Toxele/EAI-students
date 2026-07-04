import { useState, useCallback, useRef } from "react";
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
import type { AnalysisResult, GrainStatus, LayerMode } from "./types";
import { analyzeFile, applyCorrections, absUrl } from "./api";
import ImageViewer from "./components/ImageViewer";
import MetricsPanel from "./components/MetricsPanel";
import GrainEditor from "./components/GrainEditor";

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

  const selectedGrain = result?.grains.find((g) => g.id === selectedId) ?? null;

  const handleFile = async (file: File | null) => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setSelectedId(null);
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
    [result]
  );

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
                  data={LAYER_OPTIONS}
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
                  imageUrl={result.image_url}
                  talcDisplayUrl={result.talc_display_url}
                  typeLayerUrl={result.type_layer_url}
                  grains={result.grains}
                  layer={layer}
                  imageWidth={result.original_width}
                  imageHeight={result.original_height}
                  selectedId={selectedId}
                  onSelectGrain={setSelectedId}
                  onGrainBboxChange={handleGrainBboxChange}
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
                <GrainEditor
                  key={selectedGrain?.id ?? "none"}
                  grain={selectedGrain}
                  onBboxChange={handleGrainBboxChange}
                  onSave={handleSaveGrain}
                  saving={saving}
                />
              </Stack>
            )}
          </Box>
        </Stack>
      </AppShell.Main>
    </AppShell>
  );
}
