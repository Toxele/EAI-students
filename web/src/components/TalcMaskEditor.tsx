import {
  Paper,
  Title,
  Text,
  Group,
  SimpleGrid,
  Button,
  Slider,
  Stack,
  SegmentedControl,
  Divider,
} from "@mantine/core";
import { IconPointer, IconPencil, IconEraser, IconBucketDroplet, IconDeviceFloppy } from "@tabler/icons-react";
import type { TalcTool, TalcViewMode } from "../types";
import { CONFIDENCE_CSS_GRADIENT } from "../confidenceColormap";
import UndoRedoButtons from "./UndoRedoButtons";

interface Props {
  tool: TalcTool;
  onToolChange: (tool: TalcTool) => void;
  brushSize: number;
  onBrushSizeChange: (size: number) => void;
  viewMode: TalcViewMode;
  onViewModeChange: (mode: TalcViewMode) => void;
  confidenceAvailable: boolean;
  dirty: boolean;
  saving: boolean;
  onSave: () => void;
  canUndo: boolean;
  canRedo: boolean;
  onUndo: () => void;
  onRedo: () => void;
}

const TOOLS: { value: TalcTool; label: string; icon: typeof IconPointer }[] = [
  { value: "cursor", label: "Курсор", icon: IconPointer },
  { value: "pencil", label: "Карандаш", icon: IconPencil },
  { value: "eraser", label: "Ластик", icon: IconEraser },
  { value: "fill", label: "Заливка", icon: IconBucketDroplet },
];

export default function TalcMaskEditor({
  tool,
  onToolChange,
  brushSize,
  onBrushSizeChange,
  viewMode,
  onViewModeChange,
  confidenceAvailable,
  dirty,
  saving,
  onSave,
  canUndo,
  canRedo,
  onUndo,
  onRedo,
}: Props) {
  return (
    <Paper p="md" radius="xl" shadow="xs" withBorder>
      <Group gap="sm" mb="md" justify="space-between">
        <Title order={5} tt="uppercase" fz="sm">
          Правка маски талька
        </Title>
        <UndoRedoButtons canUndo={canUndo} canRedo={canRedo} onUndo={onUndo} onRedo={onRedo} />
      </Group>

      <Stack gap={4} mb="md">
        <Text size="sm" c="dimmed">
          Отображение
        </Text>
        <SegmentedControl
          fullWidth
          radius="xl"
          value={viewMode}
          onChange={(v) => onViewModeChange(v as TalcViewMode)}
          disabled={!confidenceAvailable}
          data={[
            { value: "mask", label: "Маска" },
            { value: "confidence", label: "Уверенность" },
          ]}
        />
        {!confidenceAvailable && (
          <Text size="xs" c="dimmed">
            Уверенность модели недоступна для этого результата.
          </Text>
        )}
        {confidenceAvailable && viewMode === "confidence" && (
          <Stack gap={2} mt={4}>
            <div
              style={{
                height: 12,
                borderRadius: 6,
                background: CONFIDENCE_CSS_GRADIENT,
              }}
            />
            <Group justify="space-between">
              <Text size="xs" c="dimmed">
                50%
              </Text>
              <Text size="xs" c="dimmed">
                75%
              </Text>
              <Text size="xs" c="dimmed">
                100%
              </Text>
            </Group>
          </Stack>
        )}
      </Stack>

      <Divider my="sm" />

      <SimpleGrid cols={2} spacing={8} mb="md">
        {TOOLS.map(({ value, label, icon: Icon }) => (
          <Button
            key={value}
            fullWidth
            radius="xl"
            variant={tool === value ? "filled" : "light"}
            color="nornickel"
            leftSection={<Icon size={16} />}
            onClick={() => onToolChange(value)}
          >
            {label}
          </Button>
        ))}
      </SimpleGrid>

      {(tool === "pencil" || tool === "eraser") && (
        <Stack gap={4} mb="md">
          <Text size="sm" c="dimmed">
            Толщина: {brushSize}
          </Text>
          <Slider min={2} max={60} value={brushSize} onChange={onBrushSizeChange} color="nornickel" />
        </Stack>
      )}

      <Button
        fullWidth
        radius="xl"
        leftSection={<IconDeviceFloppy size={16} />}
        loading={saving}
        disabled={!dirty}
        onClick={onSave}
      >
        Сохранить маску
      </Button>
    </Paper>
  );
}
