import { useState } from "react";
import {
  Paper,
  Title,
  Text,
  Select,
  NumberInput,
  Button,
  Stack,
  Group,
  Progress,
} from "@mantine/core";
import type { Grain, GrainStatus } from "../types";
import { statusLabel } from "../api";
import UndoRedoButtons from "./UndoRedoButtons";

interface Props {
  grain: Grain | null;
  onBboxChange: (id: number, bbox: [number, number, number, number]) => void;
  onSave: (id: number, status: GrainStatus, bbox: [number, number, number, number]) => void;
  saving: boolean;
  canUndo: boolean;
  canRedo: boolean;
  onUndo: () => void;
  onRedo: () => void;
}

export default function GrainEditor({ grain, onBboxChange, onSave, saving, canUndo, canRedo, onUndo, onRedo }: Props) {
  const [status, setStatus] = useState<GrainStatus>("ordinary");

  if (!grain) {
    return (
      <Paper p="md" radius="xl" shadow="xs" withBorder>
        <Group gap="sm" mb="xs" justify="space-between">
          <Title order={5}>Зерно</Title>
          <UndoRedoButtons canUndo={canUndo} canRedo={canRedo} onUndo={onUndo} onRedo={onRedo} />
        </Group>
        <Text size="sm" c="dimmed">
          Переключитесь на слой «Тип» и кликните по bbox
        </Text>
      </Paper>
    );
  }

  const currentStatus = status !== grain.status ? status : grain.status;
  const [x, y, w, h] = grain.bbox;

  const updateBbox = (index: 0 | 1 | 2 | 3, value: number | string) => {
    if (typeof value !== "number") return;
    const next: [number, number, number, number] = [x, y, w, h];
    next[index] = value;
    onBboxChange(grain.id, next);
  };

  return (
    <Paper p="md" radius="xl" shadow="xs" withBorder>
      <Group gap="sm" mb="md" justify="space-between">
        <Title order={5}>Зерно #{grain.id}</Title>
        <UndoRedoButtons canUndo={canUndo} canRedo={canRedo} onUndo={onUndo} onRedo={onRedo} />
      </Group>

      <Stack gap="xs" mb="md">
        <Group justify="space-between">
          <Text size="sm" c="dimmed">
            рядовое
          </Text>
          <Text size="sm" fw={600}>
            {(grain.conf_ordinary * 100).toFixed(0)}%
          </Text>
        </Group>
        <Progress value={grain.conf_ordinary * 100} color="green" size="sm" />
        <Group justify="space-between">
          <Text size="sm" c="dimmed">
            тонкое
          </Text>
          <Text size="sm" fw={600}>
            {(grain.conf_thin * 100).toFixed(0)}%
          </Text>
        </Group>
        <Progress value={grain.conf_thin * 100} color="red" size="sm" />
      </Stack>

      <Select
        label="Класс"
        mb="sm"
        value={currentStatus}
        onChange={(v) => setStatus((v as GrainStatus) ?? "ordinary")}
        data={[
          { value: "ordinary", label: "Рядовое" },
          { value: "thin", label: "Тонкое" },
          { value: "uncertain", label: "Неопределённый" },
          { value: "false_positive", label: "Ложная детекция" },
        ]}
      />

      <Text size="sm" fw={500} mb={4}>
        Bbox (x, y, w, h)
      </Text>
      <Group grow mb="md">
        <NumberInput label="x" value={x} onChange={(v) => updateBbox(0, v)} min={0} />
        <NumberInput label="y" value={y} onChange={(v) => updateBbox(1, v)} min={0} />
        <NumberInput label="w" value={w} onChange={(v) => updateBbox(2, v)} min={1} />
        <NumberInput label="h" value={h} onChange={(v) => updateBbox(3, v)} min={1} />
      </Group>

      <Button
        fullWidth
        radius="xl"
        loading={saving}
        onClick={() => onSave(grain.id, currentStatus, grain.bbox)}
      >
        Сохранить правку
      </Button>
      <Text size="xs" c="dimmed" ta="center" mt="xs">
        Сейчас: {statusLabel(grain.status)}
      </Text>
    </Paper>
  );
}
