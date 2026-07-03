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
  ThemeIcon,
} from "@mantine/core";
import { IconEdit, IconClick } from "@tabler/icons-react";
import type { Grain, GrainStatus } from "../types";
import { statusLabel } from "../api";

interface Props {
  grain: Grain | null;
  onSave: (id: number, status: GrainStatus, bbox: [number, number, number, number]) => void;
  saving: boolean;
}

export default function GrainEditor({ grain, onSave, saving }: Props) {
  const [status, setStatus] = useState<GrainStatus>("ordinary");
  const [bx, setBx] = useState<number | "">("");
  const [by, setBy] = useState<number | "">("");
  const [bw, setBw] = useState<number | "">("");
  const [bh, setBh] = useState<number | "">("");

  if (!grain) {
    return (
      <Paper p="md" radius="md" shadow="xs" withBorder>
        <Group gap="sm" mb="xs">
          <ThemeIcon size="md" variant="light" color="gray" radius="md">
            <IconClick size={16} />
          </ThemeIcon>
          <Title order={5}>Зерно</Title>
        </Group>
        <Text size="sm" c="dimmed">
          Переключитесь на слой «Тип» и кликните по bbox
        </Text>
      </Paper>
    );
  }

  const currentStatus = status !== grain.status ? status : grain.status;
  const [x, y, w, h] = grain.bbox;

  return (
    <Paper p="md" radius="md" shadow="xs" withBorder>
      <Group gap="sm" mb="md">
        <ThemeIcon size="md" variant="light" color="indigo" radius="md">
          <IconEdit size={16} />
        </ThemeIcon>
        <Title order={5}>Зерно #{grain.id}</Title>
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
        <NumberInput placeholder={`x=${x}`} value={bx} onChange={(v) => setBx(typeof v === "number" ? v : "")} min={0} />
        <NumberInput placeholder={`y=${y}`} value={by} onChange={(v) => setBy(typeof v === "number" ? v : "")} min={0} />
        <NumberInput placeholder={`w=${w}`} value={bw} onChange={(v) => setBw(typeof v === "number" ? v : "")} min={1} />
        <NumberInput placeholder={`h=${h}`} value={bh} onChange={(v) => setBh(typeof v === "number" ? v : "")} min={1} />
      </Group>

      <Button
        fullWidth
        loading={saving}
        onClick={() =>
          onSave(grain.id, currentStatus, [
            typeof bx === "number" ? bx : x,
            typeof by === "number" ? by : y,
            typeof bw === "number" ? bw : w,
            typeof bh === "number" ? bh : h,
          ])
        }
      >
        Сохранить правку
      </Button>
      <Text size="xs" c="dimmed" ta="center" mt="xs">
        Сейчас: {statusLabel(grain.status)}
      </Text>
    </Paper>
  );
}
