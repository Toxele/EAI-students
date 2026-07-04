import {
  Paper,
  Title,
  Text,
  Stack,
  Group,
  Badge,
  ThemeIcon,
  Divider,
  SimpleGrid,
} from "@mantine/core";
import {
  IconDiamond,
  IconLayersSubtract,
  IconChartDots,
} from "@tabler/icons-react";
import type { AnalysisResult } from "../types";

interface Props {
  result: AnalysisResult;
}

function MetricRow({ label, value }: { label: string; value: string | number }) {
  return (
    <Group justify="space-between">
      <Text size="sm" c="dimmed">
        {label}
      </Text>
      <Text size="sm" fw={600}>
        {value}
      </Text>
    </Group>
  );
}

export default function MetricsPanel({ result }: Props) {
  const { counts, metrics } = result;
  const talcStr =
    metrics.talc_available && metrics.talc_percent != null
      ? `${metrics.talc_percent.toFixed(1)}%`
      : "н/д";

  const sortColor =
    result.sort_code === "talc" ? "blue" : result.sort_code === "ordinary" ? "green" : "orange";

  return (
    <Stack gap="md">
      <Paper p="md" radius="xl" shadow="xs" withBorder>
        <Group gap="sm" mb="sm" className="nn-section-title">
          <ThemeIcon size="md" variant="light" color={sortColor} radius="md">
            <IconDiamond size={16} />
          </ThemeIcon>
          <Title order={5} tt="uppercase" fz="sm">
            Итоговый сорт
          </Title>
        </Group>
        <Badge size="lg" variant="light" color={sortColor} mb="xs" radius="sm">
          {result.sort_label_ru}
        </Badge>
        <Text size="sm" c="dimmed" lh={1.5} className="nn-quote-bar">
          {result.conclusion.replace(/\*\*/g, "")}
        </Text>
      </Paper>

      <Paper p="md" radius="xl" shadow="xs" withBorder>
        <Group gap="sm" mb="sm" className="nn-section-title">
          <ThemeIcon size="md" variant="light" color="blue" radius="md">
            <IconLayersSubtract size={16} />
          </ThemeIcon>
          <Title order={5} tt="uppercase" fz="sm">
            Тальк
          </Title>
        </Group>
        <MetricRow label="Доля талька" value={talcStr} />
        <Text size="xs" c="dimmed" mt="xs">
          Заглушка модели — до подключения обученного классификатора
        </Text>
      </Paper>

      <Paper p="md" radius="xl" shadow="xs" withBorder>
        <Group gap="sm" mb="sm" className="nn-section-title">
          <ThemeIcon size="md" variant="light" color="nornickel" radius="md">
            <IconChartDots size={16} />
          </ThemeIcon>
          <Title order={5} tt="uppercase" fz="sm">
            Включения k / l / j
          </Title>
        </Group>
        <SimpleGrid cols={3} mb="sm">
          <Paper p="xs" radius="sm" bg="gray.0" ta="center">
            <Text size="xs" c="dimmed">
              k
            </Text>
            <Text fw={700} size="lg">
              {counts.total_k}
            </Text>
          </Paper>
          <Paper p="xs" radius="sm" bg="green.0" ta="center">
            <Text size="xs" c="dimmed">
              l
            </Text>
            <Text fw={700} size="lg" c="green.8">
              {counts.ordinary_l}
            </Text>
          </Paper>
          <Paper p="xs" radius="sm" bg="red.0" ta="center">
            <Text size="xs" c="dimmed">
              j
            </Text>
            <Text fw={700} size="lg" c="red.8">
              {counts.thin_j}
            </Text>
          </Paper>
        </SimpleGrid>
        <Stack gap={6}>
          {counts.uncertain > 0 && (
            <MetricRow label="неопределённых" value={counts.uncertain} />
          )}
          {counts.false_positive > 0 && (
            <MetricRow label="ложных" value={counts.false_positive} />
          )}
          <Divider my={4} />
          <MetricRow label="Рядовые %" value={`${metrics.ordinary_percent.toFixed(1)}%`} />
          <MetricRow label="Тонкие %" value={`${metrics.thin_percent.toFixed(1)}%`} />
        </Stack>
      </Paper>
    </Stack>
  );
}
