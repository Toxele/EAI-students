import {
  Paper,
  Title,
  Text,
  Stack,
  Group,
  Badge,
  Progress,
  Divider,
  SimpleGrid,
} from "@mantine/core";
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

function PercentBar({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <Stack gap={4}>
      <Group justify="space-between">
        <Text size="sm" c="dimmed">
          {label}
        </Text>
        <Text size="sm" fw={600}>
          {value.toFixed(1)}%
        </Text>
      </Group>
      <Progress value={value} color={color} size="sm" />
    </Stack>
  );
}

export default function MetricsPanel({ result }: Props) {
  const { counts, metrics } = result;
  const talcAvailable = metrics.talc_available && metrics.talc_percent != null;

  const sortColor =
    result.sort_code === "talc" ? "blue" : result.sort_code === "ordinary" ? "green" : "orange";

  return (
    <Stack gap="md">
      <Paper p="md" radius="xl" shadow="xs" withBorder>
        <Title order={5} tt="uppercase" fz="sm" mb="sm" className="nn-section-title">
          Итоговый сорт
        </Title>
        <Badge size="lg" variant="light" color={sortColor} mb="xs" radius="sm">
          {result.sort_label_ru}
        </Badge>
        <Text size="sm" c="dimmed" lh={1.5} className="nn-quote-bar">
          {result.conclusion.replace(/\*\*/g, "")}
        </Text>
      </Paper>

      <Paper p="md" radius="xl" shadow="xs" withBorder>
        <Title order={5} tt="uppercase" fz="sm" mb="sm" className="nn-section-title">
          Тальк
        </Title>
        {talcAvailable ? (
          <PercentBar label="Доля талька" value={metrics.talc_percent as number} color="blue" />
        ) : (
          <Text size="sm" c="dimmed">
            н/д
          </Text>
        )}
      </Paper>

      {result.mode !== "detail" && (
        <Paper p="md" radius="xl" shadow="xs" withBorder>
          <Title order={5} tt="uppercase" fz="sm" mb="sm" className="nn-section-title">
            Срастания
          </Title>
          <SimpleGrid cols={3} mb="sm">
            <Paper p="xs" radius="sm" bg="gray.0" ta="center">
              <Text size="xs" c="dimmed">
                Всего
              </Text>
              <Text fw={700} size="lg">
                {counts.total_k}
              </Text>
            </Paper>
            <Paper p="xs" radius="sm" bg="green.0" ta="center">
              <Text size="xs" c="dimmed">
                Рядовые
              </Text>
              <Text fw={700} size="lg" c="green.8">
                {counts.ordinary_l}
              </Text>
            </Paper>
            <Paper p="xs" radius="sm" bg="red.0" ta="center">
              <Text size="xs" c="dimmed">
                Тонкие
              </Text>
              <Text fw={700} size="lg" c="red.8">
                {counts.thin_j}
              </Text>
            </Paper>
          </SimpleGrid>
          <Stack gap="sm">
            {counts.uncertain > 0 && (
              <MetricRow label="неопределённых" value={counts.uncertain} />
            )}
            {counts.false_positive > 0 && (
              <MetricRow label="ложных" value={counts.false_positive} />
            )}
            <Divider my={4} />
            <PercentBar label="Рядовые %" value={metrics.ordinary_percent} color="green" />
            <PercentBar label="Тонкие %" value={metrics.thin_percent} color="red" />
          </Stack>
        </Paper>
      )}
    </Stack>
  );
}
