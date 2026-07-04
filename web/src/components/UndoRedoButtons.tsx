import { Group, ActionIcon, Tooltip } from "@mantine/core";
import { IconArrowBackUp, IconArrowForwardUp } from "@tabler/icons-react";

interface Props {
  canUndo: boolean;
  canRedo: boolean;
  onUndo: () => void;
  onRedo: () => void;
}

export default function UndoRedoButtons({ canUndo, canRedo, onUndo, onRedo }: Props) {
  return (
    <Group gap={4}>
      <Tooltip label="Отменить (Ctrl+Z)" withArrow>
        <ActionIcon variant="light" color="nornickel" radius="xl" disabled={!canUndo} onClick={onUndo}>
          <IconArrowBackUp size={16} />
        </ActionIcon>
      </Tooltip>
      <Tooltip label="Повторить (Ctrl+Shift+Z)" withArrow>
        <ActionIcon variant="light" color="nornickel" radius="xl" disabled={!canRedo} onClick={onRedo}>
          <IconArrowForwardUp size={16} />
        </ActionIcon>
      </Tooltip>
    </Group>
  );
}
