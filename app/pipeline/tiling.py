"""
Разбиение больших изображений на тайлы для поштучной обработки.

Модели сегментации/детекции обучены на масштабе отдельных кадров —
скармливать им панораму целиком после downscale теряет мелкие детали
(тальк, зёрна схлопываются до суб-пикселя). Панорама режется на тайлы
ограниченного размера, каждый обрабатывается независимо в исходном
разрешении, результаты сшиваются обратно по смещению (x, y).

Каждый тайл берётся с контекстным полем (margin) вокруг «своей» области
(core) — если резать без запаса, модель на границе тайла не видит соседних
пикселей, и предсказания двух смежных тайлов на стыке расходятся (видимые
швы по сетке тайлов). В итоговую маску/список объектов идёт только core —
margin нужен исключительно моделью для контекста и в результат не пишется.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
from numpy.typing import NDArray


@dataclass
class Tile:
    """
    Тайл с контекстным полем.

    :param core_x, core_y: левый верхний угол core-региона в исходном изображении
    :param core_w, core_h: размер core-региона (то, что реально пишется в результат)
    :param offset_x, offset_y: смещение core внутри image (из-за поля margin)
    :param image: пиксели тайла (core + margin), обрезанные по границам изображения
    """

    core_x: int
    core_y: int
    core_w: int
    core_h: int
    offset_x: int
    offset_y: int
    image: NDArray[np.uint8]

    def crop_to_core(self, array: NDArray) -> NDArray:
        """Обрезает массив (в системе координат image) до core-региона тайла."""
        return array[
            self.offset_y : self.offset_y + self.core_h,
            self.offset_x : self.offset_x + self.core_w,
        ]


def iter_tiles(image_rgb: NDArray[np.uint8], tile_size: int, margin: int = 0) -> Iterator[Tile]:
    """Итерирует тайлы core-размером tile_size x tile_size с полем margin вокруг."""
    height, width = image_rgb.shape[:2]
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            core_w = min(tile_size, width - x)
            core_h = min(tile_size, height - y)
            rx0 = max(0, x - margin)
            ry0 = max(0, y - margin)
            rx1 = min(width, x + core_w + margin)
            ry1 = min(height, y + core_h + margin)
            tile_rgb = image_rgb[ry0:ry1, rx0:rx1]
            yield Tile(
                core_x=x,
                core_y=y,
                core_w=core_w,
                core_h=core_h,
                offset_x=x - rx0,
                offset_y=y - ry0,
                image=np.ascontiguousarray(tile_rgb),
            )
