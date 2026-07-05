"""
Splitting large images into tiles for piece-by-piece processing.

Segmentation/detection models are trained at the scale of individual
frames — feeding them the whole panorama after downscaling loses fine
detail (talc, grains collapse to sub-pixel size). The panorama is cut into
tiles of bounded size, each processed independently at full resolution,
and the results are stitched back together by offset (x, y).

Each tile is taken with a context field (margin) around its "own" area
(core) — if cut without this margin, the model can't see neighboring
pixels at the tile border, and predictions from two adjacent tiles
disagree at the seam (visible seams along the tile grid). Only the core
goes into the final mask/object list — the margin exists solely to give
the model context and isn't written to the result.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
from numpy.typing import NDArray


@dataclass
class Tile:
    """
    A tile with a context margin.

    :param core_x, core_y: top-left corner of the core region in the original image
    :param core_w, core_h: size of the core region (what's actually written to the result)
    :param offset_x, offset_y: offset of the core within image (due to the margin)
    :param image: tile pixels (core + margin), clipped to the image boundaries
    """

    core_x: int
    core_y: int
    core_w: int
    core_h: int
    offset_x: int
    offset_y: int
    image: NDArray[np.uint8]

    def crop_to_core(self, array: NDArray) -> NDArray:
        """Crops an array (in image coordinates) to the tile's core region."""
        return array[
            self.offset_y : self.offset_y + self.core_h,
            self.offset_x : self.offset_x + self.core_w,
        ]


def iter_tiles(image_rgb: NDArray[np.uint8], tile_size: int, margin: int = 0) -> Iterator[Tile]:
    """Iterates tiles with a core size of tile_size x tile_size surrounded by a margin."""
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
