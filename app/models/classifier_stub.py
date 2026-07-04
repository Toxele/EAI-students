"""
STUB: Классификатор сорта руды для детальных OM-снимков.

Сейчас: nearest neighbor по average hash к эталонам из data/ch1.
Потом: CNN / transfer learning.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from app.config import CH1_DATA_DIR
from app.pipeline.loader import imread_unicode

FOLDER_TO_SORT = {
    "Оталькованные руды": "otalkovannaya",
    "Рядовые руды": "ryadovaya",
    "Труднообогатимые руды": "trudnoobogatimaya",
}

SORT_LABELS_RU = {
    "otalkovannaya": "оталькованная",
    "ryadovaya": "рядовая",
    "trudnoobogatimaya": "труднообогатимая",
}


@dataclass
class ReferenceImage:
    path: Path
    sort_code: str
    hash_bits: NDArray[np.uint8]


@dataclass
class ClassificationResult:
    sort_code: str
    sort_label_ru: str
    best_distance: int
    matched_reference: str


class ClassifierStub:
    """STUB: сравнивает вход с фото из data/ch1 по average hash."""

    def __init__(self, data_dir: Path = CH1_DATA_DIR) -> None:
        self.data_dir = data_dir
        self.references: list[ReferenceImage] = []
        self._build_index()

    def _build_index(self) -> None:
        """Строит индекс эталонов при создании объекта."""
        if not self.data_dir.exists():
            return

        for sort_folder in self.data_dir.iterdir():
            if not sort_folder.is_dir():
                continue
            if sort_folder.name not in FOLDER_TO_SORT:
                continue

            sort_code = FOLDER_TO_SORT[sort_folder.name]

            for img_path in sort_folder.glob("*"):
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                if not img_path.is_file():
                    continue

                hash_bits = self._average_hash(img_path)
                if hash_bits is not None:
                    self.references.append(
                        ReferenceImage(path=img_path, sort_code=sort_code, hash_bits=hash_bits)
                    )

    def _average_hash(self, image_path: Path, hash_size: int = 8) -> NDArray[np.uint8] | None:
        # imread_unicode — кириллица в путях на Windows
        img = imread_unicode(image_path)
        if img is None:
            return None
        resized = cv2.resize(img, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        return (gray > gray.mean()).astype(np.uint8).flatten()

    def _hash_from_array(self, image_rgb: NDArray[np.uint8], hash_size: int = 8) -> NDArray[np.uint8]:
        resized = cv2.resize(image_rgb, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        return (gray > gray.mean()).astype(np.uint8).flatten()

    @staticmethod
    def _hamming(a: NDArray[np.uint8], b: NDArray[np.uint8]) -> int:
        return int(np.sum(a != b))

    def predict(self, image_rgb: NDArray[np.uint8]) -> ClassificationResult:
        """Возвращает сорт ближайшего эталона из data/."""
        if not self.references:
            return ClassificationResult(
                sort_code="unknown",
                sort_label_ru="неизвестно (нет эталонов)",
                best_distance=999,
                matched_reference="",
            )

        query_hash = self._hash_from_array(image_rgb)
        best_ref = self.references[0]
        best_dist = self._hamming(query_hash, best_ref.hash_bits)

        for ref in self.references[1:]:
            dist = self._hamming(query_hash, ref.hash_bits)
            if dist < best_dist:
                best_dist = dist
                best_ref = ref

        return ClassificationResult(
            sort_code=best_ref.sort_code,
            sort_label_ru=SORT_LABELS_RU[best_ref.sort_code],
            best_distance=best_dist,
            matched_reference=best_ref.path.name,
        )
