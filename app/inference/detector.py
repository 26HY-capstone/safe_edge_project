from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import isfinite

import numpy as np

BBox = tuple[float, float, float, float]
Point = tuple[float, float]


@dataclass(frozen=True, slots=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: BBox

    def __post_init__(self) -> None:
        class_id = int(self.class_id)
        class_name = str(self.class_name).strip()
        confidence = float(self.confidence)
        bbox = tuple(float(value) for value in self.bbox)

        if class_id < 0:
            raise ValueError("class_id must be non-negative")
        if not class_name:
            raise ValueError("class_name must not be empty")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if len(bbox) != 4 or not all(isfinite(value) for value in bbox):
            raise ValueError("bbox must contain four finite coordinates")

        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            raise ValueError("bbox must have positive width and height")

        object.__setattr__(self, "class_id", class_id)
        object.__setattr__(self, "class_name", class_name)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "bbox", bbox)

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def center(self) -> Point:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def bottom_center(self) -> Point:
        x1, _, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, y2)


class Detector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[Detection]:
        raise NotImplementedError

    @staticmethod
    def validate_frame(frame: np.ndarray) -> None:
        if not isinstance(frame, np.ndarray):
            raise TypeError("frame must be a numpy.ndarray")
        if frame.size == 0 or frame.ndim not in (2, 3):
            raise ValueError("frame must be a non-empty image")

    def warmup(self) -> None:
        return None

    def close(self) -> None:
        return None

    def __enter__(self) -> Detector:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()
