"""PyTorch YOLO 모델 결과를 공통 Detection 형식으로 변환하는 모듈."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from app.inference.detector import BBox, Detection, Detector
from app.video.video_source import PROJECT_ROOT


@dataclass(frozen=True, slots=True)
class PyTorchDetectorConfig:
    model_path: Path
    device: str = "cpu"
    input_size: int = 640
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.45
    classes: dict[int, str] | None = None

    def __post_init__(self) -> None:
        """모델 실행 설정의 유효 범위를 검증한다."""
        if self.input_size <= 0:
            raise ValueError("input_size must be positive")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if not 0.0 <= self.iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between 0 and 1")


class PyTorchYOLODetector(Detector):
    def __init__(
        self,
        config: PyTorchDetectorConfig,
        model: Any | None = None,
    ) -> None:
        """Ultralytics YOLO 모델을 Detector 인터페이스로 감싼다."""
        self.config = config
        self._model = model or self._load_model(config.model_path)
        self._class_names = config.classes or {}

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """단일 프레임을 YOLO로 추론하고 Detection 목록으로 변환한다."""
        self.validate_frame(frame)
        results: Iterable[object] | None = self._model.predict(
            source=frame,
            imgsz=self.config.input_size,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            device=self.config.device,
            verbose=False,
        )
        return self._results_to_detections(results)

    def warmup(self) -> None:
        """초기 추론 지연을 줄이기 위해 더미 프레임으로 모델을 한 번 실행한다."""
        dummy_frame = np.zeros(
            (self.config.input_size, self.config.input_size, 3),
            dtype=np.uint8,
        )
        self.detect(dummy_frame)

    def _load_model(self, model_path: Path) -> Any:
        """모델 파일을 확인한 뒤 Ultralytics YOLO 객체를 생성한다."""
        resolved_model_path = _resolve_project_path(model_path)
        if not resolved_model_path.is_file():
            raise FileNotFoundError(f"Model file not found: {resolved_model_path}")

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "PyTorchYOLODetector requires ultralytics. "
                "Install project dependencies from requirements.txt."
            ) from exc

        return YOLO(str(resolved_model_path))

    def _results_to_detections(
        self,
        results: Iterable[object] | None,
    ) -> list[Detection]:
        """Ultralytics Results 목록을 프로젝트 표준 Detection 목록으로 변환한다."""
        detections: list[Detection] = []

        for result in results if results is not None else []:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue

            xyxy_values = _to_numpy(getattr(boxes, "xyxy", []))
            confidence_values = _to_numpy(getattr(boxes, "conf", []))
            class_values = _to_numpy(getattr(boxes, "cls", []))

            for bbox, confidence, class_id in zip(
                xyxy_values,
                confidence_values,
                class_values,
            ):
                confidence_float = float(confidence)
                if confidence_float < self.config.confidence_threshold:
                    continue

                class_id_int = int(class_id)
                detections.append(
                    Detection(
                        class_id=class_id_int,
                        class_name=self._class_name(class_id_int, result),
                        confidence=confidence_float,
                        bbox=_bbox_from_xyxy(bbox),
                    )
                )

        return detections

    def _class_name(self, class_id: int, result: object) -> str:
        """설정 class mapping을 우선 사용하고 없으면 YOLO 결과의 names를 사용한다."""
        if class_id in self._class_names:
            return self._class_names[class_id]

        result_names = getattr(result, "names", None)
        if isinstance(result_names, Mapping) and class_id in result_names:
            return str(result_names[class_id])

        model_names = getattr(self._model, "names", None)
        if isinstance(model_names, Mapping) and class_id in model_names:
            return str(model_names[class_id])

        return str(class_id)


def create_pytorch_detector_from_model_config(
    config_path: str | Path = PROJECT_ROOT / "config" / "model.yaml",
) -> PyTorchYOLODetector:
    """model.yaml의 detector 설정으로 PyTorch YOLO detector를 생성한다."""
    config = load_pytorch_detector_config(config_path)
    return PyTorchYOLODetector(config)


def load_pytorch_detector_config(
    config_path: str | Path = PROJECT_ROOT / "config" / "model.yaml",
) -> PyTorchDetectorConfig:
    """model.yaml에서 PyTorch detector 설정을 읽는다."""
    resolved_config_path = _resolve_project_path(config_path)
    with resolved_config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}

    if not isinstance(config, Mapping):
        raise ValueError("model config must be a mapping")

    detector_config = config.get("detector", {})
    if not isinstance(detector_config, Mapping):
        raise ValueError("detector config must be a mapping")

    classes = detector_config.get("classes", {})
    if not isinstance(classes, Mapping):
        raise ValueError("detector classes must be a mapping")

    return PyTorchDetectorConfig(
        model_path=Path(str(detector_config.get("model_path", "models/best.pt"))),
        device=str(detector_config.get("device", "cpu")),
        input_size=int(detector_config.get("input_size", 640)),
        confidence_threshold=float(
            detector_config.get("confidence_threshold", 0.35)
        ),
        iou_threshold=float(detector_config.get("iou_threshold", 0.45)),
        classes={int(class_id): str(name) for class_id, name in classes.items()},
    )


def _to_numpy(value: object) -> np.ndarray:
    """torch Tensor와 numpy 배열을 후처리 가능한 numpy 배열로 변환한다."""
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()

    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()

    numpy_method = getattr(value, "numpy", None)
    if callable(numpy_method):
        return np.asarray(numpy_method())

    return np.asarray(value)


def _bbox_from_xyxy(values: Sequence[object] | np.ndarray) -> BBox:
    """YOLO xyxy 배열을 Detection bbox tuple로 변환한다."""
    coordinates = np.asarray(values, dtype=float).reshape(-1)
    if coordinates.size < 4:
        raise ValueError("bbox must contain at least four coordinates")

    x1, y1, x2, y2 = coordinates[:4]
    return (float(x1), float(y1), float(x2), float(y2))


def _resolve_project_path(path: str | Path) -> Path:
    """상대경로를 프로젝트 루트 기준으로 해석한다."""
    resolved_path = Path(path).expanduser()
    if resolved_path.is_absolute():
        return resolved_path
    if resolved_path.exists():
        return resolved_path
    return PROJECT_ROOT / resolved_path
