"""모델 추론 방식 선택, 설정 검증, warm-up을 관리하는 모듈."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

from app.inference.detector import Detector
from app.inference.onnx_backend import ONNXYOLODetector, load_onnx_detector_config
from app.inference.pytorch_backend import (
    PyTorchYOLODetector,
    load_pytorch_detector_config,
)
from app.video.video_source import PROJECT_ROOT

EXPECTED_PROTOTYPE_CLASSES = {
    0: "person",
    1: "head",
    2: "gloves",
    3: "helmet",
    4: "body",
    5: "safety_vest",
    6: "forklift",
    7: "robot_arm",
    8: "conveyor",
    9: "harness_body",
}

BACKEND_EXTENSIONS = {
    "pytorch": ".pt",
    "onnx": ".onnx",
    "tensorrt": ".engine",
}


@dataclass(frozen=True, slots=True)
class ModelSelection:
    backend: str
    model_path: Path
    classes: dict[int, str]


class ModelManager:
    def __init__(
        self,
        config_path: str | Path = PROJECT_ROOT / "config" / "model.yaml",
    ) -> None:
        self.config_path = _resolve_project_path(config_path)

    def create_detector(self, warmup: bool = True) -> Detector:
        selection = self._load_and_validate_selection()

        if selection.backend == "pytorch":
            detector: Detector = PyTorchYOLODetector(
                load_pytorch_detector_config(self.config_path)
            )
        elif selection.backend == "onnx":
            detector = ONNXYOLODetector(
                load_onnx_detector_config(self.config_path)
            )
        else:
            raise NotImplementedError(
                "TensorRT backend is not implemented yet. "
                "Validate the ONNX pipeline before enabling .engine models."
            )

        if warmup:
            detector.warmup()
        return detector

    def _load_and_validate_selection(self) -> ModelSelection:
        with self.config_path.open("r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file) or {}

        if not isinstance(config, Mapping):
            raise ValueError("model config must be a mapping")
        detector_config = config.get("detector", {})
        if not isinstance(detector_config, Mapping):
            raise ValueError("detector config must be a mapping")

        backend = str(detector_config.get("backend", "")).strip().lower()
        if backend not in BACKEND_EXTENSIONS:
            raise ValueError(
                f"unsupported detector backend: {backend or '<empty>'}"
            )

        raw_model_path = str(detector_config.get("model_path", "")).strip()
        if not raw_model_path:
            raise ValueError("detector model_path must not be empty")
        model_path = _resolve_project_path(raw_model_path)
        expected_extension = BACKEND_EXTENSIONS[backend]
        if model_path.suffix.lower() != expected_extension:
            raise ValueError(
                "detector backend and model extension do not match: "
                f"backend={backend}, model_path={model_path}"
            )
        if not model_path.is_file():
            raise FileNotFoundError(f"detector model not found: {model_path}")
        if model_path.stat().st_size == 0:
            raise ValueError(f"detector model is empty: {model_path}")

        raw_classes = detector_config.get("classes", {})
        if not isinstance(raw_classes, Mapping):
            raise ValueError("detector classes must be a mapping")
        classes = {
            int(class_id): str(class_name)
            for class_id, class_name in raw_classes.items()
        }
        if classes != EXPECTED_PROTOTYPE_CLASSES:
            raise ValueError(
                "prototype detector classes must match the integrated 10-class model: "
                f"expected={EXPECTED_PROTOTYPE_CLASSES}, actual={classes}"
            )

        return ModelSelection(
            backend=backend,
            model_path=model_path,
            classes=classes,
        )


def create_detector_from_model_config(
    config_path: str | Path = PROJECT_ROOT / "config" / "model.yaml",
    warmup: bool = True,
) -> Detector:
    return ModelManager(config_path).create_detector(warmup=warmup)


def _resolve_project_path(path: str | Path) -> Path:
    resolved_path = Path(path).expanduser()
    if resolved_path.is_absolute():
        return resolved_path
    if resolved_path.exists():
        return resolved_path
    return PROJECT_ROOT / resolved_path
