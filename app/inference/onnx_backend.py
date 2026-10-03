"""ONNX YOLO 모델 결과를 공통 Detection 형식으로 변환하는 모듈."""

from __future__ import annotations

from ast import literal_eval
from collections.abc import Mapping
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from app.inference.detector import Detection, Detector
from app.video.video_source import PROJECT_ROOT

logger = logging.getLogger(__name__)

ProviderSpec = str | tuple[str, dict[str, str]]


@dataclass(frozen=True, slots=True)
class ONNXDetectorConfig:
    model_path: Path
    device: str = "cpu"
    allow_cpu_fallback: bool = True
    input_size: int = 640
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    classes: dict[int, str] | None = None

    def __post_init__(self) -> None:
        if self.model_path.suffix.lower() != ".onnx":
            raise ValueError("ONNX model_path must use the .onnx extension")
        if self.input_size <= 0:
            raise ValueError("input_size must be positive")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if not 0.0 <= self.iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between 0 and 1")
        if not self.classes:
            raise ValueError("classes must not be empty")


class ONNXYOLODetector(Detector):
    def __init__(
        self,
        config: ONNXDetectorConfig,
        session: Any | None = None,
    ) -> None:
        self.config = config
        self._session = session or self._load_session(config.model_path)
        self._input = self._single_input()
        self._output = self._single_output()
        self._validate_model_contract()

    def detect(self, frame: np.ndarray) -> list[Detection]:
        self.validate_frame(frame)
        input_tensor, scale, pad_x, pad_y = self._preprocess(frame)
        outputs = self._session.run(
            [self._output.name],
            {self._input.name: input_tensor},
        )
        if not outputs:
            return []
        return self._decode(
            output=np.asarray(outputs[0]),
            frame_shape=frame.shape,
            scale=scale,
            pad_x=pad_x,
            pad_y=pad_y,
        )

    def warmup(self) -> None:
        dummy_frame = np.zeros(
            (self.config.input_size, self.config.input_size, 3),
            dtype=np.uint8,
        )
        self.detect(dummy_frame)

    def _load_session(self, model_path: Path) -> Any:
        resolved_model_path = _resolve_project_path(model_path)
        if not resolved_model_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {resolved_model_path}")
        if resolved_model_path.stat().st_size == 0:
            raise ValueError(f"ONNX model is empty: {resolved_model_path}")

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "ONNXYOLODetector requires onnxruntime. "
                "Install project dependencies from requirements.txt."
            ) from exc

        available_providers = ort.get_available_providers()
        providers = _providers_for_device(
            device=self.config.device,
            available_providers=available_providers,
            allow_cpu_fallback=self.config.allow_cpu_fallback,
        )
        session = ort.InferenceSession(
            str(resolved_model_path),
            providers=providers,
        )
        logger.info(
            "ONNX Runtime execution providers: requested=%s active=%s",
            self.config.device,
            session.get_providers(),
        )
        return session

    def _single_input(self) -> Any:
        inputs = self._session.get_inputs()
        if len(inputs) != 1:
            raise ValueError(f"ONNX detector must have one input, got {len(inputs)}")
        return inputs[0]

    def _single_output(self) -> Any:
        outputs = self._session.get_outputs()
        if len(outputs) != 1:
            raise ValueError(f"ONNX detector must have one output, got {len(outputs)}")
        return outputs[0]

    def _validate_model_contract(self) -> None:
        expected_channel_count = 4 + len(self.config.classes or {})
        input_shape = tuple(self._input.shape)
        output_shape = tuple(self._output.shape)

        if len(input_shape) != 4:
            raise ValueError(f"ONNX input must be NCHW, got {input_shape}")
        if _is_static_dimension(input_shape[1]) and int(input_shape[1]) != 3:
            raise ValueError(f"ONNX input channel count must be 3, got {input_shape[1]}")
        for dimension in input_shape[2:4]:
            if _is_static_dimension(dimension) and int(dimension) != self.config.input_size:
                raise ValueError(
                    "ONNX input size does not match config: "
                    f"model={input_shape}, config={self.config.input_size}"
                )

        if len(output_shape) != 3:
            raise ValueError(f"ONNX output must be rank 3, got {output_shape}")
        static_output_dimensions = {
            int(dimension)
            for dimension in output_shape[1:]
            if _is_static_dimension(dimension)
        }
        if (
            static_output_dimensions
            and expected_channel_count not in static_output_dimensions
        ):
            raise ValueError(
                "ONNX output class count does not match config: "
                f"output={output_shape}, classes={len(self.config.classes or {})}"
            )

        metadata = self._session.get_modelmeta().custom_metadata_map or {}
        model_classes = _parse_class_names(metadata.get("names"))
        if model_classes and model_classes != self.config.classes:
            raise ValueError(
                "ONNX model class names do not match config/model.yaml: "
                f"model={model_classes}, config={self.config.classes}"
            )

    def _preprocess(
        self,
        frame: np.ndarray,
    ) -> tuple[np.ndarray, float, float, float]:
        if frame.ndim == 2:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.ndim == 3 and frame.shape[2] == 3:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif frame.ndim == 3 and frame.shape[2] == 4:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        else:
            raise ValueError("frame must have 1, 3, or 4 image channels")

        frame_height, frame_width = rgb_frame.shape[:2]
        scale = min(
            self.config.input_size / frame_width,
            self.config.input_size / frame_height,
        )
        resized_width = max(1, round(frame_width * scale))
        resized_height = max(1, round(frame_height * scale))
        resized = cv2.resize(
            rgb_frame,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )

        pad_x = (self.config.input_size - resized_width) / 2.0
        pad_y = (self.config.input_size - resized_height) / 2.0
        left = int(round(pad_x - 0.1))
        top = int(round(pad_y - 0.1))
        model_input = np.full(
            (self.config.input_size, self.config.input_size, 3),
            114,
            dtype=np.uint8,
        )
        model_input[top : top + resized_height, left : left + resized_width] = resized

        input_tensor = np.ascontiguousarray(
            model_input.transpose(2, 0, 1)[None],
            dtype=np.float32,
        )
        input_tensor /= 255.0
        return input_tensor, scale, float(left), float(top)

    def _decode(
        self,
        output: np.ndarray,
        frame_shape: tuple[int, ...],
        scale: float,
        pad_x: float,
        pad_y: float,
    ) -> list[Detection]:
        prediction = np.squeeze(output, axis=0) if output.ndim == 3 else output
        if prediction.ndim != 2:
            raise ValueError(f"unsupported ONNX output shape: {output.shape}")

        channel_count = 4 + len(self.config.classes or {})
        if prediction.shape[0] == channel_count:
            prediction = prediction.T
        elif prediction.shape[1] != channel_count:
            raise ValueError(f"unsupported ONNX output shape: {output.shape}")

        class_scores = prediction[:, 4:]
        class_ids = np.argmax(class_scores, axis=1)
        confidence_values = class_scores[
            np.arange(class_scores.shape[0]),
            class_ids,
        ]
        selected = confidence_values >= self.config.confidence_threshold
        if not np.any(selected):
            return []

        boxes_xywh = prediction[selected, :4].astype(np.float32, copy=False)
        confidence_values = confidence_values[selected].astype(np.float32, copy=False)
        class_ids = class_ids[selected]
        boxes_xyxy = _restore_original_xyxy(
            boxes_xywh=boxes_xywh,
            frame_width=frame_shape[1],
            frame_height=frame_shape[0],
            scale=scale,
            pad_x=pad_x,
            pad_y=pad_y,
        )
        keep_indices = _class_aware_nms(
            boxes=boxes_xyxy,
            scores=confidence_values,
            class_ids=class_ids,
            iou_threshold=self.config.iou_threshold,
        )

        detections: list[Detection] = []
        for index in keep_indices:
            x1, y1, x2, y2 = boxes_xyxy[index]
            if x2 <= x1 or y2 <= y1:
                continue
            class_id = int(class_ids[index])
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=(self.config.classes or {})[class_id],
                    confidence=float(confidence_values[index]),
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                )
            )
        return detections


def load_onnx_detector_config(
    config_path: str | Path = PROJECT_ROOT / "config" / "model.yaml",
) -> ONNXDetectorConfig:
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

    return ONNXDetectorConfig(
        model_path=Path(str(detector_config.get("model_path", ""))),
        device=str(detector_config.get("device", "cpu")),
        allow_cpu_fallback=_required_bool(
            detector_config,
            "allow_cpu_fallback",
            default=True,
        ),
        input_size=int(detector_config.get("input_size", 640)),
        confidence_threshold=float(
            detector_config.get("confidence_threshold", 0.25)
        ),
        iou_threshold=float(detector_config.get("iou_threshold", 0.45)),
        classes={int(class_id): str(name) for class_id, name in classes.items()},
    )


def _providers_for_device(
    device: str,
    available_providers: list[str],
    allow_cpu_fallback: bool = True,
) -> list[ProviderSpec]:
    normalized_device = device.strip().lower()
    if normalized_device == "cpu":
        return _cpu_provider(available_providers)

    if normalized_device.startswith("cuda"):
        if "CUDAExecutionProvider" in available_providers:
            device_id = _cuda_device_id(normalized_device)
            return [
                ("CUDAExecutionProvider", {"device_id": device_id}),
                *_cpu_provider(available_providers),
            ]
        return _fallback_or_raise(
            requested_provider="CUDAExecutionProvider",
            available_providers=available_providers,
            allow_cpu_fallback=allow_cpu_fallback,
            install_hint=(
                "install onnxruntime-gpu and verify CUDA/cuDNN compatibility"
            ),
        )

    if normalized_device == "coreml":
        if "CoreMLExecutionProvider" in available_providers:
            return [
                (
                    "CoreMLExecutionProvider",
                    {
                        "ModelFormat": "MLProgram",
                        "MLComputeUnits": "ALL",
                        "RequireStaticInputShapes": "1",
                    },
                ),
                *_cpu_provider(available_providers),
            ]
        return _fallback_or_raise(
            requested_provider="CoreMLExecutionProvider",
            available_providers=available_providers,
            allow_cpu_fallback=allow_cpu_fallback,
            install_hint=(
                "install the official macOS onnxruntime wheel with CoreML support"
            ),
        )

    raise ValueError(f"unsupported ONNX device: {device}")


def _cpu_provider(available_providers: list[str]) -> list[str]:
    if "CPUExecutionProvider" not in available_providers:
        raise RuntimeError("ONNX Runtime CPUExecutionProvider is unavailable")
    return ["CPUExecutionProvider"]


def _fallback_or_raise(
    requested_provider: str,
    available_providers: list[str],
    allow_cpu_fallback: bool,
    install_hint: str,
) -> list[str]:
    if not allow_cpu_fallback:
        raise RuntimeError(f"{requested_provider} is unavailable; {install_hint}")

    logger.warning(
        "%s is unavailable; falling back to CPUExecutionProvider",
        requested_provider,
    )
    return _cpu_provider(available_providers)


def _cuda_device_id(device: str) -> str:
    if ":" not in device:
        return "0"
    _, raw_device_id = device.split(":", maxsplit=1)
    if not raw_device_id.isdigit():
        raise ValueError(f"invalid CUDA device: {device}")
    return raw_device_id


def _required_bool(
    values: Mapping[str, object],
    key: str,
    default: bool,
) -> bool:
    value = values.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"detector {key} must be boolean")
    return value


def _restore_original_xyxy(
    boxes_xywh: np.ndarray,
    frame_width: int,
    frame_height: int,
    scale: float,
    pad_x: float,
    pad_y: float,
) -> np.ndarray:
    boxes = np.empty_like(boxes_xywh, dtype=np.float32)
    boxes[:, 0] = (boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2.0 - pad_x) / scale
    boxes[:, 1] = (boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2.0 - pad_y) / scale
    boxes[:, 2] = (boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2.0 - pad_x) / scale
    boxes[:, 3] = (boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2.0 - pad_y) / scale
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, frame_width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, frame_height)
    return boxes


def _class_aware_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float,
) -> list[int]:
    kept: list[int] = []
    for class_id in np.unique(class_ids):
        class_indices = np.flatnonzero(class_ids == class_id)
        order = class_indices[np.argsort(scores[class_indices])[::-1]]

        while order.size:
            current = int(order[0])
            kept.append(current)
            if order.size == 1:
                break

            remaining = order[1:]
            ious = _box_iou(boxes[current], boxes[remaining])
            order = remaining[ious <= iou_threshold]

    return sorted(kept, key=lambda index: float(scores[index]), reverse=True)


def _box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    intersection_x1 = np.maximum(box[0], boxes[:, 0])
    intersection_y1 = np.maximum(box[1], boxes[:, 1])
    intersection_x2 = np.minimum(box[2], boxes[:, 2])
    intersection_y2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0.0, intersection_x2 - intersection_x1) * np.maximum(
        0.0,
        intersection_y2 - intersection_y1,
    )
    box_area = max(0.0, float(box[2] - box[0])) * max(
        0.0,
        float(box[3] - box[1]),
    )
    boxes_area = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0.0,
        boxes[:, 3] - boxes[:, 1],
    )
    union = box_area + boxes_area - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0,
    )


def _parse_class_names(raw_names: object) -> dict[int, str]:
    if not isinstance(raw_names, str) or not raw_names.strip():
        return {}
    try:
        parsed_names = literal_eval(raw_names)
    except (SyntaxError, ValueError):
        return {}
    if not isinstance(parsed_names, Mapping):
        return {}
    return {int(class_id): str(name) for class_id, name in parsed_names.items()}


def _is_static_dimension(dimension: object) -> bool:
    return isinstance(dimension, int) and not isinstance(dimension, bool)


def _resolve_project_path(path: str | Path) -> Path:
    resolved_path = Path(path).expanduser()
    if resolved_path.is_absolute():
        return resolved_path
    if resolved_path.exists():
        return resolved_path
    return PROJECT_ROOT / resolved_path
