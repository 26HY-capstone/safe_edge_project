"""experiments/sahi 스크립트들이 공유하는 I/O, 통계, Detector 생성 유틸리티.

app/ 서비스 코드(ONNXYOLODetector/Detection/EXPECTED_PROTOTYPE_CLASSES)를 그대로
재사용하고, 여기서는 "임의의 .onnx 경로를 CLI 인자만으로 바로 돌릴 수 있게 감싸는"
실험 전용 편의 함수와, 모든 실험(run_standard/run_sahi/run_hybrid)이 동일하게 쓰는
predictions.json/summary.json 저장 형식만 둔다.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from app.inference.detector import Detection
from app.inference.model_manager import EXPECTED_PROTOTYPE_CLASSES
from app.inference.onnx_backend import ONNXDetectorConfig, ONNXYOLODetector

logger = logging.getLogger(__name__)

# YOLO 데이터셋에서 흔히 쓰이는 이미지 확장자만 지원한다.
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def list_image_paths(image_dir: Path) -> list[Path]:
    """image_dir 바로 아래의 이미지 파일을 이름 순으로 정렬해 반환한다.

    실행마다 같은 순서로 처리돼야 latency 통계와 진행 상황 로그를 비교하기 쉽다.
    """
    if not image_dir.is_dir():
        raise FileNotFoundError(f"image directory not found: {image_dir}")

    paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise ValueError(
            f"no supported images found in {image_dir} (extensions: {IMAGE_EXTENSIONS})"
        )
    return paths


def read_image_or_none(path: Path) -> np.ndarray | None:
    """OpenCV로 이미지를 읽는다. 손상된 파일 하나 때문에 전체 실행이 멈추지 않도록
    실패하면 경고만 남기고 None을 반환한다."""
    image = cv2.imread(str(path))
    if image is None:
        logger.warning("failed to read image, skipping: %s", path)
    return image


def build_onnx_detector(
    model_path: Path,
    input_size: int,
    confidence_threshold: float,
    iou_threshold: float,
    device: str = "cpu",
    allow_cpu_fallback: bool = True,
    classes: dict[int, str] | None = None,
) -> ONNXYOLODetector:
    """기존 ONNXYOLODetector를 CLI 인자만으로 생성한다.

    config/model.yaml에 의존하지 않고 임의의 .onnx 경로(640/960/v2 모델 등)를 받을 수
    있게 해서, 같은 runner 코드로 A/B/E 또는 C/F, D/G를 모두 실행할 수 있게 한다.
    preprocessing/decode/NMS는 app/inference/onnx_backend.py의 기존 구현을 그대로 쓴다.
    """
    if not model_path.is_file():
        raise FileNotFoundError(f"model file not found: {model_path}")
    if model_path.stat().st_size == 0:
        raise ValueError(f"model file is empty: {model_path}")

    config = ONNXDetectorConfig(
        model_path=model_path,
        device=device,
        allow_cpu_fallback=allow_cpu_fallback,
        input_size=input_size,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        classes=classes or dict(EXPECTED_PROTOTYPE_CLASSES),
    )
    return ONNXYOLODetector(config)


def detection_to_record(detection: Detection) -> dict[str, Any]:
    """Detection을 predictions.json에 저장할 JSON 직렬화 가능한 dict로 바꾼다."""
    return {
        "class_id": detection.class_id,
        "class_name": detection.class_name,
        "confidence": detection.confidence,
        "bbox_xyxy": list(detection.bbox),
    }


def build_image_record(
    image_name: str,
    width: int,
    height: int,
    detections: Sequence[Detection],
    latency_ms: float,
) -> dict[str, Any]:
    """한 이미지의 예측 결과를 A/B/C/D/E/F/G가 공통으로 쓰는 JSON 스키마(dict)로 만든다.

    실험 종류와 무관하게 evaluate.py/compare.py가 하나의 코드로 읽을 수 있어야 하므로,
    이 스키마는 모든 run_*.py에서 동일하게 사용한다.
    """
    return {
        "image": image_name,
        "width": width,
        "height": height,
        "detections": [detection_to_record(detection) for detection in detections],
        "latency_ms": latency_ms,
    }


def compute_latency_stats(latencies_ms: Sequence[float]) -> dict[str, float]:
    """평균/p50/p95 latency(ms)를 계산한다. 빈 목록이면 모두 0.0으로 반환한다."""
    if not latencies_ms:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}

    values = np.asarray(latencies_ms, dtype=np.float64)
    return {
        "mean_ms": float(np.mean(values)),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
    }


def save_predictions(records: Sequence[dict[str, Any]], output_dir: Path) -> Path:
    """predictions.json을 output_dir에 저장하고 저장된 경로를 반환한다."""
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.json"
    with predictions_path.open("w", encoding="utf-8") as predictions_file:
        json.dump(list(records), predictions_file, ensure_ascii=False, indent=2)
    return predictions_path


def save_summary(
    output_dir: Path,
    image_count: int,
    latency_stats: dict[str, float],
    extra: dict[str, Any] | None = None,
) -> Path:
    """summary.json(처리한 이미지 수, latency 통계, 실험 메타정보)을 저장한다."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"image_count": image_count, "latency": latency_stats}
    if extra:
        summary.update(extra)

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, ensure_ascii=False, indent=2)
    return summary_path


def load_predictions(predictions_path: Path) -> list[dict[str, Any]]:
    """predictions.json을 읽어 이미지별 record 목록으로 반환한다."""
    if not predictions_path.is_file():
        raise FileNotFoundError(f"predictions file not found: {predictions_path}")

    with predictions_path.open("r", encoding="utf-8") as predictions_file:
        records = json.load(predictions_file)
    if not isinstance(records, list):
        raise ValueError(f"predictions file must contain a JSON array: {predictions_path}")
    return records


def validate_slice_size(value: int, name: str) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}")
    return value


def validate_overlap_ratio(value: float, name: str) -> float:
    if not 0.0 <= value < 1.0:
        raise ValueError(f"{name} must satisfy 0 <= value < 1, got {value}")
    return value


@dataclass(frozen=True, slots=True)
class SliceSettings:
    slice_width: int
    slice_height: int
    overlap_width_ratio: float
    overlap_height_ratio: float


def load_slice_config_defaults(config_path: Path | None) -> dict[str, Any]:
    """config.yaml의 slice 섹션을 읽어 CLI 기본값 fallback으로 쓸 dict를 만든다.

    CLI에서 값을 직접 지정하면 여기서 읽은 값보다 항상 우선한다(아래 resolve_slice_settings).
    """
    if config_path is None:
        return {}
    if not config_path.is_file():
        raise FileNotFoundError(f"config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    slice_config = config.get("slice", {}) if isinstance(config, dict) else {}
    return dict(slice_config) if isinstance(slice_config, dict) else {}


def resolve_slice_settings(
    slice_size: int | None,
    slice_width: int | None,
    slice_height: int | None,
    overlap: float | None,
    overlap_width: float | None,
    overlap_height: float | None,
    config_defaults: dict[str, Any],
) -> SliceSettings:
    """slice 관련 CLI 인자들을 하나의 SliceSettings로 정리한다.

    우선순위: 축별 CLI 인자 > 공통 CLI 인자(--slice-size/--overlap) >
    config.yaml의 slice 섹션 > 하드코딩 기본값(640, 0.2).
    """
    default_width = int(config_defaults.get("width", 640))
    default_height = int(config_defaults.get("height", 640))
    default_overlap_width = float(config_defaults.get("overlap_width_ratio", 0.2))
    default_overlap_height = float(config_defaults.get("overlap_height_ratio", 0.2))

    width = slice_width if slice_width is not None else slice_size
    width = width if width is not None else default_width
    height = slice_height if slice_height is not None else slice_size
    height = height if height is not None else default_height

    overlap_w = overlap_width if overlap_width is not None else overlap
    overlap_w = overlap_w if overlap_w is not None else default_overlap_width
    overlap_h = overlap_height if overlap_height is not None else overlap
    overlap_h = overlap_h if overlap_h is not None else default_overlap_height

    return SliceSettings(
        slice_width=validate_slice_size(width, "slice width"),
        slice_height=validate_slice_size(height, "slice height"),
        overlap_width_ratio=validate_overlap_ratio(overlap_w, "overlap width ratio"),
        overlap_height_ratio=validate_overlap_ratio(overlap_h, "overlap height ratio"),
    )


def draw_detections(image: np.ndarray, detections: Sequence[Detection]) -> np.ndarray:
    """--save-visualizations 전용: bbox/class/confidence를 그린 복사본을 반환한다."""
    visualized = image.copy()
    for detection in detections:
        x1, y1, x2, y2 = (int(round(value)) for value in detection.bbox)
        cv2.rectangle(visualized, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{detection.class_name} {detection.confidence:.2f}"
        cv2.putText(
            visualized,
            label,
            (x1, max(y1 - 8, 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
    return visualized
