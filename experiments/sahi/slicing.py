"""SAHI-style overlapping slice 생성과 slice 좌표 <-> 원본 이미지 좌표 변환을 담당하는 모듈.

이미지를 자르고, 잘린 영역의 결과를 원본 좌표로 되돌리는 SAHI 고유의 geometry 로직만 담당.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from app.inference.detector import Detection, Detector
from app.inference.onnx_backend import _class_aware_nms


@dataclass(frozen=True, slots=True)
class SliceBox:
    """원본 이미지 기준 slice 영역. offset은 원본 이미지 좌상단 기준 픽셀 좌표다."""

    offset_x: int
    offset_y: int
    width: int
    height: int


def compute_slice_boxes(
    image_width: int,
    image_height: int,
    slice_width: int,
    slice_height: int,
    overlap_width_ratio: float,
    overlap_height_ratio: float,
) -> list[SliceBox]:
    """원본 이미지 전체를 빠짐없이 덮는 overlapping slice 목록을 만든다.

    각 축(x, y)을 독립적으로 계산한 뒤 cartesian product으로 slice grid를 만든다.
    마지막 slice는 (이미지 끝 - slice 크기)로 "뒤로 당겨" 고정 크기를 유지한다 —
    이렇게 하면 오른쪽/아래쪽 끝이 누락되지 않고, 모든 slice가 모델에 넣기 좋은
    동일한 크기를 유지한다(마지막 slice만 더 작아지는 방식은 쓰지 않음).
    이미지 한 축이 slice보다 작거나 같으면 그 축은 이미지 전체 크기의 slice 1개로 처리한다.
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"image size must be positive, got {image_width}x{image_height}")
    if slice_width <= 0 or slice_height <= 0:
        raise ValueError(f"slice size must be positive, got {slice_width}x{slice_height}")
    if not 0.0 <= overlap_width_ratio < 1.0:
        raise ValueError(f"overlap_width_ratio must satisfy 0<=v<1, got {overlap_width_ratio}")
    if not 0.0 <= overlap_height_ratio < 1.0:
        raise ValueError(f"overlap_height_ratio must satisfy 0<=v<1, got {overlap_height_ratio}")

    x_offsets = _axis_offsets(image_width, slice_width, overlap_width_ratio)
    y_offsets = _axis_offsets(image_height, slice_height, overlap_height_ratio)

    return [
        SliceBox(offset_x=x_offset, offset_y=y_offset, width=x_size, height=y_size)
        for y_offset, y_size in y_offsets
        for x_offset, x_size in x_offsets
    ]


def _axis_offsets(
    total: int,
    size: int,
    overlap_ratio: float,
) -> list[tuple[int, int]]:
    """한 축에서 (offset, length) 목록을 계산한다.

    total <= size면 그 축은 자를 필요가 없으므로 (0, total) 하나만 반환한다.
    """
    if total <= size:
        return [(0, total)]

    step = max(1, round(size * (1.0 - overlap_ratio)))
    offsets: list[tuple[int, int]] = []
    offset = 0
    while offset + size < total:
        offsets.append((offset, size))
        offset += step

    last = (total - size, size)
    if not offsets or offsets[-1] != last:
        offsets.append(last)
    return offsets


def restore_detection_to_global(
    detection: Detection,
    offset_x: int,
    offset_y: int,
    image_width: int,
    image_height: int,
) -> Detection | None:
    """slice 내부 좌표의 Detection을 원본 이미지 좌표로 옮기고 경계 밖을 clip한다.

    clip 후 bbox 면적이 사라지면(완전히 경계 밖이었던 경우) None을 반환해서
    호출부가 그 detection을 버릴 수 있게 한다.
    """
    x1, y1, x2, y2 = detection.bbox
    global_x1 = x1 + offset_x
    global_y1 = y1 + offset_y
    global_x2 = x2 + offset_x
    global_y2 = y2 + offset_y

    clipped_x1 = min(max(0.0, global_x1), float(image_width))
    clipped_y1 = min(max(0.0, global_y1), float(image_height))
    clipped_x2 = min(max(0.0, global_x2), float(image_width))
    clipped_y2 = min(max(0.0, global_y2), float(image_height))

    if clipped_x2 <= clipped_x1 or clipped_y2 <= clipped_y1:
        return None

    return Detection(
        class_id=detection.class_id,
        class_name=detection.class_name,
        confidence=detection.confidence,
        bbox=(clipped_x1, clipped_y1, clipped_x2, clipped_y2),
    )


def global_nms(detections: list[Detection], iou_threshold: float) -> list[Detection]:
    """여러 slice(및 full-image)에서 모인 Detection을 class-aware NMS로 하나로 정리한다.

    app/inference/onnx_backend.py의 기존 NMS 구현(_class_aware_nms/_box_iou)을 그대로
    재사용한다 — SAHI 실험 때문에 NMS를 새로 구현하지 않는다. overlap 영역에서 같은
    객체가 여러 slice에서 중복 탐지된 경우를 이 단계에서 하나로 정리한다.
    """
    if not detections:
        return []

    boxes = np.asarray([detection.bbox for detection in detections], dtype=np.float32)
    scores = np.asarray([detection.confidence for detection in detections], dtype=np.float32)
    class_ids = np.asarray([detection.class_id for detection in detections], dtype=np.int64)

    keep_indices = _class_aware_nms(
        boxes=boxes,
        scores=scores,
        class_ids=class_ids,
        iou_threshold=iou_threshold,
    )
    return [detections[index] for index in keep_indices]


def run_sliced_inference(
    detector: Detector,
    image: np.ndarray,
    slice_width: int,
    slice_height: int,
    overlap_width_ratio: float,
    overlap_height_ratio: float,
) -> tuple[list[Detection], float]:
    """이미지를 slice로 나눠 각각 추론하고, 원본 좌표로 복원한 Detection 목록을 반환한다.

    global NMS는 호출부(run_sahi.py는 이 결과만으로, run_hybrid.py는 full-image 결과와
    합친 뒤)에서 수행하도록 여기서는 적용하지 않는다. 반환하는 latency는 slice별
    Detector.detect() 호출 시간의 합이다(slicing 자체의 배열 연산 비용은 포함하지 않음).
    """
    image_height, image_width = image.shape[:2]
    slice_boxes = compute_slice_boxes(
        image_width=image_width,
        image_height=image_height,
        slice_width=slice_width,
        slice_height=slice_height,
        overlap_width_ratio=overlap_width_ratio,
        overlap_height_ratio=overlap_height_ratio,
    )

    restored_detections: list[Detection] = []
    total_latency_ms = 0.0

    for slice_box in slice_boxes:
        crop = image[
            slice_box.offset_y : slice_box.offset_y + slice_box.height,
            slice_box.offset_x : slice_box.offset_x + slice_box.width,
        ]

        start = time.perf_counter()
        local_detections = detector.detect(crop)
        total_latency_ms += (time.perf_counter() - start) * 1000.0

        for local_detection in local_detections:
            restored = restore_detection_to_global(
                detection=local_detection,
                offset_x=slice_box.offset_x,
                offset_y=slice_box.offset_y,
                image_width=image_width,
                image_height=image_height,
            )
            if restored is not None:
                restored_detections.append(restored)

    return restored_detections, total_latency_ms


__all__ = [
    "SliceBox",
    "compute_slice_boxes",
    "restore_detection_to_global",
    "global_nms",
    "run_sliced_inference",
]
