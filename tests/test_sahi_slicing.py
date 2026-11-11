"""experiments/sahi/slicing.py의 slice coverage, 좌표 복원, global NMS를 검증한다.

실제 모델/weight 없이도 전부 검증 가능한 순수 geometry/NMS 로직만 다룬다.
"""

from __future__ import annotations

import pytest

from app.inference.detector import Detection
from experiments.sahi.slicing import (
    SliceBox,
    compute_slice_boxes,
    global_nms,
    restore_detection_to_global,
)


def _assert_axis_fully_covered(intervals: list[tuple[int, int]], total: int) -> None:
    """(offset, length) 목록이 [0, total) 구간을 간격 없이 모두 덮는지 확인한다."""
    sorted_intervals = sorted(intervals)
    covered_until = 0
    for offset, length in sorted_intervals:
        assert offset <= covered_until, f"gap detected before offset={offset} (covered_until={covered_until})"
        covered_until = max(covered_until, offset + length)
    assert covered_until >= total, f"coverage ends at {covered_until}, expected >= {total}"


def test_slice_boxes_cover_full_image_without_gaps() -> None:
    """1920x1080 이미지를 slice=640, overlap=0.2로 나누면 전체 영역이 빠짐없이 덮여야 한다."""
    boxes = compute_slice_boxes(
        image_width=1920,
        image_height=1080,
        slice_width=640,
        slice_height=640,
        overlap_width_ratio=0.2,
        overlap_height_ratio=0.2,
    )

    x_intervals = sorted({(box.offset_x, box.width) for box in boxes})
    y_intervals = sorted({(box.offset_y, box.height) for box in boxes})
    _assert_axis_fully_covered(x_intervals, total=1920)
    _assert_axis_fully_covered(y_intervals, total=1080)


def test_slice_boxes_last_slice_does_not_exceed_image_bounds() -> None:
    boxes = compute_slice_boxes(
        image_width=1920,
        image_height=1080,
        slice_width=640,
        slice_height=640,
        overlap_width_ratio=0.2,
        overlap_height_ratio=0.2,
    )

    for box in boxes:
        assert box.offset_x >= 0
        assert box.offset_y >= 0
        assert box.offset_x + box.width <= 1920
        assert box.offset_y + box.height <= 1080


def test_slice_boxes_apply_overlap_between_adjacent_slices() -> None:
    """overlap_width_ratio=0.2, slice=640이면 인접 slice는 x축에서 겹치는 영역이 있어야 한다."""
    boxes = compute_slice_boxes(
        image_width=1920,
        image_height=640,
        slice_width=640,
        slice_height=640,
        overlap_width_ratio=0.2,
        overlap_height_ratio=0.2,
    )

    x_offsets = sorted({box.offset_x for box in boxes})
    assert len(x_offsets) >= 2
    first_slice_end = x_offsets[0] + 640
    assert x_offsets[1] < first_slice_end  # 두 번째 slice가 첫 slice 끝보다 앞에서 시작 = overlap


def test_slice_boxes_handle_image_smaller_than_slice() -> None:
    """원본 이미지가 slice보다 작으면 이미지 전체 크기의 slice 1개여야 한다."""
    boxes = compute_slice_boxes(
        image_width=300,
        image_height=200,
        slice_width=640,
        slice_height=640,
        overlap_width_ratio=0.2,
        overlap_height_ratio=0.2,
    )

    assert boxes == [SliceBox(offset_x=0, offset_y=0, width=300, height=200)]


def test_slice_boxes_no_overlap_tiles_exactly() -> None:
    """overlap=0이고 정확히 나눠지는 경우, slice들이 중복 없이 딱 맞게 타일링돼야 한다."""
    boxes = compute_slice_boxes(
        image_width=1280,
        image_height=640,
        slice_width=640,
        slice_height=640,
        overlap_width_ratio=0.0,
        overlap_height_ratio=0.0,
    )

    x_offsets = sorted({box.offset_x for box in boxes})
    assert x_offsets == [0, 640]


def test_slice_boxes_rejects_invalid_overlap_ratio() -> None:
    with pytest.raises(ValueError):
        compute_slice_boxes(
            image_width=1920,
            image_height=1080,
            slice_width=640,
            slice_height=640,
            overlap_width_ratio=1.0,
            overlap_height_ratio=0.2,
        )


def _detection(bbox: tuple[float, float, float, float], confidence: float = 0.8) -> Detection:
    return Detection(class_id=3, class_name="helmet", confidence=confidence, bbox=bbox)


def test_restore_detection_to_global_adds_slice_offset() -> None:
    """spec 예시: offset=(640,0), local bbox=(100,200,200,300) -> global (740,200,840,300)."""
    detection = _detection((100.0, 200.0, 200.0, 300.0))

    restored = restore_detection_to_global(
        detection=detection,
        offset_x=640,
        offset_y=0,
        image_width=1920,
        image_height=1080,
    )

    assert restored is not None
    assert restored.bbox == (740.0, 200.0, 840.0, 300.0)
    assert restored.class_name == "helmet"
    assert restored.confidence == detection.confidence


def test_restore_detection_to_global_clips_to_image_bounds() -> None:
    """slice 경계 부근 bbox가 offset을 더한 뒤 원본 이미지 경계를 넘으면 clip돼야 한다."""
    detection = _detection((600.0, 600.0, 700.0, 700.0))  # slice 내부에서 경계에 거의 닿은 bbox

    restored = restore_detection_to_global(
        detection=detection,
        offset_x=1280,  # 1280+700=1980 > image_width(1920)
        offset_y=0,
        image_width=1920,
        image_height=1080,
    )

    assert restored is not None
    assert restored.bbox[2] == 1920.0  # x2가 이미지 폭으로 clip됨


def test_restore_detection_to_global_drops_detection_entirely_outside_bounds() -> None:
    detection = _detection((10.0, 10.0, 20.0, 20.0))

    restored = restore_detection_to_global(
        detection=detection,
        offset_x=-100,  # offset 적용 후 완전히 음수 영역 -> clip하면 면적이 사라짐
        offset_y=0,
        image_width=1920,
        image_height=1080,
    )

    assert restored is None


def test_global_nms_merges_duplicate_detection_in_overlap_region() -> None:
    """overlap 영역에서 같은 class의 거의 동일한 bbox 두 개가 하나로 정리돼야 한다."""
    detections = [
        _detection((100.0, 100.0, 200.0, 200.0), confidence=0.9),
        _detection((102.0, 101.0, 199.0, 198.0), confidence=0.7),  # 거의 같은 위치, 다른 slice에서 재탐지
    ]

    kept = global_nms(detections, iou_threshold=0.5)

    assert len(kept) == 1
    assert kept[0].confidence == 0.9  # 더 높은 confidence가 남는다


def test_global_nms_keeps_distinct_boxes_of_same_class() -> None:
    detections = [
        _detection((0.0, 0.0, 50.0, 50.0), confidence=0.9),
        _detection((500.0, 500.0, 560.0, 560.0), confidence=0.8),  # 겹치지 않는 별개 객체
    ]

    kept = global_nms(detections, iou_threshold=0.5)

    assert len(kept) == 2


def test_global_nms_does_not_suppress_across_different_classes() -> None:
    overlapping_but_different_class = Detection(
        class_id=0, class_name="person", confidence=0.85, bbox=(100.0, 100.0, 200.0, 200.0)
    )
    detections = [
        _detection((100.0, 100.0, 200.0, 200.0), confidence=0.9),  # helmet
        overlapping_but_different_class,  # person, 같은 위치지만 다른 class
    ]

    kept = global_nms(detections, iou_threshold=0.5)

    assert len(kept) == 2


def test_global_nms_empty_input_returns_empty_list() -> None:
    assert global_nms([], iou_threshold=0.5) == []
