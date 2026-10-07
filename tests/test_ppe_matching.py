"""모델/GPU 없이 PPE 공간 매칭 계약을 검증한다."""

from dataclasses import replace

import pytest

from app.inference.detector import Detection
from app.ppe.matcher import PPEMatcher
from app.tracking.tracker import TrackedObject


def worker(track_id=7, bbox=(0, 0, 100, 200)):
    detection = Detection(0, "person", 0.9, bbox)
    return TrackedObject(
        track_id, 0, "person", detection.bbox, detection.center,
        detection.bottom_center, 0.9, 0.0,
    )


def ppe(name="helmet", bbox=(40, 10, 60, 30)):
    # class_name을 사용하므로 backend의 class ID 순서에 의존하지 않는다.
    return Detection(99, name, 0.8, bbox)


def statuses(result, track_id=7):
    return {item.class_name: item for item in result[track_id].items}


def test_worker_has_all_ppe_types_and_original_evidence():
    helmet = ppe()
    vest = ppe("safety_vest", (20, 50, 80, 120))
    items = statuses(PPEMatcher().match([helmet, vest], [worker()]))
    assert {name: item.is_worn for name, item in items.items()} == {
        "helmet": True, "gloves": False, "safety_vest": True, "harness_body": False,
    }
    assert items["helmet"].detections[0] is helmet


def test_empty_inputs_and_irrelevant_classes():
    matcher = PPEMatcher()
    assert matcher.match([ppe()], []) == {}
    assert matcher.match([], []) == {}
    detections = [ppe("head"), ppe("body"), ppe("person"), ppe("forklift")]
    result = matcher.match(detections, [worker(), replace(worker(8), class_name="forklift")])
    assert list(result) == [7]
    assert not any(item.is_worn for item in result[7].items)
    assert not any(item.is_worn for item in matcher.match([], [worker()])[7].items)


@pytest.mark.parametrize("bbox,expected", [
    ((-10, 10, 10, 30), True),  # center가 왼쪽 경계
    ((90, 190, 110, 210), True),  # 오른쪽 아래 경계
    ((-12, 10, 10, 30), False),  # 겹쳐도 center가 밖이면 제외
])
def test_center_containment_includes_boundary(bbox, expected):
    result = PPEMatcher().match([ppe(bbox=bbox)], [worker()])
    assert statuses(result)["helmet"].is_worn is expected


def test_overlap_selects_one_owner_independent_of_worker_order():
    workers = [worker(7), worker(8, (40, 0, 140, 200))]
    helmet = ppe(bbox=(75, 10, 95, 30))
    matcher = PPEMatcher()
    first = matcher.match([helmet], workers)
    assert first == matcher.match([helmet], workers[::-1])
    assert not statuses(first, 7)["helmet"].is_worn
    assert statuses(first, 8)["helmet"].is_worn
    assert sum(len(item.detections) for result in first.values() for item in result.items) == 1


def test_equal_distance_uses_smallest_track_id():
    result = PPEMatcher().match([ppe()], [worker(8), worker(7)])
    assert statuses(result, 7)["helmet"].is_worn
    assert not statuses(result, 8)["helmet"].is_worn


def test_distance_is_normalized_by_person_size():
    # 픽셀 거리는 7번이 가깝지만 정규화 거리는 8번이 가깝다.
    result = PPEMatcher().match(
        [ppe(bbox=(59, 99, 61, 101))],
        [worker(7, (40, 0, 60, 200)), worker(8, (0, 0, 200, 200))],
    )
    assert statuses(result, 8)["helmet"].is_worn
    assert not statuses(result, 7)["helmet"].is_worn


def test_multiple_gloves_and_harness_and_no_state_retention():
    matcher = PPEMatcher()
    gloves = [ppe("gloves", (5, 90, 15, 110)), ppe("gloves", (85, 90, 95, 110))]
    items = statuses(matcher.match([*gloves, ppe("harness_body")], [worker()]))
    assert items["gloves"].detections == tuple(gloves)
    assert items["harness_body"].is_worn
    assert not any(item.is_worn for item in matcher.match([], [worker()])[7].items)


@pytest.mark.parametrize("bbox", [(0, 0, 0, 10), (0, 20, 10, 10), (0, 0, float("nan"), 10)])
def test_invalid_person_bbox_is_rejected(bbox):
    with pytest.raises(ValueError, match="person"):
        PPEMatcher().match([ppe()], [replace(worker(), bbox=bbox)])


def test_duplicate_person_track_ids_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        PPEMatcher().match([], [worker(), worker()])
