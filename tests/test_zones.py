"""Zone 모듈의 공간 계산과 출력 데이터 생성을 검증하는 테스트."""

from types import SimpleNamespace

import pytest

from app.tracking.tracker import TrackedObject
from app.zones.conveyor import create_conveyor_zone_info
from app.zones.forklift import create_forklift_zone_info
from app.zones.geometry import is_point_in_bbox, scale_bbox
from app.zones.models import EquipmentType
from app.zones.robot_arm import create_robot_arm_zone_info
from app.zones.worker_zone import (
    create_worker_zone_info,
    create_worker_zone_infos,
)
from app.zones.zone_manager import ZoneManager


def make_tracked_object(
    track_id: int,
    class_name: str,
    bbox=(100.0, 100.0, 300.0, 300.0),
) -> TrackedObject:
    x1, y1, x2, y2 = bbox

    center = (
        (x1 + x2) / 2,
        (y1 + y2) / 2,
    )

    bottom_center = (
        (x1 + x2) / 2,
        y2,
    )

    return TrackedObject(
        track_id=track_id,
        class_id=0,
        class_name=class_name,
        bbox=bbox,
        center=center,
        bottom_center=bottom_center,
        confidence=0.9,
        timestamp=1000.0,
    )



def test_scale_bbox():
    bbox = (100.0, 100.0, 300.0, 300.0)

    result = scale_bbox(bbox, 1.2)

    assert result == pytest.approx(
        (80.0, 80.0, 320.0, 320.0)
    )


def test_is_point_in_bbox_inside():
    bbox = (100.0, 100.0, 300.0, 300.0)

    assert is_point_in_bbox(
        (200.0, 200.0),
        bbox,
    )


def test_is_point_in_bbox_outside():
    bbox = (100.0, 100.0, 300.0, 300.0)

    assert not is_point_in_bbox(
        (400.0, 400.0),
        bbox,
    )


def test_create_worker_zone_info():
    tracked_object = make_tracked_object(
        track_id=7,
        class_name="person",
        bbox=(100.0, 100.0, 200.0, 500.0),
    )

    worker = create_worker_zone_info(
        tracked_object
    )

    assert worker.person_id == 7
    assert worker.person_bbox == (
        100.0,
        100.0,
        200.0,
        500.0,
    )
    assert worker.bottom_center == (
        150.0,
        500.0,
    )


def test_create_worker_zone_infos_filters_non_person():
    tracked_objects = [
        make_tracked_object(1, "person"),
        make_tracked_object(2, "forklift"),
        make_tracked_object(3, "person"),
    ]

    workers = create_worker_zone_infos(
        tracked_objects
    )

    assert len(workers) == 2
    assert workers[0].person_id == 1
    assert workers[1].person_id == 3


def test_create_conveyor_zone_info():
    tracked_object = make_tracked_object(
        track_id=10,
        class_name="conveyor",
    )

    result = create_conveyor_zone_info(
        tracked_object
    )

    assert result.equipment_id == 10
    assert result.equipment_type == EquipmentType.CONVEYOR

    assert result.critical_zone == (
        100.0,
        100.0,
        300.0,
        300.0,
    )

    assert result.warning_zone == pytest.approx(
        (80.0, 80.0, 320.0, 320.0)
    )


def test_create_forklift_zone_info():
    tracked_object = make_tracked_object(
        track_id=20,
        class_name="forklift",
    )

    result = create_forklift_zone_info(
        tracked_object
    )

    assert result.equipment_id == 20
    assert result.equipment_type == EquipmentType.FORKLIFT


def test_create_robot_arm_zone_info():
    tracked_object = make_tracked_object(
        track_id=30,
        class_name="robot_arm",
    )

    result = create_robot_arm_zone_info(
        tracked_object
    )

    assert result.equipment_id == 30
    assert result.equipment_type == EquipmentType.ROBOT_ARM


def test_zone_manager():
    tracked_objects = [
        make_tracked_object(1, "person"),
        make_tracked_object(2, "conveyor"),
        make_tracked_object(3, "forklift"),
        make_tracked_object(4, "robot_arm"),
    ]

    frame_packet = SimpleNamespace(
        timestamp=1234.5,
        frame_index=100,
        camera_id="cam_01",
    )

    manager = ZoneManager()

    result = manager.process(
        frame_packet=frame_packet,
        tracked_objects=tracked_objects,
    )

    assert result.timestamp == 1234.5
    assert result.frame_index == 100
    assert result.camera_id == "cam_01"

    assert len(result.workers) == 1
    assert len(result.equipments) == 3