"""Polygon Geometry와 설정 기반 Zone 생성을 검증한다."""

import numpy as np
import pytest

from app.tracking.tracker import TrackedObject
from app.video.video_source import FramePacket
from app.zones.geometry import (
    bbox_to_polygon,
    expand_bbox,
    is_point_in_polygon,
)
from app.zones.models import EquipmentState, EquipmentType
from app.zones.robot_arm import RobotArmZoneBuilder
from app.zones.zone_manager import ZoneManager, ZoneManagerConfig


def _tracked(
    track_id: int,
    class_name: str,
    bbox: tuple[float, float, float, float],
) -> TrackedObject:
    x1, y1, x2, y2 = bbox
    return TrackedObject(
        track_id=track_id,
        class_id=0,
        class_name=class_name,
        bbox=bbox,
        center=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
        bottom_center=((x1 + x2) / 2.0, y2),
        confidence=0.9,
        timestamp=1.0,
    )


def _packet(camera_id: str = "cam-1") -> FramePacket:
    return FramePacket(
        camera_id=camera_id,
        frame=np.zeros((100, 100, 3), dtype=np.uint8),
        frame_index=0,
        timestamp=1.0,
        fps=30.0,
        width=100,
        height=100,
    )


def test_point_in_polygon_includes_boundary() -> None:
    polygon = ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0))
    assert is_point_in_polygon((5.0, 5.0), polygon)
    assert is_point_in_polygon((10.0, 5.0), polygon)
    assert not is_point_in_polygon((11.0, 5.0), polygon)


def test_expand_bbox_clips_to_frame() -> None:
    assert expand_bbox(
        (5.0, 5.0, 20.0, 20.0),
        margin_px=10.0,
        frame_width=30,
        frame_height=30,
    ) == (0.0, 0.0, 29.0, 29.0)


def test_zone_manager_loads_camera_polygon_and_equipment_settings(tmp_path) -> None:
    config_path = tmp_path / "zones.yaml"
    config_path.write_text(
        """
zones:
  - zone_id: danger
    camera_id: cam-1
    zone_type: static_danger
    risk_level: critical
    polygon:
      - [0, 0]
      - [80, 0]
      - [80, 80]
      - [0, 80]
robot_arm:
  bbox_margin_px: 10
  smoothing_frames: 2
forklift:
  static_buffer_px: 20
""",
        encoding="utf-8",
    )
    manager = ZoneManager.from_config(config_path)
    result = manager.process(
        frame_packet=_packet(),
        tracked_objects=[
            _tracked(1, "person", (10.0, 10.0, 20.0, 30.0)),
            _tracked(2, "forklift", (40.0, 40.0, 60.0, 60.0)),
        ],
        equipment_states={2: EquipmentState.MOVING},
    )

    assert [zone.zone_id for zone in result.zones] == ["danger"]
    assert [worker.person_id for worker in result.workers] == [1]
    assert result.equipments[0].equipment_type == EquipmentType.FORKLIFT
    assert result.equipments[0].state == EquipmentState.MOVING
    assert result.equipments[0].warning_zone == bbox_to_polygon(
        (20.0, 20.0, 80.0, 80.0)
    )


def test_zone_manager_filters_static_zones_by_camera(tmp_path) -> None:
    config_path = tmp_path / "zones.yaml"
    config_path.write_text(
        """
zones:
  - zone_id: other-camera
    camera_id: cam-2
    zone_type: static_danger
    polygon: [[0, 0], [10, 0], [10, 10]]
""",
        encoding="utf-8",
    )
    result = ZoneManager.from_config(config_path).process(_packet(), [])
    assert result.zones == []


def test_robot_arm_zone_builder_smooths_bbox() -> None:
    builder = RobotArmZoneBuilder(warning_margin_px=0.0, smoothing_frames=2)
    builder.update([_tracked(7, "robot_arm", (0.0, 0.0, 10.0, 10.0))])
    zones = builder.update(
        [_tracked(7, "robot_arm", (10.0, 10.0, 20.0, 20.0))]
    )
    assert zones[0].equipment_bbox == (5.0, 5.0, 15.0, 15.0)


def test_zone_manager_config_rejects_negative_margin() -> None:
    with pytest.raises(ValueError, match="must not be negative"):
        ZoneManagerConfig(forklift_warning_margin_px=-1.0)
