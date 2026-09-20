"""설정 기반 Static Zone과 설비별 Dynamic Zone을 조립하는 모듈."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from app.tracking.tracker import TrackedObject
from app.video.video_source import FramePacket, PROJECT_ROOT
from app.zones.conveyor import create_conveyor_zone_infos
from app.zones.forklift import create_forklift_zone_infos
from app.zones.models import EquipmentState, Polygon, Zone, ZoneFrameResult
from app.zones.robot_arm import RobotArmZoneBuilder
from app.zones.worker_zone import create_worker_zone_infos

DEFAULT_ZONE_CONFIG_PATH = PROJECT_ROOT / "config" / "zones.yaml"
_SUPPORTED_RISK_LEVELS = {"caution", "warning", "critical"}


@dataclass(frozen=True, slots=True)
class ZoneManagerConfig:
    conveyor_warning_margin_px: float = 0.0
    forklift_warning_margin_px: float = 120.0
    robot_arm_warning_margin_px: float = 40.0
    robot_arm_smoothing_frames: int = 5

    def __post_init__(self) -> None:
        margins = (
            self.conveyor_warning_margin_px,
            self.forklift_warning_margin_px,
            self.robot_arm_warning_margin_px,
        )
        if any(margin < 0 for margin in margins):
            raise ValueError("zone margins must not be negative")
        if self.robot_arm_smoothing_frames <= 0:
            raise ValueError("robot_arm_smoothing_frames must be positive")


class ZoneManager:
    def __init__(
        self,
        zones: Sequence[Zone] = (),
        config: ZoneManagerConfig | None = None,
    ) -> None:
        self.zones = tuple(zones)
        self.config = config or ZoneManagerConfig()
        self._robot_arm_builder = RobotArmZoneBuilder(
            warning_margin_px=self.config.robot_arm_warning_margin_px,
            smoothing_frames=self.config.robot_arm_smoothing_frames,
        )

    @classmethod
    def from_config(
        cls,
        config_path: str | Path = DEFAULT_ZONE_CONFIG_PATH,
    ) -> "ZoneManager":
        config = load_zone_config(config_path)
        return cls(
            zones=_load_static_zones(config),
            config=_load_manager_config(config),
        )

    def process(
        self,
        frame_packet: FramePacket,
        tracked_objects: list[TrackedObject],
        equipment_states: Mapping[int, EquipmentState] | None = None,
    ) -> ZoneFrameResult:
        states = equipment_states or {}
        workers = create_worker_zone_infos(tracked_objects)
        equipments = create_conveyor_zone_infos(
            tracked_objects,
            warning_margin_px=self.config.conveyor_warning_margin_px,
            states=states,
            frame_width=frame_packet.width,
            frame_height=frame_packet.height,
        )
        equipments.extend(
            create_forklift_zone_infos(
                tracked_objects,
                warning_margin_px=self.config.forklift_warning_margin_px,
                states=states,
                frame_width=frame_packet.width,
                frame_height=frame_packet.height,
            )
        )
        equipments.extend(
            self._robot_arm_builder.update(
                tracked_objects,
                states=states,
                frame_width=frame_packet.width,
                frame_height=frame_packet.height,
            )
        )

        return ZoneFrameResult(
            timestamp=frame_packet.timestamp,
            frame_index=frame_packet.frame_index,
            camera_id=frame_packet.camera_id,
            workers=workers,
            equipments=equipments,
            zones=[
                zone
                for zone in self.zones
                if zone.camera_id == frame_packet.camera_id
            ],
        )

    def reset(self) -> None:
        self._robot_arm_builder.reset()


def load_zone_config(
    config_path: str | Path = DEFAULT_ZONE_CONFIG_PATH,
) -> Mapping[str, Any]:
    path = Path(config_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    if not isinstance(config, Mapping):
        raise ValueError("zone config must be a mapping")
    return config


def _load_static_zones(config: Mapping[str, Any]) -> list[Zone]:
    zone_entries = config.get("zones", [])
    if not isinstance(zone_entries, list):
        raise ValueError("zones must be a list")

    zones: list[Zone] = []
    for entry in zone_entries:
        if not isinstance(entry, Mapping):
            raise ValueError("zone entry must be a mapping")
        risk_level = str(entry.get("risk_level", "warning")).strip().lower()
        if risk_level not in _SUPPORTED_RISK_LEVELS:
            raise ValueError(f"Unsupported zone risk level: {risk_level}")
        zones.append(
            Zone(
                zone_id=_required_str(entry, "zone_id"),
                camera_id=_required_str(entry, "camera_id"),
                zone_type=_required_str(entry, "zone_type"),
                polygon=_polygon_from_value(entry.get("polygon")),
                risk_level=risk_level,
            )
        )
    return zones


def _load_manager_config(config: Mapping[str, Any]) -> ZoneManagerConfig:
    conveyor = _optional_mapping(config.get("conveyor"), "conveyor")
    forklift = _optional_mapping(config.get("forklift"), "forklift")
    robot_arm = _optional_mapping(config.get("robot_arm"), "robot_arm")
    return ZoneManagerConfig(
        conveyor_warning_margin_px=float(conveyor.get("bbox_margin_px", 0.0)),
        forklift_warning_margin_px=float(forklift.get("static_buffer_px", 120.0)),
        robot_arm_warning_margin_px=float(robot_arm.get("bbox_margin_px", 40.0)),
        robot_arm_smoothing_frames=int(robot_arm.get("smoothing_frames", 5)),
    )


def _polygon_from_value(value: object) -> Polygon:
    if not isinstance(value, list) or len(value) < 3:
        raise ValueError("zone polygon must contain at least three points")
    points: list[tuple[float, float]] = []
    for point in value:
        if not isinstance(point, list | tuple) or len(point) != 2:
            raise ValueError("polygon point must contain x and y")
        x, y = point
        if isinstance(x, bool) or isinstance(y, bool):
            raise ValueError("polygon coordinates must be numeric")
        points.append((float(x), float(y)))
    return tuple(points)


def _required_str(values: Mapping[str, Any], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _optional_mapping(value: object, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} config must be a mapping")
    return value
