"""프레임별 Zone 정보를 생성하고 ZoneFrameResult로 묶는 모듈."""

from app.tracking.tracker import TrackedObject
from app.video.video_source import FramePacket
from app.zones.conveyor import create_conveyor_zone_infos
from app.zones.forklift import create_forklift_zone_infos
from app.zones.models import ZoneFrameResult
from app.zones.robot_arm import create_robot_arm_zone_infos
from app.zones.worker_zone import create_worker_zone_infos


class ZoneManager:
    def __init__(
        self,
        conveyor_warning_scale: float = 1.2,
        forklift_warning_scale: float = 1.2,
        robot_arm_warning_scale: float = 1.2,
    ):
        self.conveyor_warning_scale = conveyor_warning_scale
        self.forklift_warning_scale = forklift_warning_scale
        self.robot_arm_warning_scale = robot_arm_warning_scale

    def process(
        self,
        frame_packet: FramePacket,
        tracked_objects: list[TrackedObject],
    ) -> ZoneFrameResult:
        workers = create_worker_zone_infos(
            tracked_objects
        )

        conveyor_zones = create_conveyor_zone_infos(
            tracked_objects,
            warning_scale=self.conveyor_warning_scale,
        )

        forklift_zones = create_forklift_zone_infos(
            tracked_objects,
            warning_scale=self.forklift_warning_scale,
        )

        robot_arm_zones = create_robot_arm_zone_infos(
            tracked_objects,
            warning_scale=self.robot_arm_warning_scale,
        )

        equipments = (
            conveyor_zones
            + forklift_zones
            + robot_arm_zones
        )

        return ZoneFrameResult(
            timestamp=frame_packet.timestamp,
            frame_index=frame_packet.frame_index,
            camera_id=frame_packet.camera_id,
            workers=workers,
            equipments=equipments,
        )