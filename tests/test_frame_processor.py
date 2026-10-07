"""프레임 처리 단계에서 detection과 tracking 연결을 검증하는 테스트."""

import numpy as np

from app.inference.detector import Detection, Detector
from app.risk.models import RiskLevel
from app.risk.risk_engine import RiskEngine
from app.tracking.tracker import SimpleTracker
from app.tracking.trajectory import TrajectoryAnalyzer
from app.video.frame_processor import FrameProcessor
from app.video.video_source import FramePacket
from app.zones.zone_manager import ZoneManager


class _StaticDetector(Detector):
    def detect(self, frame: np.ndarray) -> list[Detection]:
        self.validate_frame(frame)
        return [
            Detection(
                class_id=0,
                class_name="person",
                confidence=0.9,
                bbox=(10.0, 10.0, 40.0, 60.0),
            )
        ]


class _SequenceDetector(Detector):
    def __init__(self) -> None:
        self._next_x1 = 10.0

    def detect(self, frame: np.ndarray) -> list[Detection]:
        self.validate_frame(frame)
        x1 = self._next_x1
        self._next_x1 += 3.0
        return [
            Detection(
                class_id=0,
                class_name="person",
                confidence=0.9,
                bbox=(x1, 10.0, x1 + 30.0, 60.0),
            )
        ]


class _WorkerAndForkliftDetector(Detector):
    """Zone/Risk 연결을 검증하기 위해 작업자 1명과 지게차 1대를 함께 탐지하는 fake detector.

    작업자 bbox의 bottom_center가 지게차 bbox(=critical_zone) 내부에 들어오도록
    좌표를 구성해, ZoneManager와 RiskEngine이 실제로 CRITICAL을 계산하는지 확인한다.
    """

    def detect(self, frame: np.ndarray) -> list[Detection]:
        self.validate_frame(frame)
        return [
            Detection(
                class_id=0,
                class_name="forklift",
                confidence=0.9,
                bbox=(100.0, 100.0, 300.0, 300.0),
            ),
            Detection(
                class_id=1,
                class_name="person",
                confidence=0.9,
                bbox=(180.0, 150.0, 220.0, 250.0),
            ),
        ]


class _EmptyDetector(Detector):
    """탐지 결과가 없는 프레임에서 Zone/Risk 연결이 정상 동작하는지 검증하기 위한 fake detector."""

    def detect(self, frame: np.ndarray) -> list[Detection]:
        self.validate_frame(frame)
        return []


class _FrameSource:
    def __init__(self, frame_size: int = 100) -> None:
        self._frame_index = 0
        self._frame_size = frame_size

    def read(self) -> FramePacket | None:
        if self._frame_index >= 2:
            return None

        frame = np.zeros((self._frame_size, self._frame_size, 3), dtype=np.uint8)
        packet = FramePacket(
            camera_id="test-camera",
            frame=frame,
            frame_index=self._frame_index,
            timestamp=float(self._frame_index),
            fps=30.0,
            width=self._frame_size,
            height=self._frame_size,
        )
        self._frame_index += 1
        return packet


def test_frame_processor_returns_tracked_objects() -> None:
    processor = FrameProcessor(
        video_source=_FrameSource(),
        detector=_StaticDetector(),
        zone_manager=ZoneManager(),
        risk_engine=RiskEngine(),
        tracker=SimpleTracker(),
        draw_bbox=False,
        draw_metrics=False,
    )

    processed_frame = processor.process_next()

    assert processed_frame is not None
    assert len(processed_frame.detections) == 1
    assert len(processed_frame.tracked_objects) == 1
    assert processed_frame.tracked_objects[0].track_id == 1
    assert processed_frame.tracked_objects[0].class_name == "person"
    assert processed_frame.rendered_frame is None
    # ZoneManager/RiskEngine은 필수 의존성이므로 항상 실행된다.
    # 탐지된 설비가 없으므로 workers만 채워지고 risk_assessments는 빈 리스트가 된다.
    assert processed_frame.zone_result is not None
    assert len(processed_frame.zone_result.workers) == 1
    assert processed_frame.zone_result.equipments == []
    assert processed_frame.risk_assessments == []


def test_frame_processor_returns_motion_summaries() -> None:
    processor = FrameProcessor(
        video_source=_FrameSource(),
        detector=_SequenceDetector(),
        zone_manager=ZoneManager(),
        risk_engine=RiskEngine(),
        tracker=SimpleTracker(),
        trajectory_analyzer=TrajectoryAnalyzer(),
        draw_bbox=False,
        draw_metrics=False,
    )

    first_frame = processor.process_next()
    second_frame = processor.process_next()

    assert first_frame is not None
    assert second_frame is not None
    assert second_frame.motion_summaries[1].distance_px == 3.0
    assert second_frame.motion_summaries[1].direction == (1.0, 0.0)
    assert second_frame.motion_summaries[1].speed_px_per_sec == 3.0


def test_frame_processor_generates_zone_result_and_risk_assessments() -> None:
    """Tracking → Zone → Risk 필수 흐름을 거쳐 ZoneFrameResult와
    RiskAssessment가 실제로 생성돼야 한다."""

    processor = FrameProcessor(
        video_source=_FrameSource(),
        detector=_WorkerAndForkliftDetector(),
        tracker=SimpleTracker(),
        zone_manager=ZoneManager(),
        risk_engine=RiskEngine(),
        draw_bbox=False,
        draw_metrics=False,
    )

    processed_frame = processor.process_next()

    assert processed_frame is not None
    assert processed_frame.zone_result is not None
    assert len(processed_frame.zone_result.workers) == 1
    assert len(processed_frame.zone_result.equipments) == 1

    assert len(processed_frame.risk_assessments) == 1
    assert processed_frame.risk_assessments[0].risk_level == RiskLevel.CRITICAL


def test_frame_processor_zone_result_is_empty_without_detections() -> None:
    """탐지된 작업자/설비가 없는 프레임에서도 ZoneManager/RiskEngine 연결이
    예외 없이 빈 결과를 생성해야 한다."""

    processor = FrameProcessor(
        video_source=_FrameSource(),
        detector=_EmptyDetector(),
        tracker=SimpleTracker(),
        zone_manager=ZoneManager(),
        risk_engine=RiskEngine(),
        draw_bbox=False,
        draw_metrics=False,
    )

    processed_frame = processor.process_next()

    assert processed_frame is not None
    assert processed_frame.zone_result is not None
    assert processed_frame.zone_result.workers == []
    assert processed_frame.zone_result.equipments == []
    assert processed_frame.risk_assessments == []


# Zone 시각화 색상 상수. app/video/frame_processor.py에 정의된 값과 같아야
# 렌더링 결과를 실제 픽셀 값으로 검증할 수 있다.
_CRITICAL_ZONE_COLOR = (0, 0, 255)
_WARNING_ZONE_COLOR = (0, 255, 255)
_TRACKING_BBOX_COLOR = (255, 200, 0)


def test_frame_processor_draws_zone_rectangles_using_equipment_zone_coordinates() -> None:
    """equipment가 있으면 rendered_frame에 Critical/Warning Zone이
    equipment.critical_zone / equipment.warning_zone 좌표 그대로 그려져야 한다."""

    # frame을 넉넉하게 잡아 Zone 좌표가 전부 화면 안쪽에 들어오게 한다.
    processor = FrameProcessor(
        video_source=_FrameSource(frame_size=400),
        detector=_WorkerAndForkliftDetector(),
        tracker=SimpleTracker(),
        zone_manager=ZoneManager(),
        risk_engine=RiskEngine(),
        draw_bbox=True,
        draw_metrics=False,
    )

    processed_frame = processor.process_next()

    assert processed_frame is not None
    rendered_frame = processed_frame.rendered_frame
    assert rendered_frame is not None

    equipment = processed_frame.zone_result.equipments[0]
    critical_x1, critical_y1, _, _ = (int(v) for v in equipment.critical_zone)
    warning_x1, warning_y1, _, _ = (int(v) for v in equipment.warning_zone)

    # Critical/Warning Zone은 Tracking bbox보다 나중에 그려지므로, 각 Zone의
    # 좌상단 꼭짓점 픽셀(사각형 테두리 위이므로 항상 그려짐)은 해당 Zone 색이어야 한다.
    assert tuple(rendered_frame[critical_y1, critical_x1]) == _CRITICAL_ZONE_COLOR
    assert tuple(rendered_frame[warning_y1, warning_x1]) == _WARNING_ZONE_COLOR

    # Tracking bbox가 Critical Zone(equipment bbox와 좌표가 같음)보다 두껍게 그려지므로,
    # 겹치는 경계 부근에 두 색이 함께 남아 bbox 색도 일부 화면에 보여야 한다.
    assert np.any(np.all(rendered_frame == _TRACKING_BBOX_COLOR, axis=-1))


def test_frame_processor_clamps_zone_coordinates_outside_frame_bounds() -> None:
    """Zone 좌표가 frame 크기를 벗어나도 예외 없이 clamp되어 그려져야 한다."""

    # forklift bbox(critical_zone과 동일)의 하단·우측과, 그보다 더 넓은 warning_zone이
    # frame 밖으로 나가도록 frame을 일부러 작게 잡는다.
    processor = FrameProcessor(
        video_source=_FrameSource(frame_size=250),
        detector=_WorkerAndForkliftDetector(),
        tracker=SimpleTracker(),
        zone_manager=ZoneManager(),
        risk_engine=RiskEngine(),
        draw_bbox=True,
        draw_metrics=False,
    )

    processed_frame = processor.process_next()

    assert processed_frame is not None
    rendered_frame = processed_frame.rendered_frame
    assert rendered_frame is not None
    assert rendered_frame.shape == (250, 250, 3)

    equipment = processed_frame.zone_result.equipments[0]
    # 원본 EquipmentZoneInfo 값 자체는 clamp되지 않고 그대로 유지돼야 한다.
    assert equipment.critical_zone == (100.0, 100.0, 300.0, 300.0)
    assert equipment.warning_zone[2] > 250.0
    assert equipment.warning_zone[3] > 250.0

    # frame 안쪽인 좌상단 좌표는 원래 값 그대로 그려지고,
    # frame 밖으로 나가는 우하단 좌표는 (249, 249)로 clamp되어 예외 없이 그려진다.
    assert tuple(rendered_frame[100, 100]) == _CRITICAL_ZONE_COLOR
    assert tuple(rendered_frame[80, 80]) == _WARNING_ZONE_COLOR
    assert tuple(rendered_frame[249, 249]) == _WARNING_ZONE_COLOR


def test_frame_processor_renders_without_zone_colors_when_no_equipment() -> None:
    """탐지된 설비가 없으면 예외 없이 렌더링되고 Zone 색상도 나타나지 않아야 한다."""

    processor = FrameProcessor(
        video_source=_FrameSource(),
        detector=_EmptyDetector(),
        tracker=SimpleTracker(),
        zone_manager=ZoneManager(),
        risk_engine=RiskEngine(),
        draw_bbox=True,
        draw_metrics=False,
    )

    processed_frame = processor.process_next()

    assert processed_frame is not None
    rendered_frame = processed_frame.rendered_frame
    assert rendered_frame is not None
    assert not np.any(np.all(rendered_frame == _CRITICAL_ZONE_COLOR, axis=-1))
    assert not np.any(np.all(rendered_frame == _WARNING_ZONE_COLOR, axis=-1))


class _PPEDetector(_StaticDetector):
    def __init__(self):
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        detections = super().detect(frame)
        if self.calls == 1:
            # tracker threshold보다 낮아도 원본 Detection에서 PPE를 연결한다.
            detections.append(Detection(3, "helmet", 0.3, (20, 10, 30, 20)))
        return detections


def test_frame_processor_matches_raw_ppe_and_clears_next_frame():
    detector = _PPEDetector()
    processor = FrameProcessor(
        video_source=_FrameSource(), detector=detector, tracker=SimpleTracker(),
        zone_manager=ZoneManager(), risk_engine=RiskEngine(),
        draw_bbox=False, draw_metrics=False,
    )
    first = processor.process_next()
    second = processor.process_next()
    assert first.ppe_statuses[1].items[0].is_worn
    assert first.ppe_statuses[1].items[0].detections[0] is first.detections[1]
    assert not second.ppe_statuses[1].items[0].is_worn
    assert len(first.tracked_objects) == 1
    assert len(first.zone_result.workers) == 1
    assert first.risk_assessments == second.risk_assessments == []
    assert detector.calls == 2
    assert processor.process_next() is None


def test_frame_processor_without_tracker_has_no_ppe_statuses():
    processor = FrameProcessor(
        video_source=_FrameSource(), detector=_PPEDetector(),
        zone_manager=ZoneManager(), risk_engine=RiskEngine(),
        draw_bbox=False, draw_metrics=False,
    )
    assert processor.process_next().ppe_statuses == {}
