"""Alert UI(배지/테두리/blink)와 camera 상태별 노출 여부를 검증하는 테스트."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app import main
from app.alerts.event_manager import EventManager
from app.alerts.models import CameraAlertUpdate
from app.risk.models import RiskAssessment, RiskLevel
from app.risk.risk_logger import RiskLogger
from app.video.video_source import FramePacket, VideoSource
from app.zones.models import EquipmentType, EquipmentZoneInfo, WorkerZoneInfo, ZoneFrameResult


@dataclass(slots=True)
class _FakeProcessedFrame:
    frame_packet: FramePacket
    rendered_frame: np.ndarray | None
    zone_result: ZoneFrameResult = field(
        default_factory=lambda: ZoneFrameResult(
            timestamp=0.0,
            frame_index=0,
            camera_id="cam_01",
            workers=[],
            equipments=[],
        )
    )
    risk_assessments: list = field(default_factory=list)


class _FakeProcessor:
    def __init__(self, processed_frame: _FakeProcessedFrame | None) -> None:
        self._processed_frame = processed_frame

    def process_next(self) -> _FakeProcessedFrame | None:
        return self._processed_frame


def _frame_packet(camera_id: str = "cam_01") -> FramePacket:
    return FramePacket(
        camera_id=camera_id,
        frame=np.zeros((100, 100, 3), dtype=np.uint8),
        frame_index=0,
        timestamp=0.0,
        fps=30.0,
        width=100,
        height=100,
    )


def _zone_result_with_one_pair(camera_id: str) -> ZoneFrameResult:
    """RiskLogger.log()가 person_id=1/equipment_id=1 조합을 조회할 수 있도록
    _assessment()과 짝이 맞는 worker/equipment 하나씩 담은 ZoneFrameResult를 만든다."""
    worker = WorkerZoneInfo(
        person_id=1, person_bbox=(0.0, 0.0, 10.0, 10.0), bottom_center=(5.0, 10.0)
    )
    equipment = EquipmentZoneInfo(
        equipment_id=1,
        equipment_type=EquipmentType.FORKLIFT,
        equipment_bbox=(0.0, 0.0, 20.0, 20.0),
        warning_zone=(-5.0, -5.0, 25.0, 25.0),
        critical_zone=(0.0, 0.0, 20.0, 20.0),
        is_active=True,
    )
    return ZoneFrameResult(
        timestamp=0.0,
        frame_index=0,
        camera_id=camera_id,
        workers=[worker],
        equipments=[equipment],
    )


def _assessment(risk_level: RiskLevel, camera_id: str) -> RiskAssessment:
    return RiskAssessment(
        timestamp=0.0,
        frame_index=0,
        camera_id=camera_id,
        person_id=1,
        equipment_id=1,
        equipment_type=EquipmentType.FORKLIFT,
        risk_level=risk_level,
    )


def _blank_tile() -> np.ndarray:
    return np.zeros((main.TILE_HEIGHT, main.TILE_WIDTH, 3), dtype=np.uint8)


def _badge_bounds() -> tuple[int, int, int, int]:
    x2 = main.TILE_WIDTH - main.ALERT_BADGE_MARGIN
    x1 = x2 - main.ALERT_BADGE_WIDTH
    y2 = main.TILE_HEIGHT - main.ALERT_BADGE_MARGIN
    y1 = y2 - main.ALERT_BADGE_HEIGHT
    return x1, y1, x2, y2


def _badge_interior_has_black_pixel(frame: np.ndarray) -> bool:
    """badge 테두리(1px)를 제외한 내부에 검정 글자 픽셀이 있는지 확인한다."""
    x1, y1, x2, y2 = _badge_bounds()
    interior = frame[y1 + 3 : y2 - 3, x1 + 3 : x2 - 3]
    return bool(np.any(np.all(interior == 0, axis=-1)))


def _camera_alert_update(
    current_level: RiskLevel,
    state_started_at: float = 0.0,
    camera_id: str = "cam_01",
) -> CameraAlertUpdate:
    return CameraAlertUpdate(
        camera_id=camera_id,
        previous_level=RiskLevel.NORMAL,
        current_level=current_level,
        changed=False,
        state_started_at=state_started_at,
    )


# ---------------------------------------------------------------------------
# Badge
# ---------------------------------------------------------------------------


def test_draw_alert_badge_draws_white_rectangle_for_normal() -> None:
    frame = _blank_tile()
    main._draw_alert_badge(frame=frame, risk_level=RiskLevel.NORMAL)

    x1, y1, _, _ = _badge_bounds()
    assert tuple(frame[y1 + 3, x1 + 3]) == (255, 255, 255)


def test_draw_alert_badge_normal_has_no_text() -> None:
    frame = _blank_tile()
    main._draw_alert_badge(frame=frame, risk_level=RiskLevel.NORMAL)

    assert _badge_interior_has_black_pixel(frame) is False


def test_draw_alert_badge_warning_shows_text() -> None:
    frame = _blank_tile()
    main._draw_alert_badge(frame=frame, risk_level=RiskLevel.WARNING)

    assert _badge_interior_has_black_pixel(frame) is True


def test_draw_alert_badge_critical_shows_text() -> None:
    frame = _blank_tile()
    main._draw_alert_badge(frame=frame, risk_level=RiskLevel.CRITICAL)

    assert _badge_interior_has_black_pixel(frame) is True


def test_draw_alert_badge_keeps_fixed_size_across_levels() -> None:
    """badge 좌표는 risk_level과 무관하게 항상 같은 크기/위치여야 한다.

    사각형 테두리(1px 검정)가 정확히 (x1,y1)-(x2,y2) 위에 그려지므로,
    "흰색 채움" 확인은 테두리 바로 안쪽 지점에서 해야 한다.
    """
    x1, y1, x2, y2 = _badge_bounds()

    for risk_level in (RiskLevel.NORMAL, RiskLevel.WARNING, RiskLevel.CRITICAL):
        frame = _blank_tile()
        main._draw_alert_badge(frame=frame, risk_level=risk_level)

        assert tuple(frame[y1 + 2, x1 + 2]) == (255, 255, 255)
        assert tuple(frame[y1 + 2, x2 - 3]) == (255, 255, 255)
        assert tuple(frame[y2 - 3, x1 + 2]) == (255, 255, 255)


# ---------------------------------------------------------------------------
# Border
# ---------------------------------------------------------------------------


def test_draw_alert_overlay_normal_has_no_border() -> None:
    frame = _blank_tile()
    main._draw_alert_overlay(
        frame=frame,
        camera_alert_update=_camera_alert_update(RiskLevel.NORMAL),
        now=0.0,
    )

    assert tuple(frame[0, main.TILE_WIDTH // 2]) == (0, 0, 0)


def test_draw_alert_overlay_warning_border_is_yellow() -> None:
    frame = _blank_tile()
    update = _camera_alert_update(RiskLevel.WARNING, state_started_at=0.0)
    main._draw_alert_overlay(frame=frame, camera_alert_update=update, now=0.0)

    assert tuple(frame[0, main.TILE_WIDTH // 2]) == main.ALERT_BORDER_COLOR_WARNING


def test_draw_alert_overlay_critical_border_is_red() -> None:
    frame = _blank_tile()
    update = _camera_alert_update(RiskLevel.CRITICAL, state_started_at=0.0)
    main._draw_alert_overlay(frame=frame, camera_alert_update=update, now=0.0)

    assert tuple(frame[0, main.TILE_WIDTH // 2]) == main.ALERT_BORDER_COLOR_CRITICAL


def test_alert_border_thickness_matches_constant() -> None:
    frame = _blank_tile()
    main._draw_alert_border(frame=frame, color=main.ALERT_BORDER_COLOR_CRITICAL)

    mid_x = main.TILE_WIDTH // 2
    top_half = frame[: main.TILE_HEIGHT // 2, mid_x, :]
    painted_row_count = int(
        np.sum(np.all(top_half == main.ALERT_BORDER_COLOR_CRITICAL, axis=-1))
    )
    # cv2 선 굵기는 반올림에 따라 1px 정도 오차가 날 수 있어 범위로 확인한다.
    assert abs(painted_row_count - main.ALERT_BORDER_THICKNESS) <= 1


# ---------------------------------------------------------------------------
# Blink
# ---------------------------------------------------------------------------


def test_is_alert_border_visible_right_at_state_start() -> None:
    assert main._is_alert_border_visible(state_started_at=10.0, now=10.0) is True


def test_is_alert_border_visible_off_after_half_interval() -> None:
    assert main._is_alert_border_visible(state_started_at=10.0, now=10.5) is False


def test_is_alert_border_visible_on_after_full_interval() -> None:
    assert main._is_alert_border_visible(state_started_at=10.0, now=11.0) is True


def test_is_alert_border_visible_is_independent_per_state_started_at() -> None:
    """같은 now라도 state_started_at이 다르면 blink phase가 독립적이어야 한다."""
    now = 10.5

    assert main._is_alert_border_visible(state_started_at=10.0, now=now) is False
    assert main._is_alert_border_visible(state_started_at=10.3, now=now) is True


# ---------------------------------------------------------------------------
# Camera 상태별 노출 여부
# ---------------------------------------------------------------------------


def test_read_display_frame_camera_off_has_no_alert_badge(tmp_path) -> None:
    camera_view = main.CameraViewState(
        video_source=VideoSource(source=0, camera_id="cam_01"),
        processor=_FakeProcessor(None),
        enabled=False,
    )

    frame = main._read_display_frame(
        camera_view, RiskLogger(log_dir=tmp_path), EventManager(), now=0.0
    )

    x1, y1, _, _ = _badge_bounds()
    assert tuple(frame[y1 + 3, x1 + 3]) != (255, 255, 255)


def test_read_display_frame_no_processed_frame_has_no_alert_badge(tmp_path) -> None:
    camera_view = main.CameraViewState(
        video_source=VideoSource(source=0, camera_id="cam_01"),
        processor=_FakeProcessor(None),
        enabled=True,
    )

    frame = main._read_display_frame(
        camera_view, RiskLogger(log_dir=tmp_path), EventManager(), now=0.0
    )

    x1, y1, _, _ = _badge_bounds()
    assert tuple(frame[y1 + 3, x1 + 3]) != (255, 255, 255)


def test_read_display_frame_shows_empty_badge_when_normal(tmp_path) -> None:
    camera_view = main.CameraViewState(
        video_source=VideoSource(source=0, camera_id="cam_01"),
        processor=_FakeProcessor(
            _FakeProcessedFrame(
                frame_packet=_frame_packet(),
                rendered_frame=np.zeros((100, 100, 3), dtype=np.uint8),
            )
        ),
        enabled=True,
    )

    frame = main._read_display_frame(
        camera_view, RiskLogger(log_dir=tmp_path), EventManager(), now=0.0
    )

    x1, y1, _, _ = _badge_bounds()
    assert tuple(frame[y1 + 3, x1 + 3]) == (255, 255, 255)
    assert _badge_interior_has_black_pixel(frame) is False


# ---------------------------------------------------------------------------
# Multi-camera
# ---------------------------------------------------------------------------


def test_read_display_frame_two_cameras_show_independent_alert_states(tmp_path) -> None:
    """cam_01 WARNING과 cam_03 CRITICAL이 동시에, 서로 영향 없이 표시돼야 한다."""
    event_manager = EventManager()
    risk_logger = RiskLogger(log_dir=tmp_path)

    cam_01_view = main.CameraViewState(
        video_source=VideoSource(source=0, camera_id="cam_01"),
        processor=_FakeProcessor(
            _FakeProcessedFrame(
                frame_packet=_frame_packet("cam_01"),
                rendered_frame=np.zeros((100, 100, 3), dtype=np.uint8),
                zone_result=_zone_result_with_one_pair("cam_01"),
                risk_assessments=[_assessment(RiskLevel.WARNING, "cam_01")],
            )
        ),
        enabled=True,
    )
    cam_03_view = main.CameraViewState(
        video_source=VideoSource(source=0, camera_id="cam_03"),
        processor=_FakeProcessor(
            _FakeProcessedFrame(
                frame_packet=_frame_packet("cam_03"),
                rendered_frame=np.zeros((100, 100, 3), dtype=np.uint8),
                zone_result=_zone_result_with_one_pair("cam_03"),
                risk_assessments=[_assessment(RiskLevel.CRITICAL, "cam_03")],
            )
        ),
        enabled=True,
    )

    cam_01_frame = main._read_display_frame(cam_01_view, risk_logger, event_manager, now=0.0)
    cam_01_frame_snapshot = cam_01_frame.copy()

    cam_03_frame = main._read_display_frame(cam_03_view, risk_logger, event_manager, now=0.0)

    mid_x = main.TILE_WIDTH // 2
    assert tuple(cam_01_frame[0, mid_x]) == main.ALERT_BORDER_COLOR_WARNING
    assert tuple(cam_03_frame[0, mid_x]) == main.ALERT_BORDER_COLOR_CRITICAL

    # cam_03 처리가 이미 반환된 cam_01 frame 배열을 건드리지 않아야 한다.
    assert np.array_equal(cam_01_frame, cam_01_frame_snapshot)
