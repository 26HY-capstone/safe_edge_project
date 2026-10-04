"""app/alerts/alert_renderer.py의 badge/border/blink 렌더링 동작을 검증하는 테스트."""

from __future__ import annotations

import numpy as np

from app.alerts import alert_renderer
from app.alerts.models import CameraAlertUpdate
from app.risk.models import RiskLevel

# 실제 2x2 화면의 tile 크기와 같은 값을 써서 badge/border 좌표가 실제 환경과
# 동일하게 계산되는지 확인한다. alert_renderer 자체는 frame.shape로 계산하므로
# main.py의 TILE_WIDTH/TILE_HEIGHT에 의존하지 않는다.
_FRAME_WIDTH = 640
_FRAME_HEIGHT = 360


def _blank_frame() -> np.ndarray:
    return np.zeros((_FRAME_HEIGHT, _FRAME_WIDTH, 3), dtype=np.uint8)


def _badge_bounds() -> tuple[int, int, int, int]:
    x2 = _FRAME_WIDTH - alert_renderer.ALERT_BADGE_MARGIN
    x1 = x2 - alert_renderer.ALERT_BADGE_WIDTH
    y2 = _FRAME_HEIGHT - alert_renderer.ALERT_BADGE_MARGIN
    y1 = y2 - alert_renderer.ALERT_BADGE_HEIGHT
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
    frame = _blank_frame()
    alert_renderer._draw_alert_badge(frame=frame, risk_level=RiskLevel.NORMAL)

    x1, y1, _, _ = _badge_bounds()
    assert tuple(frame[y1 + 3, x1 + 3]) == (255, 255, 255)


def test_draw_alert_badge_normal_has_no_text() -> None:
    frame = _blank_frame()
    alert_renderer._draw_alert_badge(frame=frame, risk_level=RiskLevel.NORMAL)

    assert _badge_interior_has_black_pixel(frame) is False


def test_draw_alert_badge_warning_shows_text() -> None:
    frame = _blank_frame()
    alert_renderer._draw_alert_badge(frame=frame, risk_level=RiskLevel.WARNING)

    assert _badge_interior_has_black_pixel(frame) is True


def test_draw_alert_badge_critical_shows_text() -> None:
    frame = _blank_frame()
    alert_renderer._draw_alert_badge(frame=frame, risk_level=RiskLevel.CRITICAL)

    assert _badge_interior_has_black_pixel(frame) is True


def test_draw_alert_badge_keeps_fixed_size_across_levels() -> None:
    """badge 좌표는 risk_level과 무관하게 항상 같은 크기/위치여야 한다.

    사각형 테두리(1px 검정)가 정확히 (x1,y1)-(x2,y2) 위에 그려지므로,
    "흰색 채움" 확인은 테두리 바로 안쪽 지점에서 해야 한다.
    """
    x1, y1, x2, y2 = _badge_bounds()

    for risk_level in (RiskLevel.NORMAL, RiskLevel.WARNING, RiskLevel.CRITICAL):
        frame = _blank_frame()
        alert_renderer._draw_alert_badge(frame=frame, risk_level=risk_level)

        assert tuple(frame[y1 + 2, x1 + 2]) == (255, 255, 255)
        assert tuple(frame[y1 + 2, x2 - 3]) == (255, 255, 255)
        assert tuple(frame[y2 - 3, x1 + 2]) == (255, 255, 255)


# ---------------------------------------------------------------------------
# Border
# ---------------------------------------------------------------------------


def test_draw_alert_overlay_normal_has_no_border() -> None:
    frame = _blank_frame()
    alert_renderer.draw_alert_overlay(
        frame=frame,
        camera_alert_update=_camera_alert_update(RiskLevel.NORMAL),
        now=0.0,
    )

    assert tuple(frame[0, _FRAME_WIDTH // 2]) == (0, 0, 0)


def test_draw_alert_overlay_warning_border_is_yellow() -> None:
    frame = _blank_frame()
    update = _camera_alert_update(RiskLevel.WARNING, state_started_at=0.0)
    alert_renderer.draw_alert_overlay(frame=frame, camera_alert_update=update, now=0.0)

    assert tuple(frame[0, _FRAME_WIDTH // 2]) == alert_renderer.WARNING_COLOR


def test_draw_alert_overlay_critical_border_is_red() -> None:
    frame = _blank_frame()
    update = _camera_alert_update(RiskLevel.CRITICAL, state_started_at=0.0)
    alert_renderer.draw_alert_overlay(frame=frame, camera_alert_update=update, now=0.0)

    assert tuple(frame[0, _FRAME_WIDTH // 2]) == alert_renderer.CRITICAL_COLOR


def test_alert_border_thickness_matches_constant() -> None:
    frame = _blank_frame()
    alert_renderer._draw_alert_border(frame=frame, color=alert_renderer.CRITICAL_COLOR)

    mid_x = _FRAME_WIDTH // 2
    top_half = frame[: _FRAME_HEIGHT // 2, mid_x, :]
    painted_row_count = int(
        np.sum(np.all(top_half == alert_renderer.CRITICAL_COLOR, axis=-1))
    )
    # cv2 선 굵기는 반올림에 따라 1px 정도 오차가 날 수 있어 범위로 확인한다.
    assert abs(painted_row_count - alert_renderer.ALERT_BORDER_THICKNESS) <= 1


# ---------------------------------------------------------------------------
# Blink
# ---------------------------------------------------------------------------


def test_is_alert_border_visible_right_at_state_start() -> None:
    assert alert_renderer._is_alert_border_visible(state_started_at=10.0, now=10.0) is True


def test_is_alert_border_visible_off_after_half_interval() -> None:
    assert alert_renderer._is_alert_border_visible(state_started_at=10.0, now=10.5) is False


def test_is_alert_border_visible_on_after_full_interval() -> None:
    assert alert_renderer._is_alert_border_visible(state_started_at=10.0, now=11.0) is True


def test_is_alert_border_visible_is_independent_per_state_started_at() -> None:
    """같은 now라도 state_started_at이 다르면 blink phase가 독립적이어야 한다."""
    now = 10.5

    assert alert_renderer._is_alert_border_visible(state_started_at=10.0, now=now) is False
    assert alert_renderer._is_alert_border_visible(state_started_at=10.3, now=now) is True
