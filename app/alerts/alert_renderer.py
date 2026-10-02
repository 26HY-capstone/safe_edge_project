"""Alert 상태를 camera tile 위에 badge/border로 그리는 모듈.

EventManager가 만든 CameraAlertUpdate를 받아 어떻게 화면에 표시할지 담당한다. 
Alert 판정(hold/escalation/de-escalation)은 app/alerts/event_manager.py의
책임이고, 이 모듈은 그 결과를 그대로 받아 그리기만 하는 stateless 함수들로 구성된다.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.alerts.models import CameraAlertUpdate
from app.risk.models import RiskLevel

BLINK_INTERVAL_SECONDS = 0.5
ALERT_BORDER_THICKNESS = 8
WARNING_COLOR = (0, 255, 255)  # BGR: Yellow
CRITICAL_COLOR = (0, 0, 255)  # BGR: Red
ALERT_BADGE_WIDTH = 120
ALERT_BADGE_HEIGHT = 34
ALERT_BADGE_MARGIN = 16


def draw_alert_overlay(
    frame: np.ndarray,
    camera_alert_update: CameraAlertUpdate,
    now: float,
) -> None:
    """Alert badge와 border를 frame에 함께 그린다.

    badge는 NORMAL/WARNING/CRITICAL 모든 상태에서 그리고(크기는 항상 고정),
    border는 WARNING/CRITICAL이면서 blink가 켜진 순간에만 그린다.
    """
    _draw_alert_badge(frame=frame, risk_level=camera_alert_update.current_level)

    if camera_alert_update.current_level == RiskLevel.NORMAL:
        return

    border_color = (
        CRITICAL_COLOR
        if camera_alert_update.current_level == RiskLevel.CRITICAL
        else WARNING_COLOR
    )
    if _is_alert_border_visible(
        state_started_at=camera_alert_update.state_started_at,
        now=now,
    ):
        _draw_alert_border(frame=frame, color=border_color)


def _draw_alert_badge(frame: np.ndarray, risk_level: RiskLevel) -> None:
    """frame 우하단에 흰색 badge를 그린다. WARNING/CRITICAL이면 등급 글자를 넣는다.

    badge 좌표는 risk_level과 무관하게 항상 같은 상수로 계산해, 상태가 바뀌어도
    badge 크기와 위치가 흔들리지 않는다.
    """
    height, width = frame.shape[:2]
    x2 = width - ALERT_BADGE_MARGIN
    x1 = x2 - ALERT_BADGE_WIDTH
    y2 = height - ALERT_BADGE_MARGIN
    y1 = y2 - ALERT_BADGE_HEIGHT

    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 1)

    if risk_level == RiskLevel.NORMAL:
        # NORMAL은 글자 없는 빈 badge로 둔다.
        return

    cv2.putText(
        frame,
        risk_level.name,
        (x1 + 11, y2 - 11),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        2,
    )


def _draw_alert_border(frame: np.ndarray, color: tuple[int, int, int]) -> None:
    """frame 전체 외곽에 Alert border를 그린다.

    cv2.rectangle은 경계선 위에 두께만큼 양쪽으로 걸쳐 그리는데, 경계선을
    이미지 가장자리에 그대로 두면 바깥쪽 절반이 이미지 밖으로 잘려 실제로는
    두께가 반밖에 안 보인다. 경계선을 두께의 절반만큼 안쪽으로 들여 그려서
    지정한 두께가 그대로 보이게 한다.
    """
    height, width = frame.shape[:2]
    inset = ALERT_BORDER_THICKNESS // 2
    cv2.rectangle(
        frame,
        (inset, inset),
        (width - 1 - inset, height - 1 - inset),
        color,
        ALERT_BORDER_THICKNESS,
    )


def _is_alert_border_visible(state_started_at: float, now: float) -> bool:
    """state_started_at 기준 경과 시간으로 blink phase를 계산해 지금 border를
    보일지 결정한다. camera마다 state_started_at이 다르므로 같은 now를
    받아도 카메라별로 독립적인 phase가 나온다.
    """
    elapsed = max(0.0, now - state_started_at)
    phase = int(elapsed / BLINK_INTERVAL_SECONDS)
    return phase % 2 == 0
