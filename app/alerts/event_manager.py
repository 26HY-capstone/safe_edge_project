"""camera별 Risk 결과를 집계하고 Alert 상태를 관리하는 모듈.

Risk 계층(RiskAssessment)과 향후 UI/Sound 계층 사이에 새로 추가되는 소비 계층이다.
이미 계산된 RiskAssessment를 읽어 camera 단위 상태를 만든다.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from app.alerts.models import CameraAlertState, CameraAlertUpdate
from app.risk.models import RiskAssessment, RiskLevel

# hold 정책의 "최소 유지 시간".
# 한 번 전환된 Alert 상태를 최소 이 시간만큼 유지한 뒤에야 더 낮은 위험도로
# 내려가도록 하는 값이다. 위험도 상승은 이 값과 무관하게 항상 즉시 반영된다.
ALERT_HOLD_SECONDS = 3.0

# RiskLevel 간 우선순위를 명시적으로 정의한다. (NORMAL < WARNING < CRITICAL)
_RISK_LEVEL_PRIORITY: dict[RiskLevel, int] = {
    RiskLevel.NORMAL: 0,
    RiskLevel.WARNING: 1,
    RiskLevel.CRITICAL: 2,
}


def get_highest_risk_level(risk_assessments: list[RiskAssessment]) -> RiskLevel:
    """한 카메라의 RiskAssessment 목록에서 가장 높은 RiskLevel을 계산한다."""
    if not risk_assessments:
        return RiskLevel.NORMAL

    return max(
        (assessment.risk_level for assessment in risk_assessments),
        key=lambda risk_level: _RISK_LEVEL_PRIORITY[risk_level],
    )


class EventManager:
    """camera_id별 Alert 상태(RiskLevel + hold 시각)를 독립적으로 관리한다.

    - 위험도가 오르면 hold와 무관하게 즉시 반영한다.
    - 같은 RiskLevel이 계속 관측되면 상태를 바꾸지 않는다. 즉 hold 시작 시각을
      매 frame 다시 기록하지 않는다.
    - 위험도가 내려가면 현재 상태가 ALERT_HOLD_SECONDS 이상 유지된 뒤에만
      반영한다. 
    - 레벨이 실제로 바뀌는 순간에는 새 상태의 시작 시각을 다시 기록한다. 
      하락으로 도달한 새 레벨도 동일하게 3초 hold가 적용한다.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        # clock: 실제 운영에서는 time.monotonic. frame_index/FPS로 시간을 계산하면
        # 영상 재생 속도나 frame skip에 따라 흔들릴 수 있어 쓰지 않는다. 테스트에서는
        # 원하는 시각으로 직접 이동시킬 수 있는 fake clock을 주입해 real sleep 없이
        # hold 동작을 검증한다.
        self._clock = clock
        self._states: dict[str, CameraAlertState] = {}

    def update(
        self,
        camera_id: str,
        risk_assessments: list[RiskAssessment],
    ) -> CameraAlertUpdate:
        """한 프레임의 RiskAssessment로 camera_id의 Alert 상태를 갱신한다.

        risk_assessments를 집계해 이번 프레임의 observed RiskLevel을 구하고,
        그 camera_id의 기존 상태(처음 보는 camera_id면 NORMAL로 간주)와 비교해
        escalation/동일 레벨 지속/de-escalation 중 하나로 처리한다.
        """
        observed_level = get_highest_risk_level(risk_assessments)
        now = self._clock()

        state = self._states.get(camera_id)
        if state is None:
            # 처음 보는 camera_id는 NORMAL에서 시작한 것으로 간주한다.
            state = CameraAlertState(
                camera_id=camera_id,
                risk_level=RiskLevel.NORMAL,
                state_started_at=now,
            )
            self._states[camera_id] = state

        previous_level = state.risk_level
        new_level, changed = _resolve_next_level(
            current_level=state.risk_level,
            observed_level=observed_level,
            state_started_at=state.state_started_at,
            now=now,
        )

        if changed:
            # RiskLevel이 실제로 바뀌는 순간에만 시작 시각을 다시 기록한다.
            # 동일 레벨이 계속 관측되는 경우는 이 분기를 타지 않으므로
            # hold 타이머가 매 frame 리셋되지 않는다.
            state.risk_level = new_level
            state.state_started_at = now

        return CameraAlertUpdate(
            camera_id=camera_id,
            previous_level=previous_level,
            current_level=state.risk_level,
            changed=changed,
            state_started_at=state.state_started_at,
        )


def _resolve_next_level(
    current_level: RiskLevel,
    observed_level: RiskLevel,
    state_started_at: float,
    now: float,
) -> tuple[RiskLevel, bool]:
    """현재 상태와 이번 프레임의 observed_level을 비교해 다음 상태를 결정한다.

    반환값은 (다음 RiskLevel, 변경 여부)다. 하락 시 observed_level이 현재보다
    여러 단계 낮아도(예: CRITICAL -> NORMAL) 중간 단계를 거치지 않고 observed_level로 바로 반영한다.
    """
    current_priority = _RISK_LEVEL_PRIORITY[current_level]
    observed_priority = _RISK_LEVEL_PRIORITY[observed_level]

    if observed_priority > current_priority:
        # 위험도 상승: hold와 무관하게 항상 즉시 반영한다.
        return observed_level, True

    if observed_priority == current_priority:
        # 동일 위험도 지속: 상태도 시작 시각도 그대로 둔다.
        return current_level, False

    # 위험도 하락: 현재 상태가 최소 hold 시간만큼 유지된 뒤에만 반영한다.
    if now - state_started_at >= ALERT_HOLD_SECONDS:
        return observed_level, True

    return current_level, False
