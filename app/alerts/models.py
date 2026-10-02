"""Alert 계층에서 사용하는 camera 단위 상태 데이터 모델."""

from __future__ import annotations

from dataclasses import dataclass

from app.risk.models import RiskLevel


@dataclass(slots=True)
class CameraAlertState:
    """EventManager가 camera_id별로 유지하는 현재 Alert 상태.

    risk_level: hold 정책이 반영된, 그 카메라의 현재 대표 위험등급.
    state_started_at: risk_level이 지금 값으로 바뀐 시각(time.monotonic 기준).
                       hold 시간이 지났는지 판단하는 기준점으로 쓰인다.
    """

    camera_id: str
    risk_level: RiskLevel
    state_started_at: float


@dataclass(frozen=True, slots=True)
class CameraAlertUpdate:
    """EventManager.update() 한 번의 호출 결과.

    UI/Sound 계층은 이 객체의 changed와 current_level만으로
    "지금 바로 반응해야 하는 변화인지"를 판단할 수 있다.
    """

    camera_id: str
    previous_level: RiskLevel
    current_level: RiskLevel
    changed: bool
