"""Zone 모듈과 Risk 모듈 사이에서 사용하는 데이터 모델."""

from dataclasses import dataclass
from enum import Enum

from app.inference.detector import BBox, Point


class EquipmentType(Enum):
    """위험구역을 생성하는 설비의 종류"""

    FORKLIFT = "forklift"
    CONVEYOR = "conveyor"
    ROBOT_ARM = "robot_arm"


@dataclass(frozen=True, slots=True)
class WorkerZoneInfo:
    """
    현재 프레임에서 Tracking된 작업자 한 명의 공간 정보를 저장한다.

    Zone 모듈은 작업자의 위험 여부를 판단하지 않고,
    Risk 모듈에서 판단에 사용할 위치 정보만 전달한다.
    """

    # Tracking 과정에서 부여된 작업자의 고유 ID
    person_id: int

    # 현재 프레임에서 탐지된 작업자의 Bounding Box
    # 형식: (x1, y1, x2, y2)
    person_bbox: BBox

    # 작업자 Bounding Box의 하단 중앙 좌표
    # Risk 모듈에서 Warning/Critical Zone 진입 여부를 판단할 기준점으로 사용한다.
    bottom_center: Point


@dataclass(frozen=True, slots=True)
class EquipmentZoneInfo:
    """
    현재 프레임에서 Tracking된 설비 하나와
    해당 설비를 기준으로 생성된 위험구역 정보를 저장한다.
    """

    # Tracking 과정에서 부여된 설비의 고유 ID
    equipment_id: int

    # 설비의 종류
    # 예: FORKLIFT, CONVEYOR, ROBOT_ARM
    equipment_type: EquipmentType

    # 현재 프레임에서 탐지된 설비의 Bounding Box
    # 형식: (x1, y1, x2, y2)
    equipment_bbox: BBox

    # 설비 주변의 외부 주의구역
    # Risk 모듈은 작업자의 bottom_center가 이 영역에 포함되는지 검사한다.
    warning_zone: BBox

    # 설비와 직접적으로 가까운 위험구역
    # 초기 프로토타입에서는 equipment_bbox와 동일한 영역을 사용할 수 있다.
    critical_zone: BBox


@dataclass(frozen=True, slots=True)
class ZoneFrameResult:
    """
    특정 프레임 하나에 대한 Zone 모듈의 전체 분석 결과를 저장한다.

    WorkerZoneInfo 목록과 EquipmentZoneInfo 목록을 하나로 묶어
    Risk 모듈에 전달하는 최상위 데이터 객체이다.
    """

    # 해당 프레임의 시간 정보
    # 현재 프로젝트에서는 VideoSource의 timestamp 값을 그대로 사용한다. 추후 변경 가능.
    timestamp: float

    # 현재 카메라 세션에서의 프레임 순번
    frame_index: int

    # 해당 프레임을 제공한 카메라의 식별자
    # cam 하나만을 이용하는 개발 초기 단계에서는 "cam_01"을 사용.
    camera_id: str

    # 현재 프레임에서 Tracking된 작업자 정보 목록
    workers: list[WorkerZoneInfo]

    # 현재 프레임에서 Tracking된 설비 및 위험구역 정보 목록
    equipments: list[EquipmentZoneInfo]