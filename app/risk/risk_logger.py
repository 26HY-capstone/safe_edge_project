"""Tracking → Zone → Risk 판정 결과를 프레임 단위로 JSONL 파일에 기록하는 모듈.

운영용 Event Log, Alert, Event State/Deduplication/Cooldown과는 무관하며,
1차 프로토타입에서 Risk 판정 결과(NORMAL/WARNING/CRITICAL 전부)를 사람이 직접
눈으로 검증할 수 있도록 남기는 디버깅/검증용 로그다. 작업자 또는 설비 중
하나만 탐지된 프레임도 risk_level 없는 row로 남겨, 그 프레임에 무엇이
탐지됐는지 빠짐없이 확인할 수 있게 한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from app.risk.models import RiskAssessment
from app.zones.models import EquipmentZoneInfo, WorkerZoneInfo, ZoneFrameResult

# 프로젝트 루트(safe_edge_project) 기준 logs/ 디렉터리.
DEFAULT_LOG_DIR = Path(__file__).resolve().parents[2] / "logs"


class RiskLogger:
    """카메라별 risk_log_{camera_id}.jsonl 파일에 Risk 판정 결과를 기록한다.

    실행 시작 시 prepare()로 로그 디렉터리를 만들고 이번 실행에서 쓸 카메라들의
    로그 파일을 빈 파일로 초기화한다. 이후 매 프레임 log()를 호출하면 그 카메라의
    파일에만 한 줄씩 append되며, truncate는 다시 일어나지 않는다.
    """

    def __init__(self, log_dir: Path = DEFAULT_LOG_DIR) -> None:
        self._log_dir = log_dir

    def prepare(self, camera_ids: list[str]) -> None:
        """애플리케이션 실행 시작 시 한 번만 호출한다.

        logs/ 디렉터리가 없으면 생성하고, camera_ids에 해당하는 로그 파일을
        빈 파일로 초기화(truncate)한다. 프레임 처리 중 호출되는 log()는 이 메서드가
        만든 파일에 append만 하므로, 여기서 다루지 않은 다른 카메라의 기존 로그
        파일은 건드리지 않는다.
        """
        self._log_dir.mkdir(parents=True, exist_ok=True)

        for camera_id in camera_ids:
            self._log_path(camera_id).write_text("", encoding="utf-8")

    def log(
        self,
        zone_result: ZoneFrameResult,
        risk_assessments: list[RiskAssessment],
    ) -> None:
        """한 프레임의 Risk 판정 결과를 JSONL로 한 줄씩 append한다.

        RiskEngine은 workers x equipments 전체 조합에 대해 RiskAssessment를
        만들기 때문에, risk_assessments가 비어 있다는 건 작업자 또는 설비 중
        하나가 그 프레임에 전혀 없었다는 뜻이다(둘 다 있었다면 반드시 조합이
        생긴다). 이 경우에도 "그 프레임에 작업자/설비가 있었다"는 사실 자체는
        검증에 필요하므로, risk_level 없는 단독 row로 남긴다. 둘 다 없으면
        기록할 내용이 없으므로 아무 것도 쓰지 않는다.
        """
        if risk_assessments:
            workers_by_id = {worker.person_id: worker for worker in zone_result.workers}
            equipments_by_id = {
                equipment.equipment_id: equipment for equipment in zone_result.equipments
            }
            rows = [
                _risk_assessment_to_row(
                    assessment=assessment,
                    worker=workers_by_id[assessment.person_id],
                    equipment=equipments_by_id[assessment.equipment_id],
                )
                for assessment in risk_assessments
            ]
        elif zone_result.workers:
            # equipments가 비어 있다는 뜻(그렇지 않다면 위에서 조합이 생겼을 것).
            rows = [_worker_only_row(zone_result, worker) for worker in zone_result.workers]
        elif zone_result.equipments:
            rows = [
                _equipment_only_row(zone_result, equipment)
                for equipment in zone_result.equipments
            ]
        else:
            return

        self._append_rows(camera_id=zone_result.camera_id, rows=rows)

    def _append_rows(self, camera_id: str, rows: list[dict[str, object]]) -> None:
        """row 목록을 camera_id에 대응하는 파일에 JSONL로 append한다."""
        log_path = self._log_path(camera_id)
        with log_path.open("a", encoding="utf-8") as log_file:
            for row in rows:
                log_file.write(json.dumps(row, ensure_ascii=False))
                log_file.write("\n")

    def _log_path(self, camera_id: str) -> Path:
        """camera_id에 대응하는 risk_log_{camera_id}.jsonl 경로를 계산한다.

        camera_id는 app/main.py에서 화면 슬롯 순서 기준으로 항상 cam_01~04 형태로
        고정되어 들어오므로(파일명으로 쓸 수 없는 문자가 섞일 수 없음) 별도 sanitize는
        하지 않는다.
        """
        return self._log_dir / f"risk_log_{camera_id}.jsonl"


def is_frame_risk_log_enabled(config_path: Path) -> bool:
    """system.yaml에서 프레임 단위 검증 로그 활성화 여부를 읽는다."""
    with config_path.open("r", encoding="utf-8") as config_file:
        try:
            config = yaml.safe_load(config_file) or {}
        except yaml.YAMLError as exc:
            raise ValueError(f"invalid system config: {config_path}") from exc

    enabled = config.get("diagnostics", {}).get(
        "frame_risk_log_enabled", False
    )
    if not isinstance(enabled, bool):
        raise ValueError("diagnostics.frame_risk_log_enabled must be boolean")
    return enabled


def _risk_assessment_to_row(
    assessment: RiskAssessment,
    worker: WorkerZoneInfo,
    equipment: EquipmentZoneInfo,
) -> dict[str, object]:
    """RiskAssessment 하나를 JSONL 한 줄에 대응하는 dict로 변환한다.

    Enum(EquipmentType/RiskLevel)은 .name으로 문자열화하고, tuple 좌표는
    list로 바꿔 JSON 직렬화가 가능한 형태로 만든다.
    """
    return {
        "timestamp": assessment.timestamp,
        "frame_index": assessment.frame_index,
        "camera_id": assessment.camera_id,
        "person_id": assessment.person_id,
        "person_bottom_center": list(worker.bottom_center),
        "equipment_id": assessment.equipment_id,
        "equipment_type": assessment.equipment_type.name,
        "equipment_is_active": equipment.is_active,
        "equipment_bbox": list(equipment.equipment_bbox),
        "warning_zone": list(equipment.warning_zone),
        "critical_zone": list(equipment.critical_zone),
        "risk_level": assessment.risk_level.name,
    }


def _worker_only_row(
    zone_result: ZoneFrameResult,
    worker: WorkerZoneInfo,
) -> dict[str, object]:
    """같은 프레임에 설비가 없어 조합이 생기지 않은 작업자 하나를 기록한다.

    risk_level을 계산할 대상 설비가 없으므로 equipment_*와 risk_level은 모두
    null로 남긴다.
    """
    return {
        "timestamp": zone_result.timestamp,
        "frame_index": zone_result.frame_index,
        "camera_id": zone_result.camera_id,
        "person_id": worker.person_id,
        "person_bottom_center": list(worker.bottom_center),
        "equipment_id": None,
        "equipment_type": None,
        "equipment_is_active": None,
        "equipment_bbox": None,
        "warning_zone": None,
        "critical_zone": None,
        "risk_level": None,
    }


def _equipment_only_row(
    zone_result: ZoneFrameResult,
    equipment: EquipmentZoneInfo,
) -> dict[str, object]:
    """같은 프레임에 작업자가 없어 조합이 생기지 않은 설비 하나를 기록한다.

    risk_level을 계산할 대상 작업자가 없으므로 person_*와 risk_level은 모두
    null로 남긴다.
    """
    return {
        "timestamp": zone_result.timestamp,
        "frame_index": zone_result.frame_index,
        "camera_id": zone_result.camera_id,
        "person_id": None,
        "person_bottom_center": None,
        "equipment_id": equipment.equipment_id,
        "equipment_type": equipment.equipment_type.name,
        "equipment_is_active": equipment.is_active,
        "equipment_bbox": list(equipment.equipment_bbox),
        "warning_zone": list(equipment.warning_zone),
        "critical_zone": list(equipment.critical_zone),
        "risk_level": None,
    }
