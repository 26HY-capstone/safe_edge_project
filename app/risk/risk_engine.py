"""Zone 분석 결과를 작업자별 위험 판정으로 변환하는 모듈."""

from __future__ import annotations

from app.risk.models import RiskAssessment, RiskLevel
from app.risk.risk_rules import determine_risk_level, determine_zone_risk_level
from app.zones.models import EquipmentZoneInfo, WorkerZoneInfo, Zone, ZoneFrameResult


class RiskEngine:
    def evaluate(self, zone_result: ZoneFrameResult) -> list[RiskAssessment]:
        assessments: list[RiskAssessment] = []
        for worker in zone_result.workers:
            for equipment in zone_result.equipments:
                assessments.append(
                    self._evaluate_equipment(
                        zone_result=zone_result,
                        worker=worker,
                        equipment=equipment,
                    )
                )
            for zone in zone_result.zones:
                assessments.append(
                    self._evaluate_static_zone(
                        zone_result=zone_result,
                        worker=worker,
                        zone=zone,
                    )
                )
        return assessments

    @staticmethod
    def _evaluate_equipment(
        zone_result: ZoneFrameResult,
        worker: WorkerZoneInfo,
        equipment: EquipmentZoneInfo,
    ) -> RiskAssessment:
        risk_level = determine_risk_level(worker, equipment)
        zone_id = None
        if risk_level != RiskLevel.NORMAL:
            zone_id = f"{equipment.equipment_type.value}:{equipment.equipment_id}"
        return RiskAssessment(
            risk_level=risk_level,
            risk_type="equipment_proximity",
            camera_id=zone_result.camera_id,
            worker_track_id=worker.person_id,
            equipment_track_id=equipment.equipment_id,
            equipment_type=equipment.equipment_type,
            equipment_state=equipment.state,
            distance=None,
            zone_id=zone_id,
            ppe_status=None,
            reason=(
                f"worker is {risk_level.value} near "
                f"{equipment.equipment_type.value}"
            ),
            timestamp=zone_result.timestamp,
            frame_index=zone_result.frame_index,
        )

    @staticmethod
    def _evaluate_static_zone(
        zone_result: ZoneFrameResult,
        worker: WorkerZoneInfo,
        zone: Zone,
    ) -> RiskAssessment:
        risk_level = determine_zone_risk_level(worker, zone)
        return RiskAssessment(
            risk_level=risk_level,
            risk_type="zone_intrusion",
            camera_id=zone_result.camera_id,
            worker_track_id=worker.person_id,
            equipment_track_id=zone.equipment_track_id,
            equipment_type=None,
            equipment_state=None,
            distance=None,
            zone_id=zone.zone_id,
            ppe_status=None,
            reason=f"worker is {risk_level.value} in zone {zone.zone_id}",
            timestamp=zone_result.timestamp,
            frame_index=zone_result.frame_index,
        )
