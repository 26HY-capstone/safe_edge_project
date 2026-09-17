"""분석 결과를 종합해 최종 위험도를 계산하는 모듈."""

from __future__ import annotations

from app.risk.models import RiskAssessment
from app.risk.risk_rules import determine_risk_level
from app.zones.models import ZoneFrameResult


class RiskEngine:
    """
    한 프레임의 Zone 분석 결과를 받아 작업자-설비 조합별 위험도를 계산한다.

    개별 위험 판정 규칙은 risk_rules.py의 determine_risk_level()이 담당하며,
    RiskEngine은 한 프레임 안의 모든 작업자-설비 조합을 순회해 규칙 함수를
    호출하고 결과를 RiskAssessment로 모은다.
    """

    def evaluate(self, zone_result: ZoneFrameResult) -> list[RiskAssessment]:
        """
        ZoneFrameResult 한 프레임을 입력받아 모든 작업자-설비 조합에 대한
        RiskAssessment 목록을 생성한다.
        """

        assessments: list[RiskAssessment] = []

        # 현재 프레임의 모든 작업자와 설비 조합을 순회한다.
        for worker in zone_result.workers:
            for equipment in zone_result.equipments:
                # 개별 조합의 위험등급 판정.
                risk_level = determine_risk_level(worker, equipment)

                # ZoneFrameResult의 공통 정보와
                # 작업자·설비 식별 정보를 묶어 RiskAssessment를 생성한다.
                assessments.append(
                    RiskAssessment(
                        timestamp=zone_result.timestamp,
                        frame_index=zone_result.frame_index,
                        camera_id=zone_result.camera_id,
                        person_id=worker.person_id,
                        equipment_id=equipment.equipment_id,
                        equipment_type=equipment.equipment_type,
                        risk_level=risk_level,
                    )
                )

        return assessments
