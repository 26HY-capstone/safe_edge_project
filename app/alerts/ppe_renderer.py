"""축소된 카메라 타일에 작업자별 PPE 상태를 읽기 쉬운 크기로 표시한다."""

import cv2
import numpy as np

from app.risk.ppe_risk import PPERiskAssessment
from app.tracking.tracker import TrackedObject


def draw_ppe_overlay(
    frame: np.ndarray, tracked_objects: list[TrackedObject],
    assessments: list[PPERiskAssessment], source_size: tuple[int, int],
) -> None:
    """원본 bbox를 타일 좌표로 변환해 상태와 미착용 강조 테두리를 그린다."""
    height, width = frame.shape[:2]
    sx, sy = width / source_size[0], height / source_size[1]
    workers = {obj.track_id: obj for obj in tracked_objects if obj.class_name == "person"}
    for assessment in assessments:
        worker = workers.get(assessment.person_id)
        if worker is None:
            continue
        x1, y1, x2, y2 = worker.bbox
        x = max(0, min(round(x1 * sx), max(0, width - 270)))
        y = max(16, min(round(y1 * sy), max(16, height - 20 * (len(assessment.required_classes) + 1))))
        warning = bool(assessment.missing_classes)
        color = (0, 255, 255) if warning else (0, 220, 0)
        if warning:
            cv2.rectangle(frame, (round(x1 * sx), round(y1 * sy)),
                          (round(x2 * sx), round(y2 * sy)), color, 2)
        lines = [f"Person #{assessment.person_id} PPE" + (" WARNING" if warning else "")]
        for name in assessment.required_classes:
            state = "MISSING" if name in assessment.missing_classes else (
                "CHECK" if name in assessment.pending_classes else "OK"
            )
            lines.append(f"{name}: {state}")
        panel_width = max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
                          for line in lines)
        cv2.rectangle(frame, (x, y - 15),
                      (min(width - 1, x + panel_width + 6), y + (len(lines) - 1) * 20 + 4),
                      (25, 25, 25), -1)
        for index, line in enumerate(lines):
            origin = (x + 3, y + index * 20)
            cv2.putText(frame, line, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(frame, line, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
