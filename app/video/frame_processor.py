"""프레임 단위로 영상 입력과 탐지기를 연결하는 모듈."""

from dataclasses import dataclass, field
import time
from typing import Protocol

import cv2
import numpy as np

from app.video.video_source import FramePacket
from app.inference.detector import BBox, Detector, Detection
from app.tracking.tracker import Tracker, TrackedObject
from app.tracking.trajectory import MotionSummary, TrajectoryAnalyzer
from app.ppe.matcher import PPEMatcher, WorkerPPEStatus
from app.risk.ppe_risk import PPERiskAssessment, PPERiskEvaluator
from app.risk.models import RiskAssessment
from app.risk.risk_engine import RiskEngine
from app.zones.models import ZoneFrameResult
from app.zones.zone_manager import ZoneManager

# Zone 사각형은 채우지 않고 테두리만 얇게 그려 실제 영상과 객체를 가리지 않는다.
# BGR 순서: Critical Zone은 빨강, Warning Zone은 노랑으로 서로 명확히 구분한다.
_ZONE_THICKNESS = 2
_CRITICAL_ZONE_COLOR = (0, 0, 255)
_WARNING_ZONE_COLOR = (0, 255, 255)


class FrameSource(Protocol):
    """FrameProcessor가 영상 입력 객체에 기대하는 최소 인터페이스."""

    def read(self) -> FramePacket | None:
        """다음 프레임 패킷을 반환하고, 더 이상 읽을 프레임이 없으면 None을 반환한다."""
        ...


@dataclass
class ProcessedFrame:
    # frame_packet.frame: 원본 프레임.
    # rendered_frame: bbox/metric을 그린 화면 표시용 프레임. 렌더링 비활성 시 None.
    # zone_result: ZoneManager가 계산한 이번 프레임의 Zone 분석 결과.
    #              Zone은 파이프라인의 필수 단계이므로 항상 생성된다.
    # risk_assessments: RiskEngine이 계산한 작업자-설비 조합별 위험 판정 결과.
    #                   작업자 또는 설비가 없으면 빈 리스트가 된다.
    frame_packet: FramePacket
    detections: list[Detection]
    tracked_objects: list[TrackedObject]
    motion_summaries: dict[int, MotionSummary]
    inference_latency_ms: float
    rendered_frame: np.ndarray | None
    zone_result: ZoneFrameResult
    risk_assessments: list[RiskAssessment]
    ppe_statuses: dict[int, WorkerPPEStatus] = field(default_factory=dict)
    ppe_risk_assessments: list[PPERiskAssessment] = field(default_factory=list)


class FrameProcessor:
    def __init__(
        self,
        video_source: FrameSource,
        detector: Detector,
        zone_manager: ZoneManager,
        risk_engine: RiskEngine,
        tracker: Tracker | None = None,
        trajectory_analyzer: TrajectoryAnalyzer | None = None,
        draw_bbox: bool = True,
        draw_metrics: bool = True,
        ppe_matcher: PPEMatcher | None = None,
        ppe_risk_evaluator: PPERiskEvaluator | None = None,
    ):
        # zone_manager: tracked_objects와 frame_packet으로 작업자/설비의
        # Zone(Warning/Critical) 진입 여부를 계산한다.
        # risk_engine: zone_manager의 결과를 받아 작업자-설비 조합별 위험등급을 판정한다.
        # 실행 파이프라인은 Tracking → Zone → Risk 순서를 항상 거친다.
        self.ppe_risk_evaluator = ppe_risk_evaluator or PPERiskEvaluator()
        self.ppe_matcher = ppe_matcher if ppe_matcher is not None else PPEMatcher()
        self.video_source = video_source
        self.detector = detector
        self.tracker = tracker
        self.trajectory_analyzer = trajectory_analyzer
        self.zone_manager = zone_manager
        self.risk_engine = risk_engine
        self.draw_bbox = draw_bbox
        self.draw_metrics = draw_metrics
        self.processing_fps = 0.0

    def process_next(self) -> ProcessedFrame | None:
        frame_packet = self.video_source.read()

        if frame_packet is None:
            return None

        # detector 입력: 원본 프레임.
        # bbox 좌표 계약: 원본 영상 기준. 렌더링용 copy와 분리.
        original_frame = frame_packet.frame

        inference_start = time.perf_counter()
        detections = self.detector.detect(original_frame)
        inference_end = time.perf_counter()

        tracked_objects = []
        if self.tracker is not None:
            tracked_objects = self.tracker.update(
                detections=detections,
                frame_packet=frame_packet,
            )

        ppe_statuses = self.ppe_matcher.match(detections, tracked_objects)

        motion_summaries = {}
        if self.trajectory_analyzer is not None:
            motion_summaries = self.trajectory_analyzer.update(
                tracked_objects=tracked_objects,
                frame_packet=frame_packet,
            )

        # Zone 분석: tracked_objects와 frame_packet으로 작업자/설비의 Zone 진입 여부를 계산한다.
        zone_result = self.zone_manager.process(
            frame_packet=frame_packet,
            tracked_objects=tracked_objects,
        )

        # Risk 판정: 위에서 계산한 Zone 분석 결과로 작업자-설비 조합별 위험등급을 계산한다.
        # 작업자 또는 설비가 없으면 빈 리스트를 반환한다.
        risk_assessments = self.risk_engine.evaluate(zone_result)
        ppe_risk_assessments = self.ppe_risk_evaluator.evaluate(ppe_statuses, frame_packet)

        inference_latency_ms = (
            inference_end - inference_start
        ) * 1000.0

        if inference_latency_ms > 0:
            self.processing_fps = (
                1000.0 / inference_latency_ms
            )
        else:
            self.processing_fps = 0.0

        rendered_frame = None

        if self.draw_bbox or self.draw_metrics:
            # OpenCV drawing 함수는 입력 배열을 직접 수정한다.
            # 화면 표시용 복사본은 원본 FramePacket 보존을 위한 별도 버퍼다.
            rendered_frame = original_frame.copy()

            # 렌더링 순서: bbox 테두리 → Critical Zone → Warning Zone → ID/class label → Metrics.
            # bbox를 두껍게 먼저 그린 뒤 그 위에 얇은 Critical Zone을 겹쳐 그리면
            # 두 영역의 색이 함께 보여 서로 구분할 수 있다. label은 Zone 선 위에 그려서
            # Warning Zone처럼 bbox보다 넓은 사각형에 텍스트가 가려지지 않게 한다.
            if self.draw_bbox:
                self._draw_bboxes(
                    frame=rendered_frame,
                    detections=detections,
                    tracked_objects=tracked_objects,
                )

            # Zone 시각화는 Tracking → Zone → Risk 파이프라인의 결과를 그대로 보여주는
            # 것이므로 draw_bbox와 별개로, rendered_frame을 만드는 한 항상 표시한다.
            self._draw_zones(frame=rendered_frame, zone_result=zone_result)

            if self.draw_bbox:
                self._draw_bbox_labels(
                    frame=rendered_frame,
                    detections=detections,
                    tracked_objects=tracked_objects,
                )

            if self.draw_metrics:
                self._draw_metrics(
                    frame=rendered_frame,
                    source_fps=frame_packet.fps,
                    inference_latency_ms=inference_latency_ms,
                    processing_fps=self.processing_fps,
                )

        return ProcessedFrame(
            frame_packet=frame_packet,
            detections=detections,
            tracked_objects=tracked_objects,
            motion_summaries=motion_summaries,
            inference_latency_ms=inference_latency_ms,
            rendered_frame=rendered_frame,
            zone_result=zone_result,
            risk_assessments=risk_assessments,
            ppe_statuses=ppe_statuses,
            ppe_risk_assessments=ppe_risk_assessments,
        )

    def _draw_bboxes(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        tracked_objects: list[TrackedObject],
    ) -> None:
        """Detection 또는 Tracking bbox의 테두리를 그린다.

        Tracking 결과가 있으면 bbox 테두리만 먼저 그리고, ID/class label은
        Zone 사각형을 그린 뒤 _draw_bbox_labels()에서 그린다. Tracking 결과가 없으면
        (이 경우 ZoneManager도 설비 Zone을 만들지 않으므로) 기존처럼 bbox와 label을
        한 번에 그린다.
        """
        if tracked_objects:
            self._draw_tracked_object_boxes(frame=frame, tracked_objects=tracked_objects)
            return

        self._draw_detections(frame=frame, detections=detections)

    def _draw_bbox_labels(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        tracked_objects: list[TrackedObject],
    ) -> None:
        """Tracking bbox의 ID/class label을 Zone 사각형 위에 그린다.

        Tracking 결과가 없으면 _draw_bboxes()에서 이미 label까지 그렸으므로
        여기서는 아무 것도 하지 않는다.
        """
        if tracked_objects:
            self._draw_tracked_object_labels(frame=frame, tracked_objects=tracked_objects)

    def _draw_detections(
        self,
        frame: np.ndarray,
        detections: list[Detection],
    ) -> None:
        frame_height, frame_width = frame.shape[:2]
        font_scale, thickness = _overlay_style(frame_height)

        for detection in detections:
            # detector 출력 bbox는 원본 좌표계 기준이다.
            # 모델 후처리나 float 반올림으로 생길 수 있는 경계 초과를 drawing 전에 제한한다.
            x1, y1, x2, y2 = _clamp_bbox(detection.bbox, frame_width, frame_height)

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                thickness,
            )

            label = (
                f"{detection.class_name} "
                f"{detection.confidence:.2f}"
            )

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, int(24 * font_scale))),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 255, 0),
                thickness,
            )

    def _draw_tracked_object_boxes(
        self,
        frame: np.ndarray,
        tracked_objects: list[TrackedObject],
    ) -> None:
        """Tracking bbox의 테두리만 그린다.

        이후 그 위에 얇게 그려질 Critical Zone과 구분되도록, 기존 오버레이
        두께보다 두껍게 그린다(Critical Zone은 equipment bbox와 좌표가 같을 수 있어,
        두 선의 두께가 다르면 두 영역의 색이 동시에 보여 구분하기 쉽다).
        """
        frame_height, frame_width = frame.shape[:2]
        _, base_thickness = _overlay_style(frame_height)
        box_thickness = base_thickness + 2

        for tracked_object in tracked_objects:
            # tracking 결과 bbox도 원본 좌표계 기준이다.
            # 렌더링 단계에서만 화면 크기 안쪽으로 제한한다.
            x1, y1, x2, y2 = _clamp_bbox(tracked_object.bbox, frame_width, frame_height)

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 200, 0),
                box_thickness,
            )

    def _draw_tracked_object_labels(
        self,
        frame: np.ndarray,
        tracked_objects: list[TrackedObject],
    ) -> None:
        """Tracking ID/class/confidence label을 그린다."""
        frame_height, frame_width = frame.shape[:2]
        font_scale, thickness = _overlay_style(frame_height)

        for tracked_object in tracked_objects:
            x1, y1, _, _ = _clamp_bbox(tracked_object.bbox, frame_width, frame_height)

            label = (
                f"ID {tracked_object.track_id} "
                f"{tracked_object.class_name} "
                f"{tracked_object.confidence:.2f}"
            )

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, int(24 * font_scale))),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 200, 0),
                thickness,
            )

    def _draw_zones(
        self,
        frame: np.ndarray,
        zone_result: ZoneFrameResult,
    ) -> None:
        """ZoneManager가 계산한 각 설비의 Warning/Critical Zone 테두리를 그린다.

        좌표는 새로 계산하지 않고 equipment.warning_zone / equipment.critical_zone
        값을 그대로 사용한다. 화면 밖으로 나가는 좌표는 그리기 직전에만 clamp하며,
        원본 EquipmentZoneInfo 값 자체는 변경하지 않는다.
        """
        frame_height, frame_width = frame.shape[:2]

        for equipment in zone_result.equipments:
            # Critical Zone을 먼저 그린다. equipment bbox와 좌표가 같을 수 있어
            # 두꺼운 Tracking bbox 위에 얇은 선으로 겹쳐 그리면 두 영역을 함께 구분할 수 있다.
            critical_x1, critical_y1, critical_x2, critical_y2 = _clamp_bbox(
                equipment.critical_zone, frame_width, frame_height
            )
            cv2.rectangle(
                frame,
                (critical_x1, critical_y1),
                (critical_x2, critical_y2),
                _CRITICAL_ZONE_COLOR,
                _ZONE_THICKNESS,
            )

            # Warning Zone은 Critical Zone보다 바깥쪽에 위치하므로
            # 마지막에 그려도 다른 사각형과 크게 겹치지 않는다.
            warning_x1, warning_y1, warning_x2, warning_y2 = _clamp_bbox(
                equipment.warning_zone, frame_width, frame_height
            )
            cv2.rectangle(
                frame,
                (warning_x1, warning_y1),
                (warning_x2, warning_y2),
                _WARNING_ZONE_COLOR,
                _ZONE_THICKNESS,
            )

    def _draw_metrics(
        self,
        frame: np.ndarray,
        source_fps: float,
        inference_latency_ms: float,
        processing_fps: float,
    ) -> None:
        font_scale, thickness = _overlay_style(frame.shape[0])
        line_height = int(34 * font_scale)
        margin = int(20 * font_scale)

        cv2.putText(
            frame,
            f"Source FPS: {source_fps:.1f}",
            (margin, margin + line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
        )

        cv2.putText(
            frame,
            f"Inference FPS: {processing_fps:.1f}",
            (margin, margin + line_height * 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
        )

        cv2.putText(
            frame,
            f"Latency: {inference_latency_ms:.1f} ms",
            (margin, margin + line_height * 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
        )


def _overlay_style(frame_height: int) -> tuple[float, int]:
    """원본 프레임에 먼저 그린 뒤 축소해도 라벨이 읽히도록 해상도 비례 스타일을 계산한다."""
    font_scale = max(0.7, frame_height / 600.0)
    thickness = max(2, round(frame_height / 360.0))
    return font_scale, thickness


def _clamp_bbox(
    bbox: BBox,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    """bbox 좌표를 그리기 직전에 frame 크기 안쪽으로 clamp한다.

    detector/tracker 출력이나 Zone 계산 결과는 원본 영상 좌표계 기준이라
    프레임 경계를 벗어날 수 있다. 이 함수는 화면 표시용으로만 좌표를 제한하며,
    호출부에 전달된 원본 bbox/Zone 값 자체는 변경하지 않는다.
    """
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(x1), frame_width - 1))
    y1 = max(0, min(int(y1), frame_height - 1))
    x2 = max(0, min(int(x2), frame_width - 1))
    y2 = max(0, min(int(y2), frame_height - 1))
    return x1, y1, x2, y2
