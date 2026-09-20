"""프레임 단위로 영상 입력과 탐지기를 연결하는 모듈."""

from dataclasses import dataclass
import time

import cv2
import numpy as np

from app.alerts.alert_manager import AlertManager
from app.alerts.event_manager import EventManager, RiskEvent
from app.inference.detector import Detector, Detection
from app.risk.models import RiskAssessment, RiskLevel
from app.risk.risk_engine import RiskEngine
from app.tracking.tracker import Tracker, TrackedObject
from app.tracking.trajectory import MotionSummary, TrajectoryAnalyzer
from app.video.video_source import VideoSource, FramePacket
from app.zones.models import ZoneFrameResult
from app.zones.zone_manager import ZoneManager


@dataclass
class ProcessedFrame:
    # frame_packet.frame: 원본 프레임.
    # rendered_frame: bbox/metric을 그린 화면 표시용 프레임. 렌더링 비활성 시 None.
    frame_packet: FramePacket
    detections: list[Detection]
    tracked_objects: list[TrackedObject]
    motion_summaries: dict[int, MotionSummary]
    inference_latency_ms: float
    rendered_frame: np.ndarray | None
    zone_result: ZoneFrameResult | None
    risk_assessments: list[RiskAssessment]
    risk_events: list[RiskEvent]


class FrameProcessor:
    def __init__(
        self,
        video_source: VideoSource,
        detector: Detector,
        tracker: Tracker | None = None,
        trajectory_analyzer: TrajectoryAnalyzer | None = None,
        zone_manager: ZoneManager | None = None,
        risk_engine: RiskEngine | None = None,
        event_manager: EventManager | None = None,
        alert_manager: AlertManager | None = None,
        draw_bbox: bool = True,
        draw_metrics: bool = True,
        draw_zones: bool = True,
        draw_risks: bool = True,
    ):
        self.video_source = video_source
        self.detector = detector
        self.tracker = tracker
        self.trajectory_analyzer = trajectory_analyzer
        self.zone_manager = zone_manager
        self.risk_engine = risk_engine
        self.event_manager = event_manager
        self.alert_manager = alert_manager
        self.draw_bbox = draw_bbox
        self.draw_metrics = draw_metrics
        self.draw_zones = draw_zones
        self.draw_risks = draw_risks
        self.processing_fps = 0.0
        self._last_stream_epoch: int | None = None
        self._last_frame_index: int | None = None

    def process_next(self) -> ProcessedFrame | None:
        frame_packet = self.video_source.read()

        if frame_packet is None:
            return None

        self._reset_if_stream_restarted(frame_packet)

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

        motion_summaries = {}
        if self.trajectory_analyzer is not None:
            motion_summaries = self.trajectory_analyzer.update(
                tracked_objects=tracked_objects,
                frame_packet=frame_packet,
            )

        zone_result = None
        if self.zone_manager is not None:
            zone_result = self.zone_manager.process(
                frame_packet=frame_packet,
                tracked_objects=tracked_objects,
            )

        risk_assessments: list[RiskAssessment] = []
        if self.risk_engine is not None and zone_result is not None:
            risk_assessments = self.risk_engine.evaluate(zone_result)

        risk_events: list[RiskEvent] = []
        if self.event_manager is not None:
            risk_events = self.event_manager.update(
                risk_assessments,
                timestamp=frame_packet.timestamp,
                frame_index=frame_packet.frame_index,
            )
        if self.alert_manager is not None and risk_events:
            self.alert_manager.handle(risk_events)

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

        should_render = (
            self.draw_bbox
            or self.draw_metrics
            or (self.draw_zones and zone_result is not None)
            or (self.draw_risks and bool(risk_assessments))
        )
        if should_render:
            # OpenCV drawing 함수는 입력 배열을 직접 수정한다.
            # 화면 표시용 복사본은 원본 FramePacket 보존을 위한 별도 버퍼다.
            rendered_frame = original_frame.copy()

            if self.draw_bbox:
                self._draw_detection_results(
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

            if self.draw_zones and zone_result is not None:
                self._draw_zones(rendered_frame, zone_result)

            if self.draw_risks and risk_assessments:
                self._draw_risks(rendered_frame, risk_assessments)

        return ProcessedFrame(
            frame_packet=frame_packet,
            detections=detections,
            tracked_objects=tracked_objects,
            motion_summaries=motion_summaries,
            inference_latency_ms=inference_latency_ms,
            rendered_frame=rendered_frame,
            zone_result=zone_result,
            risk_assessments=risk_assessments,
            risk_events=risk_events,
        )

    def _reset_if_stream_restarted(self, frame_packet: FramePacket) -> None:
        restarted = (
            self._last_stream_epoch is not None
            and frame_packet.stream_epoch != self._last_stream_epoch
        )
        frame_index_reversed = (
            self._last_frame_index is not None
            and frame_packet.frame_index < self._last_frame_index
        )
        if restarted or frame_index_reversed:
            if self.tracker is not None:
                self.tracker.reset()
            if self.trajectory_analyzer is not None:
                self.trajectory_analyzer.reset()
            if self.zone_manager is not None:
                self.zone_manager.reset()
            if self.event_manager is not None:
                self.event_manager.reset()

        self._last_stream_epoch = frame_packet.stream_epoch
        self._last_frame_index = frame_packet.frame_index

    def _draw_detection_results(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        tracked_objects: list[TrackedObject],
    ) -> None:
        if tracked_objects:
            self._draw_tracked_objects(frame=frame, tracked_objects=tracked_objects)
            return

        self._draw_detections(frame=frame, detections=detections)

    def _draw_detections(
        self,
        frame: np.ndarray,
        detections: list[Detection],
    ) -> None:
        frame_height, frame_width = frame.shape[:2]

        for detection in detections:
            x1, y1, x2, y2 = detection.bbox

            # detector 출력 bbox는 원본 좌표계 기준이다.
            # 모델 후처리나 float 반올림으로 생길 수 있는 경계 초과를 drawing 전에 제한한다.
            x1 = max(0, min(int(x1), frame_width - 1))
            y1 = max(0, min(int(y1), frame_height - 1))
            x2 = max(0, min(int(x2), frame_width - 1))
            y2 = max(0, min(int(y2), frame_height - 1))

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            label = (
                f"{detection.class_name} "
                f"{detection.confidence:.2f}"
            )

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

    def _draw_tracked_objects(
        self,
        frame: np.ndarray,
        tracked_objects: list[TrackedObject],
    ) -> None:
        frame_height, frame_width = frame.shape[:2]

        for tracked_object in tracked_objects:
            x1, y1, x2, y2 = tracked_object.bbox

            # tracking 결과 bbox도 원본 좌표계 기준이다.
            # 렌더링 단계에서만 화면 크기 안쪽으로 제한한다.
            x1 = max(0, min(int(x1), frame_width - 1))
            y1 = max(0, min(int(y1), frame_height - 1))
            x2 = max(0, min(int(x2), frame_width - 1))
            y2 = max(0, min(int(y2), frame_height - 1))

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 200, 0),
                2,
            )

            label = (
                f"ID {tracked_object.track_id} "
                f"{tracked_object.class_name} "
                f"{tracked_object.confidence:.2f}"
            )

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 200, 0),
                2,
            )

    def _draw_metrics(
        self,
        frame: np.ndarray,
        source_fps: float,
        inference_latency_ms: float,
        processing_fps: float,
    ) -> None:
        cv2.putText(
            frame,
            f"Source FPS: {source_fps:.1f}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"Inference FPS: {processing_fps:.1f}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"Latency: {inference_latency_ms:.1f} ms",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

    @staticmethod
    def _draw_zones(frame: np.ndarray, zone_result: ZoneFrameResult) -> None:
        for zone in zone_result.zones:
            FrameProcessor._draw_polygon(frame, zone.polygon, (0, 215, 255))
        for equipment in zone_result.equipments:
            FrameProcessor._draw_polygon(
                frame,
                equipment.warning_zone,
                (0, 215, 255),
            )
            FrameProcessor._draw_polygon(
                frame,
                equipment.critical_zone,
                (0, 0, 255),
            )

    @staticmethod
    def _draw_risks(
        frame: np.ndarray,
        assessments: list[RiskAssessment],
    ) -> None:
        visible = [
            assessment
            for assessment in assessments
            if assessment.risk_level != RiskLevel.NORMAL
        ]
        for index, assessment in enumerate(visible[:5]):
            cv2.putText(
                frame,
                (
                    f"{assessment.risk_level.name} "
                    f"worker={assessment.worker_track_id} "
                    f"zone={assessment.zone_id or '-'}"
                ),
                (20, 125 + index * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2,
            )

    @staticmethod
    def _draw_polygon(
        frame: np.ndarray,
        polygon: tuple[tuple[float, float], ...],
        color: tuple[int, int, int],
    ) -> None:
        points = np.asarray(polygon, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [points], isClosed=True, color=color, thickness=2)
