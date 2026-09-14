"""프레임 단위로 영상 입력과 탐지기를 연결하는 모듈."""

from dataclasses import dataclass
import time

import cv2
import numpy as np

from app.video.video_source import VideoSource, FramePacket
from app.inference.detector import Detector, Detection
from app.tracking.tracker import Tracker, TrackedObject


@dataclass
class ProcessedFrame:
    # frame_packet.frame: 원본 프레임.
    # rendered_frame: bbox/metric을 그린 화면 표시용 프레임. 렌더링 비활성 시 None.
    frame_packet: FramePacket
    detections: list[Detection]
    tracked_objects: list[TrackedObject]
    inference_latency_ms: float
    rendered_frame: np.ndarray | None


class FrameProcessor:
    def __init__(
        self,
        video_source: VideoSource,
        detector: Detector,
        tracker: Tracker | None = None,
        draw_bbox: bool = True,
        draw_metrics: bool = True,
    ):
        self.video_source = video_source
        self.detector = detector
        self.tracker = tracker
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

        return ProcessedFrame(
            frame_packet=frame_packet,
            detections=detections,
            tracked_objects=tracked_objects,
            inference_latency_ms=inference_latency_ms,
            rendered_frame=rendered_frame,
        )

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
