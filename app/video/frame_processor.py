"""프레임 단위로 영상 입력과 탐지기를 연결하는 모듈."""

from dataclasses import dataclass
import time

import cv2
import numpy as np

from app.video.video_source import VideoSource, FramePacket
from app.inference.detector import Detector, Detection


@dataclass
class ProcessedFrame:
    # frame_packet.frame은 원본 프레임이다.
    # rendered_frame은 bbox/metric을 그린 화면 표시용 프레임이며, 렌더링이 꺼져 있으면 None이다.
    frame_packet: FramePacket
    detections: list[Detection]
    inference_latency_ms: float
    rendered_frame: np.ndarray | None


class FrameProcessor:
    def __init__(
        self,
        video_source: VideoSource,
        detector: Detector,
        draw_bbox: bool = True,
        draw_metrics: bool = True,
    ):
        self.video_source = video_source
        self.detector = detector
        self.draw_bbox = draw_bbox
        self.draw_metrics = draw_metrics
        self.processing_fps = 0.0

    def process_next(self) -> ProcessedFrame | None:
        frame_packet = self.video_source.read()

        if frame_packet is None:
            return None

        # detector에는 원본 프레임을 그대로 넘긴다.
        # bbox 좌표 계약은 원본 영상 기준이므로 렌더링용 copy와 섞이면 안 된다.
        original_frame = frame_packet.frame

        inference_start = time.perf_counter()
        detections = self.detector.detect(original_frame)
        inference_end = time.perf_counter()

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
            # 원본 FramePacket을 보존하기 위해 화면 표시가 필요할 때만 복사본에 그린다.
            rendered_frame = original_frame.copy()

            if self.draw_bbox:
                self._draw_detections(
                    frame=rendered_frame,
                    detections=detections,
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
            inference_latency_ms=inference_latency_ms,
            rendered_frame=rendered_frame,
        )

    def _draw_detections(
        self,
        frame: np.ndarray,
        detections: list[Detection],
    ) -> None:
        frame_height, frame_width = frame.shape[:2]

        for detection in detections:
            x1, y1, x2, y2 = detection.bbox

            # detector 출력 bbox는 원본 좌표계 기준이지만, 모델 후처리나 float 반올림으로
            # 프레임 경계를 조금 벗어날 수 있어 OpenCV drawing 전에 화면 안으로 제한한다.
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
