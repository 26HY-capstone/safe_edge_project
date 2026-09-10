from dataclasses import dataclass, replace
import time

import cv2

from app.video.video_source import VideoSource, FramePacket
from app.inference.detector import Detector, Detection


@dataclass
class ProcessedFrame:
    frame_packet: FramePacket
    detections: list[Detection]
    inference_latency_ms: float


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

        frame = frame_packet.frame.copy()

        inference_start = time.perf_counter()
        detections = self.detector.detect(frame)
        inference_end = time.perf_counter()

        inference_latency_ms = (inference_end - inference_start) * 1000.0

        if inference_latency_ms > 0:
            self.processing_fps = 1000.0 / inference_latency_ms
        else:
            self.processing_fps = 0.0

        if self.draw_bbox:
            self._draw_detections(frame, detections)

        if self.draw_metrics:
            self._draw_metrics(
                frame,
                frame_packet.fps,
                inference_latency_ms,
                self.processing_fps,
            )

        rendered_packet = replace(frame_packet, frame=frame)

        return ProcessedFrame(
            frame_packet=rendered_packet,
            detections=detections,
            inference_latency_ms=inference_latency_ms,
        )

    def _draw_detections(
        self,
        frame,
        detections: list[Detection],
    ) -> None:
        frame_height, frame_width = frame.shape[:2]

        for detection in detections:
            x1, y1, x2, y2 = detection.bbox

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

            label = f"{detection.class_name} {detection.confidence:.2f}"

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
        frame,
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
