"""Standard(full-image) YOLOv9t ONNX inference runner.

1차 실험의 A(640)/B(960), 2차 실험의 E(v2 640) 모두 이 runner 하나로 실행한다
(모델 경로와 input_size를 CLI로 바꿀 뿐, 코드는 동일하다).

사용 예:
    python experiments/sahi/run_standard.py \\
        --model /path/to/yolov9t_640.onnx \\
        --images /path/to/test/images \\
        --input-size 640 \\
        --output experiments/sahi/outputs/A_standard_640
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import time

from experiments.sahi.utils import (
    build_image_record,
    build_onnx_detector,
    compute_latency_stats,
    draw_detections,
    list_image_paths,
    read_image_or_none,
    save_predictions,
    save_summary,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standard full-image YOLO inference runner")
    parser.add_argument("--model", required=True, type=Path, help="ONNX 모델 경로")
    parser.add_argument("--images", required=True, type=Path, help="이미지 디렉터리")
    parser.add_argument("--input-size", type=int, default=640, help="모델 입력 크기(정사각형)")
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.45)
    parser.add_argument("--device", default="cpu", help="cpu / cuda[:id] / coreml")
    parser.add_argument(
        "--no-cpu-fallback",
        action="store_true",
        help="지정한 device를 쓸 수 없을 때 CPU로 자동 전환하지 않고 바로 실패시킨다",
    )
    parser.add_argument("--output", required=True, type=Path, help="predictions.json 등을 저장할 디렉터리")
    parser.add_argument("--limit", type=int, default=None, help="앞에서 N개 이미지만 처리(smoke test용)")
    parser.add_argument("--save-visualizations", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    image_paths = list_image_paths(args.images)
    if args.limit is not None:
        image_paths = image_paths[: args.limit]

    detector = build_onnx_detector(
        model_path=args.model,
        input_size=args.input_size,
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold,
        device=args.device,
        allow_cpu_fallback=not args.no_cpu_fallback,
    )
    detector.warmup()

    records: list[dict] = []
    latencies_ms: list[float] = []
    vis_dir = args.output / "visualizations"

    for image_path in image_paths:
        image = read_image_or_none(image_path)
        if image is None:
            continue

        start = time.perf_counter()
        detections = detector.detect(image)
        latency_ms = (time.perf_counter() - start) * 1000.0
        latencies_ms.append(latency_ms)

        height, width = image.shape[:2]
        records.append(
            build_image_record(
                image_name=image_path.name,
                width=width,
                height=height,
                detections=detections,
                latency_ms=latency_ms,
            )
        )

        if args.save_visualizations:
            vis_dir.mkdir(parents=True, exist_ok=True)
            import cv2

            cv2.imwrite(str(vis_dir / image_path.name), draw_detections(image, detections))

    predictions_path = save_predictions(records, args.output)
    latency_stats = compute_latency_stats(latencies_ms)
    summary_path = save_summary(
        args.output,
        image_count=len(records),
        latency_stats=latency_stats,
        extra={
            "experiment": "standard",
            "model": str(args.model),
            "input_size": args.input_size,
            "confidence_threshold": args.confidence_threshold,
            "iou_threshold": args.iou_threshold,
        },
    )

    logger.info("saved predictions: %s", predictions_path)
    logger.info("saved summary: %s", summary_path)
    logger.info("images processed: %d / %d", len(records), len(image_paths))
    logger.info("latency(ms): %s", latency_stats)


if __name__ == "__main__":
    main()
