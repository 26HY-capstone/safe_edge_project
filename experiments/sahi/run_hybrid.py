"""Hybrid(Full-image + SAHI) inference runner.

1차 실험의 D(640), 2차 실험의 G(v2 640) 모두 이 runner 하나로 실행한다. 같은
이미지에 대해 standard full-image 추론과 SAHI sliced 추론을 모두 수행한 뒤,
두 결과를 원본 좌표 기준으로 합쳐 global NMS를 한 번만 적용한다. 큰 객체는
full-image 추론으로 안정적으로 잡고, 작은 PPE는 slice 추론으로 보완하는 것이 목적이다.

사용 예:
    python experiments/sahi/run_hybrid.py \\
        --model /path/to/yolov9t_640.onnx \\
        --images /path/to/test/images \\
        --slice-size 640 \\
        --overlap 0.2 \\
        --output experiments/sahi/outputs/D_hybrid_640
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import time

from experiments.sahi.slicing import global_nms, run_sliced_inference
from experiments.sahi.utils import (
    build_image_record,
    build_onnx_detector,
    compute_latency_stats,
    draw_detections,
    list_image_paths,
    load_slice_config_defaults,
    read_image_or_none,
    resolve_slice_settings,
    save_predictions,
    save_summary,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid (full-image + SAHI) YOLO inference runner")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.45, help="global NMS IoU threshold")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-cpu-fallback", action="store_true")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--slice-size", type=int, default=None)
    parser.add_argument("--slice-width", type=int, default=None)
    parser.add_argument("--slice-height", type=int, default=None)
    parser.add_argument("--overlap", type=float, default=None)
    parser.add_argument("--overlap-width", type=float, default=None)
    parser.add_argument("--overlap-height", type=float, default=None)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save-visualizations", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    slice_settings = resolve_slice_settings(
        slice_size=args.slice_size,
        slice_width=args.slice_width,
        slice_height=args.slice_height,
        overlap=args.overlap,
        overlap_width=args.overlap_width,
        overlap_height=args.overlap_height,
        config_defaults=load_slice_config_defaults(args.config),
    )

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

        start_full = time.perf_counter()
        full_image_detections = detector.detect(image)
        full_latency_ms = (time.perf_counter() - start_full) * 1000.0

        sliced_detections, slice_latency_ms = run_sliced_inference(
            detector=detector,
            image=image,
            slice_width=slice_settings.slice_width,
            slice_height=slice_settings.slice_height,
            overlap_width_ratio=slice_settings.overlap_width_ratio,
            overlap_height_ratio=slice_settings.overlap_height_ratio,
        )

        start_nms = time.perf_counter()
        final_detections = global_nms(
            [*full_image_detections, *sliced_detections],
            iou_threshold=args.iou_threshold,
        )
        nms_latency_ms = (time.perf_counter() - start_nms) * 1000.0
        latency_ms = full_latency_ms + slice_latency_ms + nms_latency_ms
        latencies_ms.append(latency_ms)

        height, width = image.shape[:2]
        records.append(
            build_image_record(
                image_name=image_path.name,
                width=width,
                height=height,
                detections=final_detections,
                latency_ms=latency_ms,
            )
        )

        if args.save_visualizations:
            vis_dir.mkdir(parents=True, exist_ok=True)
            import cv2

            cv2.imwrite(str(vis_dir / image_path.name), draw_detections(image, final_detections))

    predictions_path = save_predictions(records, args.output)
    latency_stats = compute_latency_stats(latencies_ms)
    summary_path = save_summary(
        args.output,
        image_count=len(records),
        latency_stats=latency_stats,
        extra={
            "experiment": "hybrid",
            "model": str(args.model),
            "input_size": args.input_size,
            "slice_width": slice_settings.slice_width,
            "slice_height": slice_settings.slice_height,
            "overlap_width_ratio": slice_settings.overlap_width_ratio,
            "overlap_height_ratio": slice_settings.overlap_height_ratio,
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
