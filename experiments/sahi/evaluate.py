"""predictions.json과 YOLO format ground-truth label을 비교해 mAP류 지표를 계산한다.

A/B/C/D/E/F/G 모두 run_*.py가 만든 동일한 predictions.json 스키마를 쓰므로,
이 evaluate.py 하나로 모든 실험을 평가할 수 있다.

사용 예:
    python experiments/sahi/evaluate.py \\
        --predictions experiments/sahi/outputs/C_sahi_640/predictions.json \\
        --labels /path/to/test/labels \\
        --images /path/to/test/images
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from app.inference.model_manager import EXPECTED_PROTOTYPE_CLASSES
from experiments.sahi.metrics import GroundTruthBox, PredictionBox, evaluate_dataset
from experiments.sahi.utils import list_image_paths, load_predictions

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAHI 실험 predictions evaluator")
    parser.add_argument("--predictions", required=True, type=Path, help="predictions.json 경로")
    parser.add_argument("--labels", required=True, type=Path, help="YOLO format label(.txt) 디렉터리")
    parser.add_argument("--images", required=True, type=Path, help="평가 대상 이미지 디렉터리")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="metrics.json 저장 경로(기본값: predictions.json과 같은 디렉터리)",
    )
    return parser.parse_args()


def load_yolo_labels(
    label_path: Path,
    image_width: int,
    image_height: int,
) -> list[GroundTruthBox]:
    """YOLO format label 파일(class_id cx cy w h, 0~1 정규화)을 절대 픽셀 xyxy로 변환한다.

    label 파일이 없으면 그 이미지는 GT가 없는 것으로 본다(에러가 아니다). 한 줄이
    형식에 맞지 않으면 그 줄만 경고 로그를 남기고 건너뛴다(라벨 하나의 오류로
    전체 평가가 멈추지 않도록 한다).
    """
    if not label_path.is_file():
        return []

    boxes: list[GroundTruthBox] = []
    for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) != 5:
            logger.warning("malformed YOLO label line, skipping: %s:%d", label_path, line_number)
            continue

        try:
            class_id = int(float(parts[0]))
            cx, cy, w, h = (float(value) for value in parts[1:])
        except ValueError:
            logger.warning("non-numeric YOLO label line, skipping: %s:%d", label_path, line_number)
            continue

        x1 = (cx - w / 2.0) * image_width
        y1 = (cy - h / 2.0) * image_height
        x2 = (cx + w / 2.0) * image_width
        y2 = (cy + h / 2.0) * image_height
        boxes.append(GroundTruthBox(class_id=class_id, bbox_xyxy=(x1, y1, x2, y2)))

    return boxes


def _group_predictions_by_class(
    records_by_stem: dict[str, dict[str, Any]],
) -> dict[int, list[tuple[str, PredictionBox]]]:
    grouped: dict[int, list[tuple[str, PredictionBox]]] = {}
    for image_id, record in records_by_stem.items():
        for detection in record.get("detections", []):
            prediction = PredictionBox(
                class_id=int(detection["class_id"]),
                confidence=float(detection["confidence"]),
                bbox_xyxy=tuple(detection["bbox_xyxy"]),
            )
            grouped.setdefault(prediction.class_id, []).append((image_id, prediction))
    return grouped


def _group_ground_truths_by_class(
    ground_truths_by_image: dict[str, list[GroundTruthBox]],
) -> dict[int, list[tuple[str, GroundTruthBox]]]:
    grouped: dict[int, list[tuple[str, GroundTruthBox]]] = {}
    for image_id, boxes in ground_truths_by_image.items():
        for box in boxes:
            grouped.setdefault(box.class_id, []).append((image_id, box))
    return grouped


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    image_paths = list_image_paths(args.images)
    raw_records = load_predictions(args.predictions)
    records_by_stem = {Path(record["image"]).stem: record for record in raw_records}

    ground_truths_by_image: dict[str, list[GroundTruthBox]] = {}
    missing_predictions = 0

    for image_path in image_paths:
        stem = image_path.stem
        record = records_by_stem.get(stem)
        if record is None:
            # predictions.json에 이 이미지 항목이 없으면(예: 이미지 읽기 실패로 run_*.py가
            # 건너뛴 경우) detection 0개로 취급한다 — 전체 평가를 멈추지 않는다.
            missing_predictions += 1
            width, height = _read_image_size(image_path)
        else:
            width, height = record["width"], record["height"]

        label_path = args.labels / f"{stem}.txt"
        ground_truths_by_image[stem] = load_yolo_labels(label_path, width, height)

    if missing_predictions:
        logger.warning(
            "%d image(s) have no matching entry in predictions.json; treated as 0 detections",
            missing_predictions,
        )

    predictions_by_class = _group_predictions_by_class(
        {stem: records_by_stem[stem] for stem in records_by_stem if stem in ground_truths_by_image}
    )
    ground_truths_by_class = _group_ground_truths_by_class(ground_truths_by_image)

    result = evaluate_dataset(
        predictions_by_class=predictions_by_class,
        ground_truths_by_class=ground_truths_by_class,
        class_names=dict(EXPECTED_PROTOTYPE_CLASSES),
    )

    output_path = args.output or (args.predictions.parent / "metrics.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(result, output_file, ensure_ascii=False, indent=2)

    logger.info("saved metrics: %s", output_path)
    logger.info("overall: %s", result["overall"])


def _read_image_size(image_path: Path) -> tuple[int, int]:
    import cv2

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"failed to read image for size lookup: {image_path}")
    height, width = image.shape[:2]
    return width, height


if __name__ == "__main__":
    main()
