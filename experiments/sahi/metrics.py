"""Ground-truth와 prediction을 비교해 mAP류 지표를 계산하는 순수 로직 모듈.

pycocotools 같은 외부 라이브러리 없이 COCO 스타일 101-point interpolation으로
AP를 직접 계산한다(이 프로젝트는 아직 실제 데이터셋이 로컬에 없어 pycocotools가
요구하는 COCO JSON 변환 과정 없이도 YOLO 라벨을 바로 쓸 수 있는 쪽을 택했다).
평가 로직을 evaluate.py의 CLI/파일 I/O와 분리해서, 실제 데이터 없이도 synthetic
GT/prediction으로 단위 테스트할 수 있게 만든다.

주의: pycocotools의 정확한 내부 구현(ignore 플래그, crowd 처리 등)을 그대로
재현하지는 않는다. "크기별 AP"는 COCO의 면적 기준(아래 SMALL/MEDIUM 상수)만
가져온 단순화된 근사치다. README에도 이 사실을 명시한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# COCO 공식 평가 기준의 bbox 면적(px^2) 경계값. 작은 PPE 객체 확인이 목적이므로
# 그대로 재사용한다. (참고: https://cocodataset.org/#detection-eval)
SMALL_AREA_MAX = 32 * 32
MEDIUM_AREA_MAX = 96 * 96

# mAP50-95: IoU 0.50부터 0.95까지 0.05 간격 10개 threshold의 평균.
IOU_THRESHOLDS_50_95: tuple[float, ...] = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))

BBoxXYXY = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class GroundTruthBox:
    """한 이미지 안의 ground-truth 객체 하나."""

    class_id: int
    bbox_xyxy: BBoxXYXY

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox_xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @property
    def size_bucket(self) -> str:
        """COCO 면적 기준으로 small/medium/large 중 하나를 반환한다."""
        area = self.area
        if area < SMALL_AREA_MAX:
            return "small"
        if area < MEDIUM_AREA_MAX:
            return "medium"
        return "large"


@dataclass(frozen=True, slots=True)
class PredictionBox:
    """한 이미지 안의 prediction 하나."""

    class_id: int
    confidence: float
    bbox_xyxy: BBoxXYXY


def box_iou(box_a: BBoxXYXY, box_b: BBoxXYXY) -> float:
    """두 xyxy bbox의 IoU를 계산한다(단일 쌍, numpy 배열 불필요)."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    intersection = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def _match_predictions_to_ground_truths(
    sorted_predictions: list[tuple[str, PredictionBox]],
    gt_by_image: dict[str, list[GroundTruthBox]],
    iou_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """confidence 내림차순으로 정렬된 prediction들을 image별 GT에 그리디하게 매칭한다.

    반환값은 (true_positive_flags, false_positive_flags)다. 같은 GT를 두 prediction이
    동시에 차지할 수 없다 — confidence가 더 높은 prediction이 먼저 그 GT를 가져간다.
    """
    matched = {image_id: [False] * len(gts) for image_id, gts in gt_by_image.items()}
    true_positive = np.zeros(len(sorted_predictions), dtype=np.float64)
    false_positive = np.zeros(len(sorted_predictions), dtype=np.float64)

    for index, (image_id, prediction) in enumerate(sorted_predictions):
        candidates = gt_by_image.get(image_id, [])
        best_iou = 0.0
        best_gt_index = -1
        for gt_index, gt in enumerate(candidates):
            if matched[image_id][gt_index]:
                continue
            iou = box_iou(prediction.bbox_xyxy, gt.bbox_xyxy)
            if iou > best_iou:
                best_iou = iou
                best_gt_index = gt_index

        if best_gt_index >= 0 and best_iou >= iou_threshold:
            matched[image_id][best_gt_index] = True
            true_positive[index] = 1.0
        else:
            false_positive[index] = 1.0

    return true_positive, false_positive


def _interpolated_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """COCO 스타일 101-point interpolation으로 precision-recall curve를 적분한다."""
    if recalls.size == 0:
        return 0.0

    recall_levels = np.linspace(0.0, 1.0, 101)
    interpolated = np.zeros_like(recall_levels)
    for index, recall_level in enumerate(recall_levels):
        mask = recalls >= recall_level
        interpolated[index] = precisions[mask].max() if np.any(mask) else 0.0
    return float(np.mean(interpolated))


def compute_ap_for_class(
    predictions: list[tuple[str, PredictionBox]],
    ground_truths: list[tuple[str, GroundTruthBox]],
    iou_threshold: float,
    size_bucket: str | None = None,
) -> float | None:
    """한 class, 한 IoU threshold에서 AP를 계산한다.

    size_bucket이 주어지면(예: "small") 그 크기의 GT만 평가 대상으로 삼는다
    (AP_small/medium/large 계산용). 그 class/버킷에 GT가 전혀 없으면 정의되지
    않으므로 None을 반환한다(평균을 낼 때 이 class는 제외해야 한다는 신호).
    """
    filtered_gts = [
        (image_id, gt)
        for image_id, gt in ground_truths
        if size_bucket is None or gt.size_bucket == size_bucket
    ]
    if not filtered_gts:
        return None

    gt_by_image: dict[str, list[GroundTruthBox]] = {}
    for image_id, gt in filtered_gts:
        gt_by_image.setdefault(image_id, []).append(gt)
    total_gt_count = len(filtered_gts)

    sorted_predictions = sorted(predictions, key=lambda item: item[1].confidence, reverse=True)
    if not sorted_predictions:
        return 0.0

    true_positive, false_positive = _match_predictions_to_ground_truths(
        sorted_predictions=sorted_predictions,
        gt_by_image=gt_by_image,
        iou_threshold=iou_threshold,
    )
    cumulative_tp = np.cumsum(true_positive)
    cumulative_fp = np.cumsum(false_positive)
    recalls = cumulative_tp / total_gt_count
    precisions = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1e-9)

    return _interpolated_ap(recalls, precisions)


def compute_precision_recall_for_class(
    predictions: list[tuple[str, PredictionBox]],
    ground_truths: list[tuple[str, GroundTruthBox]],
    iou_threshold: float = 0.5,
) -> tuple[float, float] | None:
    """추론 시 이미 적용된 confidence threshold "그 지점"에서의 Precision/Recall.

    predictions에 들어있는 모든 항목(이미 상위 threshold로 걸러진 것)을 그대로 쓰고,
    AP처럼 confidence를 스윕하지 않는다. 그 class에 GT가 전혀 없으면 None.
    """
    if not ground_truths:
        return None

    gt_by_image: dict[str, list[GroundTruthBox]] = {}
    for image_id, gt in ground_truths:
        gt_by_image.setdefault(image_id, []).append(gt)
    total_gt_count = len(ground_truths)

    sorted_predictions = sorted(predictions, key=lambda item: item[1].confidence, reverse=True)
    if not sorted_predictions:
        return 0.0, 0.0

    true_positive, false_positive = _match_predictions_to_ground_truths(
        sorted_predictions=sorted_predictions,
        gt_by_image=gt_by_image,
        iou_threshold=iou_threshold,
    )
    tp_total = float(np.sum(true_positive))
    fp_total = float(np.sum(false_positive))
    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    recall = tp_total / total_gt_count
    return precision, recall


def _mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def evaluate_dataset(
    predictions_by_class: dict[int, list[tuple[str, PredictionBox]]],
    ground_truths_by_class: dict[int, list[tuple[str, GroundTruthBox]]],
    class_names: dict[int, str],
) -> dict[str, object]:
    """전체 class에 대해 AP50/AP50-95/AP_small/medium/large/Precision/Recall을 계산한다.

    predictions_by_class/ground_truths_by_class는 class_id -> [(image_id, box), ...]
    형태로 이미 분류돼 있다고 가정한다(evaluate.py가 predictions.json/YOLO label을
    읽어서 이 형태로 만들어 넘긴다).
    """
    per_class: dict[str, dict[str, object]] = {}

    for class_id, class_name in sorted(class_names.items()):
        preds = predictions_by_class.get(class_id, [])
        gts = ground_truths_by_class.get(class_id, [])

        ap50 = compute_ap_for_class(preds, gts, iou_threshold=0.5)
        ap_per_threshold = [
            compute_ap_for_class(preds, gts, iou_threshold=threshold)
            for threshold in IOU_THRESHOLDS_50_95
        ]
        ap50_95 = _mean_or_none([value for value in ap_per_threshold if value is not None])

        ap_small = compute_ap_for_class(preds, gts, iou_threshold=0.5, size_bucket="small")
        ap_medium = compute_ap_for_class(preds, gts, iou_threshold=0.5, size_bucket="medium")
        ap_large = compute_ap_for_class(preds, gts, iou_threshold=0.5, size_bucket="large")

        precision_recall = compute_precision_recall_for_class(preds, gts, iou_threshold=0.5)
        precision, recall = precision_recall if precision_recall is not None else (None, None)

        per_class[class_name] = {
            "class_id": class_id,
            "gt_count": len(gts),
            "ap50": ap50,
            "ap50_95": ap50_95,
            "ap_small": ap_small,
            "ap_medium": ap_medium,
            "ap_large": ap_large,
            "precision": precision,
            "recall": recall,
        }

    overall = {
        "map50": _mean_or_none(
            [metrics["ap50"] for metrics in per_class.values() if metrics["ap50"] is not None]
        ),
        "map50_95": _mean_or_none(
            [metrics["ap50_95"] for metrics in per_class.values() if metrics["ap50_95"] is not None]
        ),
        "ap_small": _mean_or_none(
            [metrics["ap_small"] for metrics in per_class.values() if metrics["ap_small"] is not None]
        ),
        "ap_medium": _mean_or_none(
            [metrics["ap_medium"] for metrics in per_class.values() if metrics["ap_medium"] is not None]
        ),
        "ap_large": _mean_or_none(
            [metrics["ap_large"] for metrics in per_class.values() if metrics["ap_large"] is not None]
        ),
    }

    return {"per_class": per_class, "overall": overall}
