"""experiments/sahi/metrics.py의 AP/Precision/Recall 계산을 synthetic GT/prediction으로
검증한다. 실제 모델이나 데이터셋이 전혀 필요 없다."""

from __future__ import annotations

from experiments.sahi.metrics import (
    GroundTruthBox,
    PredictionBox,
    box_iou,
    compute_ap_for_class,
    compute_precision_recall_for_class,
    evaluate_dataset,
)


def test_box_iou_identical_boxes_is_one() -> None:
    box = (0.0, 0.0, 10.0, 10.0)
    assert box_iou(box, box) == 1.0


def test_box_iou_non_overlapping_boxes_is_zero() -> None:
    assert box_iou((0.0, 0.0, 10.0, 10.0), (20.0, 20.0, 30.0, 30.0)) == 0.0


def test_box_iou_known_partial_overlap() -> None:
    # (0,0,10,10)와 (5,5,15,15): 교집합 (5,5,10,10)=25, 합집합 100+100-25=175
    iou = box_iou((0.0, 0.0, 10.0, 10.0), (5.0, 5.0, 15.0, 15.0))
    assert abs(iou - 25.0 / 175.0) < 1e-9


def test_perfect_match_gives_ap_one() -> None:
    gt = [("img1", GroundTruthBox(class_id=0, bbox_xyxy=(0.0, 0.0, 100.0, 100.0)))]
    pred = [("img1", PredictionBox(class_id=0, confidence=0.9, bbox_xyxy=(0.0, 0.0, 100.0, 100.0)))]

    ap = compute_ap_for_class(pred, gt, iou_threshold=0.5)

    assert ap == 1.0


def test_higher_confidence_false_positive_reduces_ap() -> None:
    """confidence가 더 높은 FP가 TP보다 먼저 평가되면 AP가 1.0보다 낮아져야 한다.

    반대로 FP의 confidence가 TP보다 낮으면(= recall 1.0 달성 후의 FP) 101-point
    interpolation 특성상 AP가 줄지 않는 것이 COCO/VOC의 올바른 동작이다 — 이미
    도달한 recall 지점의 "역대 최고 precision"을 그대로 쓰기 때문이다.
    """
    gt = [("img1", GroundTruthBox(class_id=0, bbox_xyxy=(0.0, 0.0, 100.0, 100.0)))]
    pred = [
        ("img1", PredictionBox(class_id=0, confidence=0.95, bbox_xyxy=(500.0, 500.0, 600.0, 600.0))),
        ("img1", PredictionBox(class_id=0, confidence=0.8, bbox_xyxy=(0.0, 0.0, 100.0, 100.0))),
    ]

    ap = compute_ap_for_class(pred, gt, iou_threshold=0.5)

    assert ap is not None and ap < 1.0


def test_missed_ground_truth_reduces_ap_below_one() -> None:
    gt = [
        ("img1", GroundTruthBox(class_id=0, bbox_xyxy=(0.0, 0.0, 100.0, 100.0))),
        ("img1", GroundTruthBox(class_id=0, bbox_xyxy=(200.0, 200.0, 300.0, 300.0))),
    ]
    pred = [("img1", PredictionBox(class_id=0, confidence=0.9, bbox_xyxy=(0.0, 0.0, 100.0, 100.0)))]

    ap = compute_ap_for_class(pred, gt, iou_threshold=0.5)

    assert ap is not None and ap < 1.0


def test_ap_is_none_when_class_has_no_ground_truth() -> None:
    ap = compute_ap_for_class(predictions=[], ground_truths=[], iou_threshold=0.5)
    assert ap is None


def test_ap_size_bucket_filters_ground_truth_by_area() -> None:
    small_gt = GroundTruthBox(class_id=0, bbox_xyxy=(0.0, 0.0, 10.0, 10.0))  # area=100 < 1024 -> small
    large_gt = GroundTruthBox(class_id=0, bbox_xyxy=(0.0, 0.0, 200.0, 200.0))  # area=40000 -> large
    gt = [("img1", small_gt), ("img2", large_gt)]
    pred = [
        ("img1", PredictionBox(class_id=0, confidence=0.9, bbox_xyxy=(0.0, 0.0, 10.0, 10.0))),
    ]

    ap_small = compute_ap_for_class(pred, gt, iou_threshold=0.5, size_bucket="small")
    ap_large = compute_ap_for_class(pred, gt, iou_threshold=0.5, size_bucket="large")

    assert ap_small == 1.0  # small GT와 정확히 일치하는 prediction이 있음
    assert ap_large == 0.0  # large GT를 맞춘 prediction이 없음


def test_size_bucket_boundaries() -> None:
    assert GroundTruthBox(class_id=0, bbox_xyxy=(0.0, 0.0, 10.0, 10.0)).size_bucket == "small"  # 100
    assert GroundTruthBox(class_id=0, bbox_xyxy=(0.0, 0.0, 50.0, 50.0)).size_bucket == "medium"  # 2500
    assert GroundTruthBox(class_id=0, bbox_xyxy=(0.0, 0.0, 200.0, 200.0)).size_bucket == "large"  # 40000


def test_precision_recall_perfect_match() -> None:
    gt = [("img1", GroundTruthBox(class_id=0, bbox_xyxy=(0.0, 0.0, 100.0, 100.0)))]
    pred = [("img1", PredictionBox(class_id=0, confidence=0.9, bbox_xyxy=(0.0, 0.0, 100.0, 100.0)))]

    result = compute_precision_recall_for_class(pred, gt, iou_threshold=0.5)

    assert result == (1.0, 1.0)


def test_precision_recall_with_one_false_positive_and_one_miss() -> None:
    gt = [
        ("img1", GroundTruthBox(class_id=0, bbox_xyxy=(0.0, 0.0, 100.0, 100.0))),
        ("img1", GroundTruthBox(class_id=0, bbox_xyxy=(200.0, 200.0, 300.0, 300.0))),
    ]
    pred = [
        ("img1", PredictionBox(class_id=0, confidence=0.9, bbox_xyxy=(0.0, 0.0, 100.0, 100.0))),  # TP
        ("img1", PredictionBox(class_id=0, confidence=0.8, bbox_xyxy=(500.0, 500.0, 600.0, 600.0))),  # FP
    ]

    precision, recall = compute_precision_recall_for_class(pred, gt, iou_threshold=0.5)

    assert precision == 0.5  # TP=1, FP=1
    assert recall == 0.5  # TP=1, total_gt=2


def test_precision_recall_is_none_without_ground_truth() -> None:
    assert compute_precision_recall_for_class([], [], iou_threshold=0.5) is None


def test_evaluate_dataset_builds_per_class_and_overall_metrics() -> None:
    class_names = {0: "person", 1: "helmet"}
    predictions_by_class = {
        0: [("img1", PredictionBox(class_id=0, confidence=0.9, bbox_xyxy=(0.0, 0.0, 100.0, 100.0)))],
    }
    ground_truths_by_class = {
        0: [("img1", GroundTruthBox(class_id=0, bbox_xyxy=(0.0, 0.0, 100.0, 100.0)))],
        1: [("img1", GroundTruthBox(class_id=1, bbox_xyxy=(0.0, 0.0, 20.0, 20.0)))],
    }

    result = evaluate_dataset(predictions_by_class, ground_truths_by_class, class_names)

    assert result["per_class"]["person"]["ap50"] == 1.0
    assert result["per_class"]["person"]["gt_count"] == 1
    # helmet은 GT는 있지만 prediction이 전혀 없으므로 AP=0.0 (None이 아니라 0점)
    assert result["per_class"]["helmet"]["ap50"] == 0.0
    assert result["per_class"]["helmet"]["gt_count"] == 1
    # 두 class 모두 GT가 있으므로 overall map50은 두 AP의 평균
    assert result["overall"]["map50"] == 0.5


def test_evaluate_dataset_class_without_any_ground_truth_excluded_from_overall() -> None:
    class_names = {0: "person", 2: "forklift"}
    predictions_by_class: dict = {}
    ground_truths_by_class = {
        0: [("img1", GroundTruthBox(class_id=0, bbox_xyxy=(0.0, 0.0, 100.0, 100.0)))],
        # forklift(class_id=2)는 이 test set에 GT가 전혀 없음
    }

    result = evaluate_dataset(predictions_by_class, ground_truths_by_class, class_names)

    assert result["per_class"]["forklift"]["ap50"] is None
    assert result["per_class"]["forklift"]["gt_count"] == 0
    # forklift는 GT가 없어 overall 평균에서 제외되고, person만의 AP(0.0)가 그대로 overall이 된다
    assert result["overall"]["map50"] == 0.0
