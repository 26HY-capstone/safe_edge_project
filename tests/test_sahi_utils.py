"""experiments/sahi/utils.py의 공통 유틸리티(이미지 나열, latency 통계, slice 설정
해석, predictions 저장/로드)를 모델/실제 데이터 없이 검증한다."""

from __future__ import annotations

import json

import numpy as np
import pytest

from app.inference.detector import Detection
from experiments.sahi.utils import (
    build_image_record,
    compute_latency_stats,
    list_image_paths,
    load_predictions,
    resolve_slice_settings,
    save_predictions,
    save_summary,
)


def test_list_image_paths_returns_sorted_supported_extensions(tmp_path) -> None:
    (tmp_path / "b.jpg").write_bytes(b"x")
    (tmp_path / "a.png").write_bytes(b"x")
    (tmp_path / "c.txt").write_bytes(b"x")  # 지원하지 않는 확장자

    paths = list_image_paths(tmp_path)

    assert [path.name for path in paths] == ["a.png", "b.jpg"]


def test_list_image_paths_raises_for_missing_directory(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        list_image_paths(tmp_path / "missing")


def test_list_image_paths_raises_for_empty_directory(tmp_path) -> None:
    with pytest.raises(ValueError):
        list_image_paths(tmp_path)


def test_compute_latency_stats_empty_list_returns_zeros() -> None:
    assert compute_latency_stats([]) == {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}


def test_compute_latency_stats_matches_numpy_percentile() -> None:
    latencies = [10.0, 20.0, 30.0, 40.0, 50.0]

    stats = compute_latency_stats(latencies)

    assert stats["mean_ms"] == pytest.approx(30.0)
    assert stats["p50_ms"] == pytest.approx(float(np.percentile(latencies, 50)))
    assert stats["p95_ms"] == pytest.approx(float(np.percentile(latencies, 95)))


def test_build_image_record_serializes_detections() -> None:
    detection = Detection(class_id=3, class_name="helmet", confidence=0.81, bbox=(100.0, 200.0, 150.0, 260.0))

    record = build_image_record(
        image_name="example.jpg",
        width=1920,
        height=1080,
        detections=[detection],
        latency_ms=25.3,
    )

    assert record == {
        "image": "example.jpg",
        "width": 1920,
        "height": 1080,
        "detections": [
            {
                "class_id": 3,
                "class_name": "helmet",
                "confidence": 0.81,
                "bbox_xyxy": [100.0, 200.0, 150.0, 260.0],
            }
        ],
        "latency_ms": 25.3,
    }


def test_build_image_record_with_no_detections() -> None:
    record = build_image_record(
        image_name="empty.jpg", width=100, height=100, detections=[], latency_ms=5.0
    )

    assert record["detections"] == []


def test_save_and_load_predictions_round_trip(tmp_path) -> None:
    records = [
        build_image_record(image_name="a.jpg", width=10, height=10, detections=[], latency_ms=1.0)
    ]

    predictions_path = save_predictions(records, tmp_path)
    loaded = load_predictions(predictions_path)

    assert loaded == records


def test_load_predictions_raises_for_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_predictions(tmp_path / "missing.json")


def test_save_summary_includes_image_count_and_latency(tmp_path) -> None:
    summary_path = save_summary(
        tmp_path,
        image_count=5,
        latency_stats={"mean_ms": 10.0, "p50_ms": 9.0, "p95_ms": 15.0},
        extra={"experiment": "standard"},
    )

    assert summary_path.is_file()
    saved = json.loads(summary_path.read_text(encoding="utf-8"))
    assert saved["image_count"] == 5
    assert saved["latency"]["mean_ms"] == 10.0
    assert saved["experiment"] == "standard"


def test_resolve_slice_settings_uses_hardcoded_defaults_when_nothing_given() -> None:
    settings = resolve_slice_settings(
        slice_size=None,
        slice_width=None,
        slice_height=None,
        overlap=None,
        overlap_width=None,
        overlap_height=None,
        config_defaults={},
    )

    assert settings.slice_width == 640
    assert settings.slice_height == 640
    assert settings.overlap_width_ratio == 0.2
    assert settings.overlap_height_ratio == 0.2


def test_resolve_slice_settings_cli_overrides_config_defaults() -> None:
    settings = resolve_slice_settings(
        slice_size=None,
        slice_width=960,
        slice_height=None,
        overlap=0.3,
        overlap_width=None,
        overlap_height=None,
        config_defaults={"width": 640, "height": 640, "overlap_width_ratio": 0.2, "overlap_height_ratio": 0.2},
    )

    assert settings.slice_width == 960  # CLI 축별 인자가 config보다 우선
    assert settings.slice_height == 640  # 지정 안 된 축은 config 기본값
    assert settings.overlap_width_ratio == 0.3
    assert settings.overlap_height_ratio == 0.3  # 공통 --overlap이 양쪽 축에 적용


def test_resolve_slice_settings_square_slice_size_sets_both_axes() -> None:
    settings = resolve_slice_settings(
        slice_size=960,
        slice_width=None,
        slice_height=None,
        overlap=None,
        overlap_width=None,
        overlap_height=None,
        config_defaults={},
    )

    assert settings.slice_width == 960
    assert settings.slice_height == 960


def test_resolve_slice_settings_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError):
        resolve_slice_settings(
            slice_size=640,
            slice_width=None,
            slice_height=None,
            overlap=1.5,
            overlap_width=None,
            overlap_height=None,
            config_defaults={},
        )
