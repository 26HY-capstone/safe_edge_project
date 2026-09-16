"""PyTorch YOLO backend의 Detection 변환을 검증하는 테스트."""

from pathlib import Path

import numpy as np

from app.inference.detector import Detection
from app.inference.pytorch_backend import (
    PyTorchDetectorConfig,
    PyTorchYOLODetector,
    load_pytorch_detector_config,
)


class _FakeBoxes:
    def __init__(self) -> None:
        self.xyxy = np.asarray(
            [
                [10.0, 20.0, 50.0, 80.0],
                [5.0, 5.0, 10.0, 10.0],
            ],
            dtype=np.float32,
        )
        self.conf = np.asarray([0.9, 0.1], dtype=np.float32)
        self.cls = np.asarray([0, 6], dtype=np.float32)


class _FakeResult:
    boxes = _FakeBoxes()
    names = {0: "person", 6: "forklift"}


class _FakeYOLOModel:
    names = {0: "person", 6: "forklift"}

    def __init__(self) -> None:
        self.predict_kwargs = None

    def predict(self, **kwargs):
        self.predict_kwargs = kwargs
        return [_FakeResult()]


def test_pytorch_yolo_detector_converts_results_to_detections() -> None:
    model = _FakeYOLOModel()
    detector = PyTorchYOLODetector(
        config=PyTorchDetectorConfig(
            model_path=Path("models/best.pt"),
            device="cpu",
            input_size=320,
            confidence_threshold=0.35,
            iou_threshold=0.45,
            classes={0: "worker", 6: "forklift"},
        ),
        model=model,
    )

    detections = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))

    assert detections == [
        Detection(
            class_id=0,
            class_name="worker",
            confidence=float(np.float32(0.9)),
            bbox=(10.0, 20.0, 50.0, 80.0),
        )
    ]
    assert model.predict_kwargs["imgsz"] == 320
    assert model.predict_kwargs["conf"] == 0.35
    assert model.predict_kwargs["iou"] == 0.45
    assert model.predict_kwargs["device"] == "cpu"
    assert model.predict_kwargs["verbose"] is False


def test_load_pytorch_detector_config_reads_model_yaml(tmp_path) -> None:
    config_path = tmp_path / "model.yaml"
    config_path.write_text(
        """
detector:
  backend: pytorch
  model_path: models/custom.pt
  device: cpu
  input_size: 640
  confidence_threshold: 0.4
  iou_threshold: 0.5
  classes:
    0: person
    6: forklift
""",
        encoding="utf-8",
    )

    config = load_pytorch_detector_config(config_path)

    assert config == PyTorchDetectorConfig(
        model_path=Path("models/custom.pt"),
        device="cpu",
        input_size=640,
        confidence_threshold=0.4,
        iou_threshold=0.5,
        classes={0: "person", 6: "forklift"},
    )
