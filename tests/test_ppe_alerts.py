"""PPE 누락 누적부터 화면·경고음 전달까지 모델 없이 검증한다."""

from dataclasses import replace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from app import main
from app.alerts.alert_manager import AlertManager
from app.alerts.event_manager import EventManager
from app.alerts.ppe_renderer import draw_ppe_overlay
from app.inference.detector import Detection, Detector
from app.ppe.matcher import PPEMatcher
from app.risk.models import RiskAssessment, RiskLevel
from app.risk.ppe_risk import PPERiskConfig, PPERiskEvaluator, load_ppe_risk_config
from app.risk.risk_engine import RiskEngine
from app.tracking.tracker import SimpleTracker, TrackedObject
from app.video.frame_processor import FrameProcessor
from app.video.video_source import FramePacket
from app.zones.models import EquipmentType
from app.zones.zone_manager import ZoneManager


def packet(index=0, camera="cam_01"):
    return FramePacket(
        camera,
        np.zeros((360, 640, 3), dtype=np.uint8),
        index,
        float(index),
        30,
        640,
        360,
    )


def statuses(worn=()):
    person = Detection(0, "person", 0.9, (100, 40, 240, 330))
    worker = TrackedObject(
        7,
        0,
        "person",
        person.bbox,
        person.center,
        person.bottom_center,
        0.9,
        0,
    )
    ppe = [
        Detection(3, name, 0.9, (150, 50, 170, 70))
        for name in worn
    ]
    return PPEMatcher().match(ppe, [worker])


def test_warning_requires_continuous_missing_and_recovers_per_item():
    now = [0.0]
    evaluator = PPERiskEvaluator(clock=lambda: now[0])

    first = evaluator.evaluate(statuses(["helmet"]), packet())[0]
    assert first.pending_classes == ("gloves", "safety_vest")
    assert first.risk_level == RiskLevel.NORMAL

    now[0] = 1
    second = evaluator.evaluate(
        statuses(["helmet", "gloves"]),
        packet(1),
    )[0]

    assert second.missing_classes == ("safety_vest",)
    assert second.risk_level == RiskLevel.WARNING

    now[0] = 1.1
    recovered = evaluator.evaluate(
        statuses(["helmet", "gloves", "safety_vest"]),
        packet(2),
    )[0]

    assert recovered.missing_classes == recovered.pending_classes == ()

    now[0] = 1.2
    assert not evaluator.evaluate(
        statuses(),
        packet(3),
    )[0].missing_classes


@pytest.mark.parametrize(
    "reset",
    ["absent", "restart", "gap", "camera"],
)
def test_discontinuity_does_not_reuse_missing_duration(reset):
    now = [0.0]
    evaluator = PPERiskEvaluator(clock=lambda: now[0])

    evaluator.evaluate(statuses(), packet())
    now[0] = 1.5

    if reset == "absent":
        evaluator.evaluate({}, packet(1))

    if reset == "gap":
        now[0] = 3

    result = evaluator.evaluate(
        statuses(),
        packet(
            0 if reset == "restart" else 2,
            "cam_02" if reset == "camera" else "cam_01",
        ),
    )[0]

    assert result.missing_classes == ()


def test_config_load_and_validation(tmp_path):
    config_path = tmp_path / "system.yaml"
    config_path.write_text(
        "ppe_risk:\n"
        "  required_classes: [helmet, harness_body]\n"
        "  missing_seconds: 0\n"
    )

    config = load_ppe_risk_config(config_path)

    result = PPERiskEvaluator(config).evaluate(
        statuses(["helmet"]),
        packet(),
    )[0]

    assert result.missing_classes == ("harness_body",)

    for values in (
        {"required_classes": ("head",)},
        {"required_classes": ()},
        {"missing_seconds": -1},
        {"missing_seconds": float("nan")},
        {"max_observation_gap_seconds": 0},
    ):
        with pytest.raises(ValueError):
            PPERiskConfig(**values)


def test_ppe_warning_coexists_with_equipment_critical_and_hold():
    now = [0.0]
    manager = EventManager(clock=lambda: now[0])

    ppe = PPERiskEvaluator(
        PPERiskConfig(missing_seconds=0)
    ).evaluate(
        statuses(),
        packet(),
    )[0]

    assert manager.update(
        "cam_01",
        [ppe],
    ).current_level == RiskLevel.WARNING

    critical = RiskAssessment(
        0,
        0,
        "cam_01",
        7,
        8,
        EquipmentType.FORKLIFT,
        RiskLevel.CRITICAL,
    )

    assert manager.update(
        "cam_01",
        [ppe, critical],
    ).current_level == RiskLevel.CRITICAL

    assert manager.update(
        "cam_02",
        [],
    ).current_level == RiskLevel.NORMAL

    now[0] = 2

    assert manager.update(
        "cam_01",
        [],
    ).current_level == RiskLevel.CRITICAL

    now[0] = 3

    assert manager.update(
        "cam_01",
        [],
    ).current_level == RiskLevel.NORMAL


class _Detector(Detector):
    def detect(self, frame):
        return [
            Detection(
                0,
                "person",
                0.9,
                (100, 40, 240, 330),
            ),
            Detection(
                3,
                "helmet",
                0.9,
                (150, 40, 190, 65),
            ),
        ]


class _Source:
    camera_id = "cam_01"

    def __init__(self):
        self.index = 0

    def read(self):
        result = packet(self.index)
        self.index += 1
        return result


def test_end_to_end_matching_missing_overlay_badge_and_sound(monkeypatch):
    source = _Source()
    now = [0.0]

    processor = FrameProcessor(
        source,
        _Detector(),
        ZoneManager(),
        RiskEngine(),
        tracker=SimpleTracker(),
        ppe_risk_evaluator=PPERiskEvaluator(
            clock=lambda: now[0]
        ),
        draw_bbox=False,
        draw_metrics=False,
    )

    sound = Mock()

    manager = AlertManager(
        EventManager(clock=lambda: now[0]),
        sound,
    )

    view = main.CameraViewState(
        source,
        processor,
        True,
    )

    logger = Mock()

    first = main._read_display_frame(
        view,
        logger,
        manager,
        now=now[0],
    )

    assert first.alert_update.current_level == RiskLevel.NORMAL

    now[0] = 1.0
    texts = []
    original = cv2.putText

    def record(frame, text, *args, **kwargs):
        texts.append(text)
        return original(frame, text, *args, **kwargs)

    monkeypatch.setattr(
        cv2,
        "putText",
        record,
    )

    result = main._read_display_frame(
        view,
        logger,
        manager,
        now=now[0],
    )

    assert result.alert_update.current_level == RiskLevel.WARNING
    assert "helmet: OK" in texts
    assert "gloves: MISSING" in texts
    assert "safety_vest: MISSING" in texts
    assert "Person #1 PPE WARNING" in texts
    assert "WARNING" in texts
    assert np.any(result.frame)

    manager.update_sound(
        [result.alert_update],
        now[0],
    )

    sound.update.assert_called_once_with(
        [result.alert_update],
        now[0],
    )


def test_renderer_scales_bbox_and_does_not_change_source():
    source = packet().frame
    frame = cv2.resize(
        source,
        (320, 180),
    )

    worker = SimpleTracker().update(
        _Detector().detect(source),
        packet(),
    )[0]

    assessments = PPERiskEvaluator(
        PPERiskConfig(missing_seconds=0)
    ).evaluate(
        {
            1: replace(
                statuses()[7],
                track_id=1,
            )
        },
        packet(),
    )

    draw_ppe_overlay(
        frame,
        [worker],
        assessments,
        (640, 360),
    )

    assert np.any(frame[165, 120])
    assert not np.any(source)