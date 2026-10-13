"""작업자별 PPE 누락을 시간 누적하여 WARNING으로 판정한다."""

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
import time

import yaml

from app.ppe.matcher import PPE_CLASSES, WorkerPPEStatus
from app.risk.models import RiskLevel
from app.video.video_source import FramePacket


@dataclass(frozen=True, slots=True)
class PPERiskConfig:
    required_classes: tuple[str, ...] = ("helmet", "gloves", "safety_vest")
    missing_seconds: float = 1.0
    max_observation_gap_seconds: float = 2.0

    def __post_init__(self) -> None:
        """필수 장비와 누적 시간 설정을 검증한다."""
        if not self.required_classes or len(set(self.required_classes)) != len(self.required_classes):
            raise ValueError("required_classes must be non-empty and unique")
        if any(name not in PPE_CLASSES for name in self.required_classes):
            raise ValueError("unsupported required PPE class")
        if not isfinite(self.missing_seconds) or self.missing_seconds < 0:
            raise ValueError("missing_seconds must be finite and non-negative")
        if not isfinite(self.max_observation_gap_seconds) or self.max_observation_gap_seconds <= 0:
            raise ValueError("max_observation_gap_seconds must be finite and positive")


def load_ppe_risk_config(path: Path) -> PPERiskConfig:
    """system.yaml에서 PPE 경고 설정을 읽는다."""
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    values = dict(data.get("ppe_risk", {}))
    if "required_classes" in values:
        values["required_classes"] = tuple(values["required_classes"])
    return PPERiskConfig(**values)


@dataclass(frozen=True, slots=True)
class PPERiskAssessment:
    camera_id: str
    person_id: int
    timestamp: float
    frame_index: int
    required_classes: tuple[str, ...]
    missing_classes: tuple[str, ...]
    pending_classes: tuple[str, ...]
    risk_level: RiskLevel


class PPERiskEvaluator:
    def __init__(
        self, config: PPERiskConfig | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """설비 위험과 분리된 PPE 누적 상태와 주입 가능한 시계를 준비한다."""
        self.config = config or PPERiskConfig()
        self._clock = clock
        self._since: dict[tuple[str, int, str], float] = {}
        self._last: dict[str, tuple[int, float]] = {}

    def evaluate(
        self, statuses: dict[int, WorkerPPEStatus], packet: FramePacket,
    ) -> list[PPERiskAssessment]:
        """연속 누락만 누적하며 사라진 작업자와 재시작된 영상 상태는 버린다."""
        now = self._clock()
        camera = packet.camera_id
        previous = self._last.get(camera)
        reset = previous is not None and (
            packet.frame_index <= previous[0] or now < previous[1]
            or now - previous[1] > self.config.max_observation_gap_seconds
        )
        self._since = {
            key: start for key, start in self._since.items()
            if key[0] != camera or (not reset and key[1] in statuses)
        }
        self._last[camera] = (packet.frame_index, now)
        assessments = []
        for track_id, status in statuses.items():
            worn = {item.class_name for item in status.items if item.is_worn}
            missing, pending = [], []
            for name in self.config.required_classes:
                key = (camera, track_id, name)
                if name in worn:
                    self._since.pop(key, None)
                    continue
                since = self._since.setdefault(key, now)
                target = missing if now - since >= self.config.missing_seconds else pending
                target.append(name)
            assessments.append(PPERiskAssessment(
                camera_id=camera, person_id=track_id, timestamp=packet.timestamp,
                frame_index=packet.frame_index, required_classes=self.config.required_classes,
                missing_classes=tuple(missing), pending_classes=tuple(pending),
                risk_level=RiskLevel.WARNING if missing else RiskLevel.NORMAL,
            ))
        return assessments
