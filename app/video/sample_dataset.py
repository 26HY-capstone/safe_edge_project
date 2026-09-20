"""샘플 metadata JSON에서 ceiling/eye 영상 쌍을 선택하는 모듈."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any

from app.video.video_source import PROJECT_ROOT, VideoSource


@dataclass(frozen=True, slots=True)
class CeilingEyeVideoPair:
    meta_path: Path
    run_id: str
    ceiling_path: Path
    eye_path: Path


def select_random_ceiling_eye_video_pair(
    sample_dir: str | Path,
    seed: int | None = None,
) -> CeilingEyeVideoPair:
    resolved_sample_dir = _resolve_project_path(sample_dir)
    rng = random.Random(seed)
    meta_paths = sorted(resolved_sample_dir.glob("*.meta.json"))
    if not meta_paths:
        raise FileNotFoundError(f"No metadata JSON found in {resolved_sample_dir}")

    meta_path = rng.choice(meta_paths)
    metadata = _load_metadata(meta_path)
    run_id = _required_str(metadata, "run_id")
    cameras = metadata.get("cameras")
    if not isinstance(cameras, list):
        raise ValueError(f"cameras must be a list: {meta_path}")

    ceiling_members = _rgb_members_by_alias_prefix(cameras, "ceiling")
    eye_members = _rgb_members_by_alias_prefix(cameras, "eye")
    if not ceiling_members:
        raise ValueError(f"No ceiling video member found in {meta_path}")
    if not eye_members:
        raise ValueError(f"No eye video member found in {meta_path}")

    ceiling_path = resolved_sample_dir / rng.choice(ceiling_members)
    eye_path = resolved_sample_dir / rng.choice(eye_members)
    _validate_video_file(ceiling_path)
    _validate_video_file(eye_path)
    return CeilingEyeVideoPair(
        meta_path=meta_path,
        run_id=run_id,
        ceiling_path=ceiling_path,
        eye_path=eye_path,
    )


def create_random_ceiling_eye_video_sources(
    sample_dir: str | Path,
    seed: int | None = None,
    loop: bool = True,
) -> list[VideoSource]:
    pair = select_random_ceiling_eye_video_pair(sample_dir=sample_dir, seed=seed)
    return [
        VideoSource(
            source=pair.ceiling_path,
            camera_id=f"{pair.run_id}-ceiling",
            loop=loop,
        ),
        VideoSource(
            source=pair.eye_path,
            camera_id=f"{pair.run_id}-eye",
            loop=loop,
        ),
    ]


def _load_metadata(meta_path: Path) -> Mapping[str, Any]:
    with meta_path.open("r", encoding="utf-8") as meta_file:
        metadata = json.load(meta_file)
    if not isinstance(metadata, Mapping):
        raise ValueError(f"metadata must be a JSON object: {meta_path}")
    return metadata


def _rgb_members_by_alias_prefix(
    cameras: list[object],
    alias_prefix: str,
) -> list[str]:
    members: list[str] = []
    for camera in cameras:
        if not isinstance(camera, Mapping):
            continue
        alias = camera.get("camera_alias")
        rgb_member = camera.get("rgb_member")
        if (
            isinstance(alias, str)
            and isinstance(rgb_member, str)
            and alias.startswith(alias_prefix)
        ):
            members.append(rgb_member)
    return sorted(members)


def _required_str(values: Mapping[str, Any], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _validate_video_file(video_path: Path) -> None:
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")


def _resolve_project_path(path: str | Path) -> Path:
    resolved_path = Path(path).expanduser()
    if resolved_path.is_absolute():
        return resolved_path
    return PROJECT_ROOT / resolved_path
