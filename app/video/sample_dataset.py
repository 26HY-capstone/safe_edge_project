"""샘플 데이터셋의 metadata JSON에서 영상 입력을 선택하는 모듈."""

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
    """metadata JSON 하나를 고르고 같은 run의 ceiling/eye 영상을 하나씩 선택한다."""
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
    """선택된 ceiling/eye 영상 쌍을 VideoSource 목록으로 변환한다."""
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
    """JSON 파일을 읽고 최상위 객체가 mapping인지 검증한다."""
    with meta_path.open("r", encoding="utf-8") as meta_file:
        metadata = json.load(meta_file)

    if not isinstance(metadata, Mapping):
        raise ValueError(f"metadata must be a JSON object: {meta_path}")
    return metadata


def _rgb_members_by_alias_prefix(
    cameras: list[object],
    alias_prefix: str,
) -> list[str]:
    """camera_alias prefix에 해당하는 rgb_member 파일명을 반환한다."""
    rgb_members: list[str] = []

    for camera in cameras:
        if not isinstance(camera, Mapping):
            continue

        alias = camera.get("camera_alias")
        rgb_member = camera.get("rgb_member")
        if not isinstance(alias, str) or not isinstance(rgb_member, str):
            continue
        if alias.startswith(alias_prefix):
            rgb_members.append(rgb_member)

    return sorted(rgb_members)


def _required_str(values: Mapping[str, Any], key: str) -> str:
    """필수 문자열 값을 공백 제거 후 반환한다."""
    value = values.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _validate_video_file(video_path: Path) -> None:
    """metadata가 가리키는 영상 파일이 실제로 존재하는지 확인한다."""
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")


def _resolve_project_path(path: str | Path) -> Path:
    """상대경로를 프로젝트 루트 기준으로 해석한다."""
    resolved_path = Path(path).expanduser()
    if resolved_path.is_absolute():
        return resolved_path
    if resolved_path.exists():
        return resolved_path
    return PROJECT_ROOT / resolved_path
