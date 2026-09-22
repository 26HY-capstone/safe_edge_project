"""샘플 metadata JSON에서 ceiling/eye 영상 선택을 검증하는 테스트."""

import json
from pathlib import Path

import pytest

from app.video.sample_dataset import (
    CeilingEyeVideoPair,
    create_random_ceiling_eye_video_sources,
    select_random_ceiling_eye_video_pair,
)


def _write_sample_run(sample_dir: Path, run_id: str) -> None:
    ceiling_name = f"{run_id}.ceiling_00.rgb.mp4"
    eye_name = f"{run_id}.eye_00.rgb.mp4"

    (sample_dir / ceiling_name).touch()
    (sample_dir / eye_name).touch()
    (sample_dir / f"{run_id}.meta.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "cameras": [
                    {
                        "camera_alias": "ceiling_00",
                        "rgb_member": ceiling_name,
                    },
                    {
                        "camera_alias": "eye_00",
                        "rgb_member": eye_name,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def test_select_random_ceiling_eye_video_pair_uses_same_metadata(
    tmp_path,
) -> None:
    _write_sample_run(tmp_path, "run-a")

    pair = select_random_ceiling_eye_video_pair(tmp_path, seed=1)

    assert pair == CeilingEyeVideoPair(
        meta_path=tmp_path / "run-a.meta.json",
        run_id="run-a",
        ceiling_path=tmp_path / "run-a.ceiling_00.rgb.mp4",
        eye_path=tmp_path / "run-a.eye_00.rgb.mp4",
    )


def test_create_random_ceiling_eye_video_sources(tmp_path) -> None:
    _write_sample_run(tmp_path, "run-a")

    video_sources = create_random_ceiling_eye_video_sources(
        sample_dir=tmp_path,
        seed=1,
        loop=False,
    )

    assert len(video_sources) == 2
    assert video_sources[0].camera_id == "run-a-ceiling"
    assert video_sources[0].source == tmp_path / "run-a.ceiling_00.rgb.mp4"
    assert video_sources[0].loop is False
    assert video_sources[1].camera_id == "run-a-eye"
    assert video_sources[1].source == tmp_path / "run-a.eye_00.rgb.mp4"


def test_select_random_ceiling_eye_video_pair_rejects_missing_eye(
    tmp_path,
) -> None:
    run_id = "run-a"
    ceiling_name = f"{run_id}.ceiling_00.rgb.mp4"
    (tmp_path / ceiling_name).touch()
    (tmp_path / f"{run_id}.meta.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "cameras": [
                    {
                        "camera_alias": "ceiling_00",
                        "rgb_member": ceiling_name,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="No eye video member"):
        select_random_ceiling_eye_video_pair(tmp_path, seed=1)
