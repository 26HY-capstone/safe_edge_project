"""샘플 metadata에서 ceiling/eye 영상 쌍을 찾는 흐름을 검증한다."""

import json

import pytest

from app.video.sample_dataset import select_random_ceiling_eye_video_pair


def test_select_random_ceiling_eye_video_pair(tmp_path) -> None:
    metadata = {
        "run_id": "run-01",
        "cameras": [
            {
                "camera_alias": "ceiling-01",
                "rgb_member": "ceiling.mp4",
            },
            {"camera_alias": "eye-01", "rgb_member": "eye.mp4"},
        ],
    }
    (tmp_path / "sample.meta.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    (tmp_path / "ceiling.mp4").touch()
    (tmp_path / "eye.mp4").touch()

    pair = select_random_ceiling_eye_video_pair(tmp_path, seed=1)

    assert pair.run_id == "run-01"
    assert pair.ceiling_path == tmp_path / "ceiling.mp4"
    assert pair.eye_path == tmp_path / "eye.mp4"


def test_select_random_pair_rejects_missing_eye_video(tmp_path) -> None:
    metadata = {
        "run_id": "run-01",
        "cameras": [
            {
                "camera_alias": "ceiling-01",
                "rgb_member": "ceiling.mp4",
            }
        ],
    }
    (tmp_path / "sample.meta.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="No eye video member"):
        select_random_ceiling_eye_video_pair(tmp_path)
