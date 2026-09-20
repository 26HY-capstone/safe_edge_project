"""네 화면 데모의 프레임 배치를 검증한다."""

import numpy as np

from app.demo.multi_camera_viewer import (
    TILE_HEIGHT,
    TILE_WIDTH,
    compose_2x2_grid,
)


def test_compose_2x2_grid_resizes_and_fills_empty_tiles() -> None:
    frame = np.full((10, 20, 3), 255, dtype=np.uint8)

    grid = compose_2x2_grid([frame])

    assert grid.shape == (TILE_HEIGHT * 2, TILE_WIDTH * 2, 3)
    assert grid.dtype == np.uint8
