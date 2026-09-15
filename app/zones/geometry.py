"""좌표, polygon, 거리 계산을 제공하는 공간 분석 모듈."""

from app.inference.detector import BBox, Point


def is_point_in_bbox(point: Point, bbox: BBox) -> bool:
    """
    주어진 점이 Bounding Box 내부 또는 경계에 포함되는지 판정한다.
    """

    # 점의 x, y 좌표를 분리한다.
    x, y = point

    # Bounding Box의 좌상단과 우하단 좌표를 분리한다.
    x1, y1, x2, y2 = bbox

    # x와 y가 각각 Bounding Box 범위 안에 있는지 확인한다.
    return x1 <= x <= x2 and y1 <= y <= y2