"""좌표, polygon, 거리 계산을 제공하는 공간 분석 모듈."""

from app.inference.detector import BBox, Point


def is_point_in_bbox(point: Point, bbox: BBox) -> bool:
    """
    주어진 점이 Bounding Box 내부 또는 경계에 포함되는지 판정한다.
    """
    x, y = point
    x1, y1, x2, y2 = bbox

    return x1 <= x <= x2 and y1 <= y <= y2


def get_bottom_center(bbox: BBox) -> Point:
    """
    Bounding Box의 하단 중앙 좌표를 계산한다.
    """
    x1, y1, x2, y2 = bbox

    center_x = (x1 + x2) / 2
    bottom_y = y2

    return center_x, bottom_y


def scale_bbox(bbox: BBox, scale: float) -> BBox:
    """
    Bounding Box의 중심을 기준으로 폭과 높이를 scale 배율만큼 확장한다.
    """
    if scale <= 0:
        raise ValueError("scale must be greater than 0")

    x1, y1, x2, y2 = bbox

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    width = x2 - x1
    height = y2 - y1

    scaled_width = width * scale
    scaled_height = height * scale

    new_x1 = center_x - scaled_width / 2
    new_y1 = center_y - scaled_height / 2
    new_x2 = center_x + scaled_width / 2
    new_y2 = center_y + scaled_height / 2

    return new_x1, new_y1, new_x2, new_y2