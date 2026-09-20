"""좌표, bbox, polygon 계산을 제공하는 공간 분석 모듈."""

from collections.abc import Sequence

from app.inference.detector import BBox, Point
from app.zones.models import Polygon


def is_point_in_bbox(point: Point, bbox: BBox) -> bool:
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def is_point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    """경계를 포함해 점이 polygon 안에 있는지 ray casting으로 판정한다."""
    if len(polygon) < 3:
        return False

    x, y = point
    inside = False
    previous = polygon[-1]

    for current in polygon:
        if _is_point_on_segment(point, previous, current):
            return True

        x1, y1 = previous
        x2, y2 = current
        crosses = (y1 > y) != (y2 > y)
        if crosses:
            intersection_x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < intersection_x:
                inside = not inside
        previous = current

    return inside


def get_bottom_center(bbox: BBox) -> Point:
    x1, _, x2, y2 = bbox
    return ((x1 + x2) / 2.0, y2)


def bbox_to_polygon(bbox: BBox) -> Polygon:
    x1, y1, x2, y2 = bbox
    return ((x1, y1), (x2, y1), (x2, y2), (x1, y2))


def expand_bbox(
    bbox: BBox,
    margin_px: float,
    frame_width: int | None = None,
    frame_height: int | None = None,
) -> BBox:
    if margin_px < 0.0:
        raise ValueError("margin_px must be non-negative")

    x1, y1, x2, y2 = bbox
    expanded = (
        x1 - margin_px,
        y1 - margin_px,
        x2 + margin_px,
        y2 + margin_px,
    )
    return clip_bbox(expanded, frame_width=frame_width, frame_height=frame_height)


def scale_bbox(bbox: BBox, scale: float) -> BBox:
    """기존 호출부 호환을 위해 bbox를 중심 기준 배율로 확장한다."""
    if scale <= 0.0:
        raise ValueError("scale must be greater than 0")

    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    half_width = (x2 - x1) * scale / 2.0
    half_height = (y2 - y1) * scale / 2.0
    return (
        center_x - half_width,
        center_y - half_height,
        center_x + half_width,
        center_y + half_height,
    )


def clip_bbox(
    bbox: BBox,
    frame_width: int | None,
    frame_height: int | None,
) -> BBox:
    x1, y1, x2, y2 = bbox
    if frame_width is not None:
        maximum_x = max(0.0, float(frame_width - 1))
        x1 = min(max(0.0, x1), maximum_x)
        x2 = min(max(0.0, x2), maximum_x)
    if frame_height is not None:
        maximum_y = max(0.0, float(frame_height - 1))
        y1 = min(max(0.0, y1), maximum_y)
        y2 = min(max(0.0, y2), maximum_y)
    return (x1, y1, x2, y2)


def _is_point_on_segment(point: Point, start: Point, end: Point) -> bool:
    x, y = point
    x1, y1 = start
    x2, y2 = end
    cross_product = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    if abs(cross_product) > 1e-9:
        return False
    return (
        min(x1, x2) - 1e-9 <= x <= max(x1, x2) + 1e-9
        and min(y1, y2) - 1e-9 <= y <= max(y1, y2) + 1e-9
    )
