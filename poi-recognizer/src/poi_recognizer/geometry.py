"""Original-image geometry, fixed ID intervals and perspective crops."""

import math
from collections.abc import Mapping
from numbers import Integral
from pathlib import Path
from typing import Any

from .errors import POIError

Polygon = tuple[tuple[float, float], ...]


def validate_polygon(
    points: Any,
    width: int | None = None,
    height: int | None = None,
    *,
    quadrilateral: bool = False,
) -> Polygon:
    """Validate simple native 4/8/12-point labels or convex detector quads."""
    from shapely.geometry import Polygon as Shape

    try:
        for dimension in (width, height):
            if dimension is not None and (
                isinstance(dimension, bool) or not isinstance(dimension, Integral) or dimension <= 0
            ):
                raise ValueError("image dimension")
        if isinstance(points, Mapping) or len(points) not in (
            (4,) if quadrilateral else (4, 8, 12)
        ):
            raise ValueError("vertex count")
        polygon_list: list[tuple[float, float]] = []
        for point in points:
            if (
                isinstance(point, Mapping)
                or len(point) != 2
                or any(isinstance(v, (str, bytes, bool)) for v in point)
            ):
                raise ValueError("vertex shape")
            x, y = float(point[0]), float(point[1])
            if not math.isfinite(x) or not math.isfinite(y) or x < 0 or y < 0:
                raise ValueError("coordinate")
            if (width is not None and x > width) or (height is not None and y > height):
                raise ValueError("bounds")
            polygon_list.append((x, y))
        shape = Shape(polygon_list)
        if (
            len(set(polygon_list)) != len(polygon_list)
            or not shape.is_valid
            or not math.isfinite(shape.area)
            or shape.area <= 0
        ):
            raise ValueError("degenerate or self-intersecting polygon")
        if quadrilateral:
            crosses = []
            for index in range(4):
                a, b, c = (polygon_list[(index + offset) % 4] for offset in range(3))
                crosses.append((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
            if not (all(c > 0 for c in crosses) or all(c < 0 for c in crosses)):
                raise ValueError("nonconvex detector quadrilateral")
        return tuple(polygon_list)
    except (TypeError, ValueError, OverflowError, IndexError, KeyError) as exc:
        raise POIError("input_invalid", "geometry", "Invalid original-image polygon") from exc


def read_image(path: Path) -> Any:
    """Decode stored raster pixels into BGR; never apply an EXIF coordinate change."""
    import numpy as np
    from PIL import Image

    try:
        if not path.is_file():
            raise POIError("image_missing", "decode", "Image file is missing")
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"))[:, :, ::-1].copy()
    except (OSError, ValueError) as exc:
        raise POIError("decode_failed", "decode", "Image cannot be decoded") from exc


def crop_text(image: Any, polygon: Polygon) -> Any:
    """Rectify quads; enclose native/concave polygons in a rotated rectangle."""
    import cv2
    import numpy as np

    try:
        image = np.asarray(image)
    except (TypeError, ValueError) as exc:
        raise POIError("input_invalid", "geometry", "Cannot crop an invalid image") from exc
    if (
        image.ndim not in (2, 3)
        or image.shape[0] <= 0
        or image.shape[1] <= 0
        or (image.ndim == 3 and image.shape[2] <= 0)
    ):
        raise POIError("input_invalid", "geometry", "Cannot crop an empty or invalid image")
    height, width = image.shape[:2]
    validated = validate_polygon(polygon, width, height)
    points = np.asarray(validated, dtype=np.float32)
    if len(points) != 4 or not cv2.isContourConvex(points):
        points = np.asarray(cv2.boxPoints(cv2.minAreaRect(points)), dtype=np.float32)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    points = points[np.argsort(angles)]
    points = np.roll(points, -int(np.argmin(points.sum(axis=1))), axis=0)
    crop_width = max(
        2,
        math.ceil(
            max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3]))
        ),
    )
    crop_height = max(
        2,
        math.ceil(
            max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2]))
        ),
    )
    target = np.asarray(
        [
            [0, 0],
            [crop_width - 1, 0],
            [crop_width - 1, crop_height - 1],
            [0, crop_height - 1],
        ],
        dtype=np.float32,
    )
    transform = np.asarray(cv2.getPerspectiveTransform(points, target), dtype=np.float64)
    if not np.isfinite(transform).all() or np.linalg.det(transform) == 0:
        raise POIError("input_invalid", "geometry", "Crop transform is degenerate")
    crop = cv2.warpPerspective(
        image,
        transform,
        (crop_width, crop_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    if crop_height / crop_width >= 1.5:
        crop = np.rot90(crop)
    return np.ascontiguousarray(crop)
