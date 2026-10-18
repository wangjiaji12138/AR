"""Shared OCR resize rules; dynamic recognition widths never squeeze long crops."""

import math
from typing import Any

import cv2
import numpy as np

DET_SIDE_LEN = 960
DET_LIMIT_TYPE = "max"
REC_HEIGHT = 48
REC_MIN_WIDTH = 320


def recognition_width(ratio: float) -> int:
    """Round up to the mobile backbone's width stride, with a 320px minimum."""
    return max(REC_MIN_WIDTH, math.ceil(REC_HEIGHT * ratio / 8) * 8)


def resize_recognition(image: Any, width: int | None = None) -> tuple[Any, float]:
    """Resize proportionally to height 48, normalize BGR and pad on the right."""
    height, original_width = image.shape[:2]
    resized_width = math.ceil(REC_HEIGHT * original_width / height)
    width = width or recognition_width(original_width / height)
    if width < resized_width:
        raise ValueError("Recognition width would compress the text crop")
    resized = cv2.resize(image, (resized_width, REC_HEIGHT), interpolation=cv2.INTER_LINEAR)
    normalized = (resized.astype(np.float32).transpose(2, 0, 1) / 255 - 0.5) / 0.5
    padded = np.zeros((3, REC_HEIGHT, width), dtype=np.float32)
    padded[:, :, :resized_width] = normalized
    return padded, resized_width / width


class DynamicRecResize:
    """PaddleOCR data operator, registered by the project training launcher."""

    def __init__(self, **kwargs: Any) -> None:
        pass

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        data["image"], data["valid_ratio"] = resize_recognition(data["image"])
        return data


class RecognitionCollator:
    """Pad the five official recognition fields to the widest image in a batch."""

    def __call__(self, batch: list[Any]) -> list[Any]:
        width = max(sample[0].shape[2] for sample in batch)
        images = np.zeros((len(batch), 3, REC_HEIGHT, width), dtype=np.float32)
        ratios = []
        for index, sample in enumerate(batch):
            own_width = sample[0].shape[2]
            images[index, :, :, :own_width] = sample[0]
            ratios.append(float(sample[4]) * own_width / width)
        return [
            images,
            *[np.stack([sample[field] for sample in batch]) for field in (1, 2, 3)],
            np.asarray(ratios, dtype=np.float32),
        ]
