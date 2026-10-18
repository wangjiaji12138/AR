"""Verify pixel-identical train/inference resize, padding and detection settings."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from poi_recognizer.preprocessing import (
    DET_LIMIT_TYPE,
    DET_SIDE_LEN,
    DynamicRecResize,
    RecognitionCollator,
    recognition_width,
    resize_recognition,
)
from poi_recognizer.training import training_config


def test_variable_width_batch_matches_inference_without_squeezing() -> None:
    rng = np.random.default_rng(42)
    images = [rng.integers(0, 256, (30, w, 3), dtype=np.uint8) for w in (60, 401, 1001)]
    samples = []
    for image in images:
        item = DynamicRecResize()({"image": image})
        samples.append(
            [item["image"], np.array([1, 2]), np.array([2, 1, 3]), 2, item["valid_ratio"]]
        )
    batch = RecognitionCollator()(samples)
    width = recognition_width(max(image.shape[1] / image.shape[0] for image in images))
    assert width > 960 and batch[0].shape == (3, 3, 48, width)
    for index, image in enumerate(images):
        direct, ratio = resize_recognition(image, width)
        np.testing.assert_array_equal(batch[0][index], direct)
        assert float(batch[4][index]) == pytest.approx(ratio)
    with pytest.raises(ValueError, match="compress"):
        resize_recognition(images[-1], 320)


def test_detection_validation_uses_inference_resize(tmp_path: Path) -> None:
    config = {
        "Global": {},
        "Optimizer": {"lr": {}},
        "Train": {"dataset": {"transforms": []}, "loader": {}},
        "Eval": {"dataset": {"transforms": [{"DetResizeForTest": None}]}, "loader": {}},
    }
    source = tmp_path / "source"
    path = source / "configs/det/PP-OCRv5/PP-OCRv5_mobile_det.yml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(config))
    run = tmp_path / "run"
    (run / "data").mkdir(parents=True)
    (run / "data/det_train.txt").write_text("image\t[]\n")
    result = training_config(source, tmp_path, run, "det", 1, 2, 0.0001, amp=False)
    assert result["Eval"]["dataset"]["transforms"] == [
        {"DetResizeForTest": {"limit_type": DET_LIMIT_TYPE, "limit_side_len": DET_SIDE_LEN}}
    ]
    assert result["Global"]["use_amp"] is False
