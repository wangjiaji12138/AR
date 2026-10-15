"""Small original fixtures verify split isolation and the official-script boundary."""

import io
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from PIL import Image

from poi_recognizer.dataset import prepare_data, recognition_exclusion
from poi_recognizer.geometry import crop_text, validate_polygon
from poi_recognizer.training import train, training_config


def test_fixed_splits_missing_images_and_whole_label_filtering(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    polygon = [[1, 1], [30, 1], [30, 10], [1, 10]]
    labels = {
        f"gt_{i}": [{"points": polygon, "transcription": "店", "illegibility": False}]
        for i in range(15000)
    }
    labels["gt_10500"].append({"points": polygon, "transcription": "店外", "illegibility": False})
    for number in (10499, 10500, 11999, 12000):
        Image.new("RGB", (32, 16), color=(number % 255, 20, 30)).save(images / f"gt_{number}.jpg")
    label_path = tmp_path / "labels.json"
    label_path.write_text(json.dumps(labels))
    before = label_path.read_bytes()
    dictionary = tmp_path / "dict.txt"
    dictionary.write_text("店\n")
    output = tmp_path / "prepared"
    report = prepare_data(images, label_path, output, dictionary, 25)
    assert report["planned"] == {"train": 10500, "val": 1500, "test": 3000}
    assert report["train"]["used_ids"] == ["gt_10499"]
    assert report["val"]["used_ids"] == ["gt_10500", "gt_11999"]
    assert report["excluded"]["unknown_character"] == 1
    assert "店外" in (output / "det_val.txt").read_text()
    assert "店外" not in (output / "rec_val.txt").read_text()
    assert "gt_12000" not in (output / "det_val.txt").read_text()
    assert not list((output / "crops").glob("gt_12000*"))
    assert (output / "test_ids.txt").read_text().splitlines()[0] == "gt_12000"
    assert label_path.read_bytes() == before
    with pytest.raises(FileExistsError):
        prepare_data(images, label_path, output, dictionary, 25)


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("店", None),
        ("店外", "unknown_character"),
        ("店\t", "empty_or_control_text"),
        ("店" * 24, "long_text"),
        ("店" * 23, "ctc_length"),
    ],
)
def test_recognition_filters(text: str, reason: str | None) -> None:
    assert recognition_exclusion(text, {"店"}, 25) == reason


def test_native_polygon_crop_and_invalid_geometry() -> None:
    polygon = validate_polygon(
        [[1, 1], [8, 1], [15, 1], [15, 5], [15, 9], [8, 9], [1, 9], [1, 5]], 16, 10
    )
    crop = crop_text(np.full((10, 16, 3), 90, dtype=np.uint8), polygon)
    assert crop.shape[0] > 0 and crop.shape[1] > 0
    assert np.all(crop == 90)


@pytest.mark.parametrize("thread_override", [None, "2"])
def test_config_and_subprocess_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, thread_override: str | None
) -> None:
    thread_variables = ("OPENCV_FOR_THREADS_NUM", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
    for name in thread_variables:
        monkeypatch.delenv(name, raising=False)
        if thread_override is not None:
            monkeypatch.setenv(name, thread_override)
    source = tmp_path / "source"
    original: dict[str, Any] = {
        "Global": {},
        "Optimizer": {"lr": {}},
        "Architecture": {"unchanged": True},
        "Train": {
            "dataset": {"transforms": [{"RecConAug": {}}, {"RecAug": None}]},
            "loader": {"num_workers": 8},
            "sampler": {},
        },
        "Eval": {"dataset": {"transforms": []}, "loader": {"num_workers": 4}},
    }
    config_file = source / "configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml"
    config_file.parent.mkdir(parents=True)
    config_file.write_text(yaml.safe_dump(original))
    run = tmp_path / "run"
    (run / "data").mkdir(parents=True)
    (run / "data/rec_train.txt").write_text("a\tx\nb\ty\nc\tz\n")
    config = training_config(source, tmp_path / "weights", run, "rec", 1, 2, 0.0001)
    assert config["Architecture"] == original["Architecture"]
    assert config["Train"]["loader"]["num_workers"] == 8
    assert config["Eval"]["loader"]["num_workers"] == 4
    assert config["Global"]["eval_batch_step"] == [0, 2]
    assert config["Global"]["eval_batch_epoch"] == 1
    assert config["Train"]["sampler"]["scales"] == [[320, 48]]
    assert config["Train"]["dataset"]["transforms"] == [{"RecAug": None}]
    assert config["Eval"]["dataset"]["label_file_list"] == [str(run / "data/rec_val.txt")]

    def fail(command: list[str], **kwargs: Any) -> None:
        assert command[2].endswith("_paddle_train.py")
        assert command[3] == str(source)
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "2"
        for name in thread_variables:
            assert kwargs["env"][name] == (thread_override or "1")
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(subprocess, "Popen", fail)
    with pytest.raises(subprocess.CalledProcessError):
        train(config, source, run, "rec", 2)

    class CompletedProcess:
        stdout = io.StringIO("")

        def __enter__(self) -> "CompletedProcess":
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: CompletedProcess())
    with pytest.raises(RuntimeError, match="latest.pdparams"):
        train(config, source, run, "rec", 2)


@pytest.mark.parametrize("tty", [False, True])
def test_live_progress_compact_logs_and_partial_epoch(tty: bool) -> None:
    from poi_recognizer.training_output import TrainingOutput

    class Terminal(io.StringIO):
        def isatty(self) -> bool:
            return tty

    log, terminal = io.StringIO(), Terminal()
    output = TrainingOutput("rec", log, terminal)
    records = [
        "train dataloader has 3 iters",
        "epoch: [1/2], global_step: 1, loss: 3.0",
        "epoch: [1/2], global_step: 2, loss: 2.0",
        "epoch: [1/2], global_step: 3, loss: 1.0",
        "eval model:: 50%|#####| 1/2 [00:01<00:01]",
        "eval model:: 100%|##########| 2/2 [00:02<00:00]",
        "cur metric, acc: 0.8, norm_edit_dis: 0.9",
        "best metric, acc: 0.8, best_epoch: 1",
        "epoch: [2/2], global_step: 4, loss: 0.5",
        "WARNING: example warning",
        "Traceback: example failure",
    ]
    for record in records:
        output.consume(record)
    output.finish()
    saved = log.getvalue()
    assert "loss: 3.0" not in saved and "loss: 2.0" not in saved
    assert saved.count("train epoch:") == 2
    assert "loss: 1.0" in saved and "loss: 0.5" in saved
    assert "eval epoch 1: 2/2" in saved
    assert "1/2 [00:01" not in saved
    assert "acc: 0.8, norm_edit_dis: 0.9" in saved
    assert "WARNING: example warning" in saved and "Traceback: example failure" in saved
    assert "\r" not in saved
    assert ("\r" in terminal.getvalue()) == tty


def test_rects_lines_ignore_flags_and_stable_split(tmp_path: Path) -> None:
    images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    line = {"points": [1, 1, 30, 1, 30, 10, 1, 10], "transcription": "店", "ignore": 0}
    for index in range(10):
        ident = f"train_ReCTS_{index:06d}"
        Image.new("RGB", (32, 16)).save(images / f"{ident}.jpg")
        (labels / f"{ident}.json").write_text(
            json.dumps(
                {
                    "lines": [line, {**line, "ignore": 1}, {**line, "points": [1, 2]}],
                    "chars": [{**line, "transcription": "外"}],
                }
            )
        )
    dictionary = tmp_path / "dict.txt"
    dictionary.write_text("店\n")
    first = tmp_path / "first"
    report = prepare_data(images, labels, first, dictionary, 25, dataset="rects")
    assert report["planned"] == {"train": 7, "val": 1, "test": 2}
    assert report["train"]["rec"] == 7 and report["val"]["rec"] == 1
    assert report["excluded"] == {"ignored_text": 8, "invalid_polygon": 8}
    train_ids = (first / "train_ids.txt").read_text().splitlines()
    val_ids = (first / "val_ids.txt").read_text().splitlines()
    test_ids = (first / "test_ids.txt").read_text().splitlines()
    assert not set(train_ids) & set(val_ids)
    assert not set(test_ids) & (set(train_ids) | set(val_ids))
    assert len(set(train_ids + val_ids + test_ids)) == 10
    assert not (first / "det_test.txt").exists()
    for image_id in test_ids:
        assert not list((first / "crops").glob(f"{image_id}_*.jpg"))
        # Held-out labels and images must never be read during preparation.
        (labels / f"{image_id}.json").write_text("invalid JSON")
        (images / f"{image_id}.jpg").unlink()
    detections = [
        json.loads(row.split("\t")[1]) for row in (first / "det_train.txt").read_text().splitlines()
    ]
    assert all([item["transcription"] for item in row] == ["店", "###"] for row in detections)
    assert detections[0][0]["points"] == [[1, 1], [30, 1], [30, 10], [1, 10]]
    # Missing images cannot change membership or fill gaps from validation.
    (images / f"{train_ids[0]}.jpg").unlink()
    second = tmp_path / "second"
    changed = prepare_data(images, labels, second, dictionary, 25, dataset="rects")
    assert changed["train"]["images"] == 6
    assert (second / "train_ids.txt").read_text().splitlines() == train_ids
    assert (second / "val_ids.txt").read_text().splitlines() == val_ids
    assert (second / "test_ids.txt").read_text().splitlines() == test_ids


@pytest.mark.parametrize("missing_transfer", [False, True])
@pytest.mark.parametrize("smoke", [False, True])
def test_two_stage_training_handoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing_transfer: bool, smoke: bool
) -> None:
    from poi_recognizer import cli

    source, weights = tmp_path / "source", tmp_path / "weights"
    weights.mkdir()
    dictionary = source / "ppocr/utils/dict/ppocrv5_dict.txt"
    dictionary.parent.mkdir(parents=True)
    dictionary.write_text("店\n")
    (source / "tools").mkdir()
    (source / "tools/train.py").touch()
    for task in ("det", "rec"):
        original = {
            "Global": {"max_text_length": 25},
            "Optimizer": {"lr": {}},
            "Train": {"dataset": {"transforms": []}, "loader": {}, "sampler": {}},
            "Eval": {"dataset": {"transforms": []}, "loader": {}},
        }
        yaml_path = source / f"configs/{task}/PP-OCRv5/PP-OCRv5_mobile_{task}.yml"
        yaml_path.parent.mkdir(parents=True)
        yaml_path.write_text(yaml.safe_dump(original))
        (weights / f"PP-OCRv5_mobile_{task}_pretrained.pdparams").touch()
    for directory in ("images", "rects_images", "rects_labels"):
        (tmp_path / directory).mkdir()
    for index in range(3):
        (tmp_path / f"rects_labels/{index}.json").write_text("{}")
    (tmp_path / "labels.json").write_text("{}")
    config = tmp_path / "train.json"
    config.write_text(
        json.dumps(
            {
                "project_root": ".",
                "images": "images",
                "labels": "labels.json",
                "rects_images": "rects_images",
                "rects_labels": "rects_labels",
                "paddleocr_source": "source",
                "pretrained": "weights",
                "output": "output",
                "epochs": 7,
                "rects_epochs": 3,
            }
        )
    )
    events = []

    def prepare(*args: Any, **kwargs: Any) -> dict[str, Any]:
        events.append(("prepare", kwargs["dataset"]))
        assert args[5] is smoke
        output = args[2]
        output.mkdir()
        for task in ("det", "rec"):
            (output / f"{task}_train.txt").write_text("image\t店\n")
        return {"train": {"images": 1}, "val": {"images": 1}}

    def fake_train(effective: dict[str, Any], source: Path, run: Path, task: str, gpu: int) -> None:
        events.append(("train", run.name, task))
        initial = Path(effective["Global"]["pretrained_model"])
        assert initial.is_file()
        assert effective["Global"]["checkpoints"] is None
        assert effective["Global"]["epoch_num"] == (
            1 if smoke else (3 if run.name == "rects" else 7)
        )
        assert effective["Train"]["loader"]["batch_size_per_card"] == (2 if smoke else 8)
        if run.name == "rects":
            assert initial == weights / f"PP-OCRv5_mobile_{task}_pretrained.pdparams"
        else:
            assert initial == run / f"rects/{task}/best_accuracy.pdparams"
        (run / task).mkdir()
        if not (missing_transfer and task == "rec"):
            (run / task / "best_accuracy.pdparams").write_text("trained weights")

    monkeypatch.setattr(cli, "prepare_data", prepare)
    monkeypatch.setattr(cli, "train", fake_train)
    arguments = ["--config", str(config)] + (["--smoke"] if smoke else [])
    assert cli.main(arguments) == (1 if missing_transfer else 0)
    assert events[:3] == [
        ("prepare", "rects"),
        ("train", "rects", "det"),
        ("train", "rects", "rec"),
    ]
    if missing_transfer:
        assert len(events) == 3
    else:
        assert events[3] == ("prepare", "lsvt")
        assert [event[-1] for event in events[4:]] == ["det", "rec"]
