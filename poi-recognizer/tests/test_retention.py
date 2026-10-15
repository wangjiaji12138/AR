"""Exercise destructive retention only on synthetic temporary output trees."""

import json
import math
import random
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from poi_recognizer import retention


def make_run(tmp_path: Path, count: int = 200) -> tuple[Path, Path]:
    raw = tmp_path / "raw"
    raw.mkdir()
    for index in range(count):
        (raw / f"image_{index}.jpg").write_bytes(f"original {index}".encode())
    run = tmp_path / "run"
    for stage in (run / "rects", run):
        data = stage / "data"
        (data / "crops").mkdir(parents=True)
        for index in range(count):
            (data / f"crops/crop_{index}.jpg").write_bytes(f"crop {index}".encode())
        for name in (
            "train_ids.txt",
            "val_ids.txt",
            "test_ids.txt",
            "report.json",
            "skipped.jsonl",
        ):
            (data / name).write_text("original metadata\n")
        for task in ("det", "rec"):
            model = stage / task
            (model / "best_model").mkdir(parents=True)
            for name in (
                "best_accuracy.pdparams",
                "best_accuracy.pdopt",
                "best_accuracy.states",
                "latest.pdparams",
                "latest.pdopt",
                "latest.states",
                "iter_epoch_2.pdparams",
                "iter_epoch_2.pdopt",
                "iter_epoch_2.states",
                "best_model/model.pdparams",
                "config.yml",
            ):
                (model / name).write_bytes(b"synthetic weights/state")
            (stage / f"{task}.yml").write_text(yaml.safe_dump({"Global": {"epoch_num": 2}}))
            (stage / f"{task}.log").write_text(f"INFO: save model in {model}/iter_epoch_2\n")
            for split in ("train", "val"):
                rows = []
                for index in range(count):
                    path = (
                        raw / f"image_{index}.jpg"
                        if task == "det"
                        else data / f"crops/crop_{index}.jpg"
                    )
                    label = (
                        json.dumps([{"transcription": f"店{index}"}])
                        if task == "det"
                        else f"店{index}"
                    )
                    rows.append(f"{path}\t{label}\n")
                (data / f"{task}_{split}.txt").write_text("".join(rows))
    (run / "settings.json").write_text("{}")
    return run, raw


@pytest.mark.parametrize("count", [1, 101, 200])
def test_paired_examples_best_weights_metadata_and_idempotence(tmp_path: Path, count: int) -> None:
    run, raw = make_run(tmp_path, count)
    original = {path: path.read_bytes() for path in raw.iterdir()}
    retention._ensure_finished(run)
    retention.compact_run(run)
    for stage in (run / "rects", run):
        data = stage / "data"
        examples = data / "examples"
        manifest = json.loads((examples / "manifest.json").read_text())
        assert manifest["fraction"] == 0.01
        assert not (data / "crops").exists()
        assert (data / "report.json").read_text() == "original metadata\n"
        assert (data / "test_ids.txt").read_text() == "original metadata\n"
        for task in ("det", "rec"):
            model = stage / task
            assert sorted(path.name for path in model.iterdir()) == [
                "best_accuracy.pdparams",
                "best_accuracy.states",
            ]
            assert (stage / f"{task}.log").is_file()
            assert (stage / f"{task}.yml").is_file()
            for split in ("train", "val"):
                name = f"{task}_{split}.txt"
                assert not (data / name).exists()
                rows = (examples / name).read_text().splitlines()
                assert len(rows) == math.ceil(count / 100)
                assert manifest["counts"][name] == {"original": count, "retained": len(rows)}
                for row in rows:
                    image, text = row.split("\t", 1)
                    path = examples / image
                    index = int(path.stem.split("_")[-1])
                    assert (
                        path.read_bytes()
                        == f"{'original' if task == 'det' else 'crop'} {index}".encode()
                    )
                    assert (
                        json.loads(text)[0]["transcription"] if task == "det" else text
                    ) == f"店{index}"
    before = {path: path.read_bytes() for path in run.rglob("*") if path.is_file()}
    retention.compact_run(run)
    assert before == {path: path.read_bytes() for path in run.rglob("*") if path.is_file()}
    assert original == {path: path.read_bytes() for path in raw.iterdir()}


def test_missing_model_prevents_any_cleanup(tmp_path: Path) -> None:
    run, _ = make_run(tmp_path, 1)
    (run / "rec/best_accuracy.pdparams").unlink()
    with pytest.raises(ValueError, match="Missing best weights"):
        retention.compact_run(run)
    assert (run / "rects/data/crops/crop_0.jpg").is_file()
    assert (run / "rects/det/latest.pdparams").is_file()
    assert not (run / "rects/data/examples").exists()


@pytest.mark.parametrize("broken_stage", ["rects", "."])
def test_copy_failure_keeps_full_labels_caches_and_checkpoints(
    tmp_path: Path, broken_stage: str
) -> None:
    run, _ = make_run(tmp_path)
    selected = random.Random(42).sample(range(200), 2)[0]
    (run / broken_stage / f"data/crops/crop_{selected}.jpg").unlink()
    with pytest.raises(FileNotFoundError):
        retention.compact_run(run)
    assert (run / "rects/data/det_train.txt").is_file()
    assert len(list((run / "rects/data/crops").iterdir())) >= 199
    assert (run / "rects/rec/latest.pdopt").is_file()
    assert not (run / "rects/data/.examples-pending").exists()


def test_interrupted_archive_cleanup_resumes_without_resampling(tmp_path: Path) -> None:
    run, _ = make_run(tmp_path)
    retention._archive_examples(run / "rects/data")
    before = (run / "rects/data/examples/manifest.json").read_bytes()
    (run / "rects/data/det_train.txt").unlink()
    retention.compact_run(run)
    assert (run / "rects/data/examples/manifest.json").read_bytes() == before


def test_damaged_existing_archive_prevents_cache_deletion(tmp_path: Path) -> None:
    run, _ = make_run(tmp_path, 1)
    retention._archive_examples(run / "rects/data")
    (run / "rects/data/examples/crops/train/crop_0.jpg").unlink()
    with pytest.raises(ValueError, match="Missing example image"):
        retention.compact_run(run)
    assert (run / "rects/data/crops/crop_0.jpg").is_file()
    assert (run / "rects/rec/latest.pdopt").is_file()


def test_incomplete_old_training_cannot_be_compacted(tmp_path: Path) -> None:
    run, _ = make_run(tmp_path, 1)
    # Avoid accepting epoch 20 as epoch 2, or a best checkpoint before the final epoch.
    (run / "rec.log").write_text(f"INFO: save model in {run}/rec/iter_epoch_20\n")
    with pytest.raises(ValueError, match="did not finish"):
        retention._ensure_finished(run)
    assert (run / "rec/latest.pdopt").is_file()


def test_process_token_missing_or_current_process() -> None:
    import os

    assert retention._process_token(os.getpid()) is not None
    assert retention._process_token(2**30) is None


def test_live_training_process_blocks_cleanup(tmp_path: Path) -> None:
    run, _ = make_run(tmp_path, 1)
    # A tiny stdin reader supplies the same argv signature without importing Paddle.
    with subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys; sys.stdin.read()",
            "_paddle_train.py",
            str(run / "rec.yml"),
        ],
        stdin=subprocess.PIPE,
    ) as process:
        try:
            with pytest.raises(ValueError, match="still using"):
                retention._ensure_finished(run)
        finally:
            process.terminate()
            process.wait(timeout=5)
    assert (run / "rec/latest.pdopt").is_file()


def test_deferred_cleanup_waits_for_process_then_validates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = []
    tokens = iter(["original process", "original process", None])
    monkeypatch.setattr(sys, "argv", ["retention", "--run", str(tmp_path), "--wait-pid", "123"])
    monkeypatch.setattr(retention, "_process_token", lambda pid: next(tokens))
    monkeypatch.setattr(retention.time, "sleep", lambda seconds: events.append(("wait", seconds)))
    monkeypatch.setattr(retention, "_ensure_finished", lambda run: events.append(("validate", run)))
    monkeypatch.setattr(retention, "compact_run", lambda run: events.append(("compact", run)))
    retention.main()
    assert events == [("wait", 30), ("validate", tmp_path), ("compact", tmp_path)]
