"""Synthetic fixtures for offline reports; no model or GPU tasks run here."""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
from PIL import Image

from poi_recognizer import evaluate
from poi_recognizer.evaluation_report import (
    match_predictions,
    quality,
    timing_summary,
    write_report,
)


def label(x: int, text: str) -> dict[str, Any]:
    return {"points": [[x, 0], [x + 8, 0], [x + 8, 8], [x, 8]], "transcription": text}


def prediction(x: int, text: str) -> dict[str, Any]:
    return {"points": label(x, text)["points"], "text": text, "score": 0.9}


def test_matching_duplicates_ignored_missed_wrong_and_exact() -> None:
    labels = [label(0, "店 名"), label(10, "超市"), label(20, "漏检"), label(30, "###")]
    predictions = [
        prediction(0, "店名"),
        prediction(0, "重复"),
        prediction(10, "错字"),
        prediction(30, "忽略"),
        prediction(50, "误检"),
    ]
    counts = match_predictions(labels, predictions)
    assert counts == {"targets": 3, "predictions": 4, "detected": 2, "correct": 1}
    assert [p["status"] for p in predictions] == [
        "correct",
        "false_positive",
        "wrong_text",
        "ignored",
        "false_positive",
    ]
    assert labels[2]["status"] == "missed"
    assert predictions[2]["target"] == "超市"
    assert quality(counts, "correct")["precision"] == 0.25
    assert quality(counts, "detected")["recall"] == 2 / 3


def test_empty_predictions_and_empty_labels() -> None:
    assert match_predictions([], []) == dict(targets=0, predictions=0, detected=0, correct=0)
    labels = [label(0, "未知字" * 30)]
    assert match_predictions(labels, [prediction(0, "错")])["targets"] == 1
    assert quality(dict(targets=0, predictions=0, correct=0), "correct")["f1"] == 0
    assert match_predictions([label(0, "店")], [])["detected"] == 0
    predictions = [prediction(0, "店")]
    assert match_predictions([], predictions)["predictions"] == 1
    assert predictions[0]["status"] == "false_positive"


def record() -> dict[str, Any]:
    return {
        "id": "0000-gt_1",
        "labels": [label(0, "店<script>alert(1)</script>")],
        "predictions": [prediction(0, '店<img src=x onerror="alert(1)">')],
        "seconds": dict(read=0.01, det=0.1, crop=0.02, rec=0.08, pipeline=0.2),
    }


def test_timing_and_html_escaping(tmp_path: Path) -> None:
    data = record()
    match_predictions(data["labels"], data["predictions"])
    summary = timing_summary([data, copy.deepcopy(data)])
    assert summary["images_per_second"] == 5
    assert summary["pipeline"]["mean_ms"] == 200
    assert summary["read"]["p95_ms"] == 10
    write_report(tmp_path, [data], summary)
    page = (tmp_path / "index.html").read_text()
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;" in page and "&lt;img" in page
    assert "images/0000-gt_1-predictions.jpg" in page
    assert 'value="missed"' in page
    assert "<th>预测文字</th><th>真实标注</th>" in page
    assert "<h3>逐条识别对照</h3>" in page
    assert "✗ 识别错误" in page


def fake_run(tmp_path: Path) -> tuple[Path, Path]:
    run, source = tmp_path / "run", tmp_path / "source"
    source.mkdir()
    for task in ("det", "rec"):
        (run / task).mkdir(parents=True)
        (run / task / "best_accuracy.pdparams").write_bytes(b"synthetic checkpoint")
        (run / f"{task}.yml").write_text(yaml.safe_dump({"Global": {"checkpoints": "old"}}))
        metric = "hmean" if task == "det" else "acc"
        (run / f"{task}.log").write_text(
            f"epoch: [1/50], loss: 2.0, ips: 20.0\ncur metric, {metric}: 0.6\n"
        )
    return run, source


def test_snapshot_and_source_readonly(tmp_path: Path) -> None:
    run, source = fake_run(tmp_path)
    before = {p: p.read_bytes() for p in run.rglob("*") if p.is_file()}
    output = tmp_path / "report"
    output.mkdir()
    evaluate.snapshot_models(run, output, source)
    config = yaml.safe_load((output / "models/rec/export.yml").read_text())
    assert config["Global"]["checkpoints"] is None
    assert config["Global"]["pretrained_model"] == str(output / "models/rec/weights")
    assert before == {p: p.read_bytes() for p in before}
    assert len(json.loads((output / "snapshot.json").read_text())) == 4


def test_snapshot_rejects_concurrent_save(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run, source = fake_run(tmp_path)
    output = tmp_path / "report"
    output.mkdir()
    original = evaluate.shutil.copy2

    def changing_copy(src: Path, dst: Path) -> Any:
        result = original(src, dst)
        src.write_bytes(b"concurrent training update")
        return result

    monkeypatch.setattr(evaluate.shutil, "copy2", changing_copy)
    with pytest.raises(RuntimeError, match="changed while copying"):
        evaluate.snapshot_models(run, output, source)


def test_synthetic_command_warmup_report_and_bad_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, source = fake_run(tmp_path)
    (run / "data").mkdir()
    image = tmp_path / "gt_10500.jpg"
    Image.new("RGB", (64, 32), "white").save(image)
    missing = tmp_path / "missing.jpg"
    (run / "data/det_val.txt").write_text(
        f'{image}\t{json.dumps([label(0, "超市"), label(10, "电话123456")])}\n{missing}\t[]\n'
    )
    output = tmp_path / "report"

    class Cuda:
        @staticmethod
        def get_device_name() -> str:
            return "synthetic GPU"

        @staticmethod
        def max_memory_allocated() -> int:
            return 0

        max_memory_reserved = max_memory_allocated

    class Paddle:
        __version__ = "synthetic"

        class device:
            cuda = Cuda

    calls = []

    def fake_infer(*args: Any) -> Any:
        calls.append(1)
        return [prediction(0, "超市"), prediction(10, "电话123456")], dict(
            det=0.1, crop=0.02, rec=0.08, pipeline=0.2
        )

    monkeypatch.setattr(evaluate, "export_model", lambda *args, **kwargs: None)
    monkeypatch.setattr(evaluate, "build_predictor", lambda *args: (None, None, Paddle))
    monkeypatch.setattr(evaluate, "infer", fake_infer)
    # One valid image only during warmup; the missing image is reported during measured evaluation.
    result = evaluate.main(
        ["--run", str(run), "--source", str(source), "--output", str(output), "--warmup", "1"]
    )
    assert result == 0
    summary = json.loads((output / "summary.json").read_text())
    assert summary["images"] == 1 and len(summary["skipped"]) == 1
    assert summary["end_to_end"]["f1"] == 1
    assert summary["counts"]["targets"] == summary["counts"]["predictions"] == 2
    assert "poi_filter" not in summary
    assert "电话123456" in (output / "predictions.jsonl").read_text()
    assert summary["timing"]["images_per_second"] == 5
    assert len(calls) == 2  # Warmup excluded from measured counts and times.
    assert (output / "images/0000-gt_10500-predictions.jpg").is_file()
    assert (output / "curves.html").is_file()
    assert evaluate.main(["--run", str(run), "--output", str(run / "report")]) == 1
    assert not (run / "report").exists()


def test_export_failure_shows_log_tail(tmp_path: Path) -> None:
    """A tiny Python child exercises real process/log handling without importing Paddle."""
    source = tmp_path / "source"
    (source / "tools").mkdir(parents=True)
    (source / "tools/export_model.py").write_text(
        'import sys\nsys.stderr.write("synthetic export error\\n")\nsys.exit(3)\n'
    )
    with pytest.raises(RuntimeError, match="synthetic export error"):
        evaluate.export_model("det", source, tmp_path, dict(os.environ))
    assert "synthetic export error" in (tmp_path / "det-export.log").read_text()


def test_export_wait_status_and_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    class ExportProcess:
        def __init__(self, command: list[str], **kwargs: Any) -> None:
            assert command[:2] == [sys.executable, "-u"]
            assert kwargs["stderr"] == subprocess.STDOUT
            kwargs["stdout"].write("initializing device\n")
            kwargs["stdout"].flush()
            self.waits = 0

        def __enter__(self) -> "ExportProcess":
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def wait(self, timeout: int) -> int:
            assert timeout == 5
            self.waits += 1
            if self.waits == 1:
                raise subprocess.TimeoutExpired("synthetic export", timeout)
            return 0

    monkeypatch.setattr(evaluate.subprocess, "Popen", ExportProcess)
    with caplog.at_level("INFO"):
        evaluate.export_model("det", tmp_path, tmp_path, {})
    assert "still running" in caplog.text
    assert "initializing device" in caplog.text
    assert "Exported det" in caplog.text


def test_ctrl_c_cleans_export_and_returns_130(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    run, source = fake_run(tmp_path)
    (run / "data").mkdir()
    (run / "data/det_val.txt").write_text("missing.jpg\t[]\n")
    actions = []

    class InterruptedExport:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.waits = 0

        def __enter__(self) -> "InterruptedExport":
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def wait(self, timeout: int) -> int:
            self.waits += 1
            if self.waits == 1:
                raise KeyboardInterrupt
            if self.waits == 2:
                raise subprocess.TimeoutExpired("synthetic export", timeout)
            return -9

        def terminate(self) -> None:
            actions.append("terminate")

        def kill(self) -> None:
            actions.append("kill")

    monkeypatch.setattr(evaluate.subprocess, "Popen", InterruptedExport)
    output = tmp_path / "report"
    with caplog.at_level("INFO"):
        code = evaluate.main(["--run", str(run), "--source", str(source), "--output", str(output)])
    assert code == 130
    assert actions == ["terminate", "kill"]
    assert "Evaluation interrupted" in caplog.text
    assert "Traceback" not in caplog.text
    assert (output / "models/det/weights.pdparams").is_file()


def test_report_shows_region_predictions_ground_truth_and_misses(tmp_path: Path) -> None:
    data = record()
    data["labels"] = [label(0, "超市"), label(10, "餐厅"), label(20, "药店")]
    data["predictions"] = [prediction(0, "超巿"), prediction(10, "餐厅")]
    data["image_shape"] = [32, 64, 3]
    match_predictions(data["labels"], data["predictions"])
    write_report(tmp_path, [data], {})
    page = (tmp_path / "index.html").read_text()
    assert "超巿" in page and "超市" in page
    assert "✓ 识别正确" in page and "✗ 识别错误" in page
    assert "—（未检出）" in page and "药店" in page and "✗ 漏检" in page
    assert page.count('class="crop"') == 3
    assert 'href="images/0000-gt_1-original.jpg"' in page
    assert "<details><summary>文字" not in page


def test_illegibility_excluded_from_scores_drawings_and_report(tmp_path: Path) -> None:
    from poi_recognizer.evaluation_report import draw_boxes

    labels = [
        dict(label(0, "不可辨认标注"), illegibility=True),
        dict(label(20, "合法文字"), illegibility=False),
    ]
    predictions = [prediction(0, "忽略区域预测"), prediction(20, "合法文字")]
    counts = match_predictions(labels, predictions)
    assert counts == dict(targets=1, predictions=1, detected=1, correct=1)
    assert labels[0]["status"] == predictions[0]["status"] == "ignored"
    data = record()
    data.update(labels=labels, predictions=predictions)
    write_report(tmp_path, [data], {})
    page = (tmp_path / "index.html").read_text()
    assert "忽略区域预测" not in page and "不可辨认标注" not in page
    assert "合法文字" in page and "预测 2" in page
    assert 'value="ignored"' not in page
    # An ignored-only drawing must contain no frame or numbered badge.
    image = Image.new("RGB", (64, 32), "white")
    draw_boxes(image, predictions[:1], tmp_path / "ignored.png")
    with Image.open(tmp_path / "ignored.png") as rendered:
        assert rendered.tobytes() == image.tobytes()
    draw_boxes(image, predictions, tmp_path / "visible.png")
    with Image.open(tmp_path / "visible.png") as rendered:
        assert rendered.tobytes() != image.tobytes()


def test_pretrained_snapshot_uses_original_weights_without_best_checkpoint(tmp_path: Path) -> None:
    run, source = fake_run(tmp_path)
    pretrained = tmp_path / "pretrained"
    pretrained.mkdir()
    for task in ("det", "rec"):
        (pretrained / f"PP-OCRv5_mobile_{task}_pretrained.pdparams").write_bytes(task.encode())
        (run / task / "best_accuracy.pdparams").unlink()
    output = tmp_path / "report"
    output.mkdir()
    evaluate.snapshot_models(run, output, source, pretrained)
    for task in ("det", "rec"):
        assert (output / f"models/{task}/weights.pdparams").read_bytes() == task.encode()
    snapshot = json.loads((output / "snapshot.json").read_text())
    assert all("pretrained" in item["source"] for item in snapshot[:2])


def test_curves_include_losses_learning_rate_and_visible_small_changes(tmp_path: Path) -> None:
    run, _ = fake_run(tmp_path)
    (run / "rec.log").write_text(
        "epoch: [1/50], lr: 0.0001, CTCLoss: 6.0, NRTRLoss: 2.0, loss: 8.0, ips: 30.0\n"
        "cur metric, acc: 0.59, norm_edit_dis: 0.79\n"
        "epoch: [2/50], lr: 0.00009, CTCLoss: 2.0, NRTRLoss: 1.0, loss: 3.0, ips: 30.0\n"
        "cur metric, acc: 0.65, norm_edit_dis: 0.83\n"
    )
    evaluate.training_curves(run, tmp_path)
    page = (tmp_path / "curves.html").read_text()
    rows = json.loads((tmp_path / "history.json").read_text())
    assert (
        "rec loss" in page and "rec CTCLoss" in page and "rec NRTRLoss" in page and "rec lr" in page
    )
    assert "+6.00 个百分点" in page
    assert "-5" in page and "epoch 2: 65.00%" in page
    assert "非整轮均值" in page and "舍入" in page
    assert rows[-1]["lr"] == 0.00009
    assert rows[-1]["CTCLoss"] == 2.0
    assert rows[-1]["loss"] == 3.0
