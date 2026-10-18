"""Manually export saved OCR weights and inspect real validation images offline."""

import argparse
import json
import logging
import os
import random
import re
import resource
import shutil
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from .evaluation_report import draw_boxes, match_predictions, quality, timing_summary, write_report
from .geometry import crop_text, read_image
from .preprocessing import (
    DET_LIMIT_TYPE,
    DET_SIDE_LEN,
    recognition_width,
    resize_recognition,
)


def export_model(task: str, source: Path, output: Path, environment: dict[str, str]) -> None:
    """Run official export with bounded waits, visible status and child cleanup on Ctrl+C."""
    log_path = output / f"{task}-export.log"
    command = [
        sys.executable,
        "-u",
        str(source / "tools/export_model.py"),
        "-c",
        str(output / f"models/{task}/export.yml"),
    ]
    started = time.monotonic()
    logging.info("Exporting %s: initializing Paddle/CUDA, then exporting; log: %s", task, log_path)
    with (
        log_path.open("w") as log,
        subprocess.Popen(
            command, cwd=source, env=environment, stdout=log, stderr=subprocess.STDOUT
        ) as process,
    ):
        try:
            while True:
                try:
                    returncode = process.wait(timeout=5)
                    break
                except subprocess.TimeoutExpired:
                    with log_path.open("rb") as recent:
                        recent.seek(max(0, log_path.stat().st_size - 4096))
                        lines = recent.read().decode("utf-8", errors="replace").splitlines()
                    latest = next(
                        (line.strip() for line in reversed(lines) if line.strip()), "no log yet"
                    )
                    logging.info(
                        "Exporting %s: still running, %.0fs elapsed; %s",
                        task,
                        time.monotonic() - started,
                        latest[:300],
                    )
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            raise
    if returncode:
        with log_path.open("rb") as recent:
            recent.seek(max(0, log_path.stat().st_size - 6000))
            tail = recent.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{task} export exited with code {returncode}; log: {log_path}\n{tail}")
    logging.info("Exported %s in %.1fs", task, time.monotonic() - started)


def snapshot_models(run: Path, output: Path, source: Path, pretrained: Path | None = None) -> None:
    """Copy fixed weights/configs before export; reject checkpoints changed during copying."""
    weights = {
        task: (
            pretrained / f"PP-OCRv5_mobile_{task}_pretrained.pdparams"
            if pretrained is not None
            else run / task / "best_accuracy.pdparams"
        )
        for task in ("det", "rec")
    }
    paths = list(weights.values())
    paths += [run / f"{task}.yml" for task in ("det", "rec")]
    before = [(path.stat().st_size, path.stat().st_mtime_ns) for path in paths]
    for task in ("det", "rec"):
        target = output / "models" / task
        target.mkdir(parents=True)
        shutil.copy2(weights[task], target / "weights.pdparams")
        config = yaml.safe_load((run / f"{task}.yml").read_text())
        config["Global"].update(
            checkpoints=None,
            pretrained_model=str(target / "weights"),
            save_inference_dir=str(target / "inference"),
            save_model_dir=str(target),
            character_dict_path=str(source / "ppocr/utils/dict/ppocrv5_dict.txt"),
        )
        (target / "export.yml").write_text(yaml.safe_dump(config, allow_unicode=True))
    after = [(path.stat().st_size, path.stat().st_mtime_ns) for path in paths]
    if before != after:
        raise RuntimeError(
            "Checkpoint changed while copying; retry after the training save finishes"
        )
    (output / "snapshot.json").write_text(
        json.dumps(
            [
                {"source": str(path), "size": size, "mtime_ns": stamp}
                for path, (size, stamp) in zip(paths, before)
            ],
            indent=2,
        )
    )


def training_curves(run: Path, output: Path) -> dict[str, Any]:
    """Stream existing logs into small metric curves without reading entire files into memory."""
    rows: list[dict[str, Any]] = []
    panels = []
    training: dict[str, Any] = {}
    for task in ("det", "rec"):
        epoch = 0
        last_train: dict[str, float] = {}
        batches: deque[dict[str, float]] = deque(maxlen=1000)
        with (run / f"{task}.log").open(errors="replace") as log:
            for line in log:
                batch = re.search(r"epoch: \[(\d+)/(\d+)\]", line)
                if batch:
                    epoch = int(batch[1])
                    last_train = {
                        k: float(v)
                        for k, v in re.findall(
                            r"\b(loss|CTCLoss|NRTRLoss|lr|ips|avg_batch_cost): ([\d.eE+-]+)", line
                        )
                    }
                    batches.append(last_train)
                if "cur metric," in line:
                    current = line.split("cur metric,", 1)[1].split(" | best metric,", 1)[0]
                    metrics = {
                        k: float(v)
                        for k, v in re.findall(
                            r"\b(precision|recall|hmean|acc|norm_edit_dis|fps): ([\d.]+)", current
                        )
                    }
                    rows.append({"task": task, "epoch": epoch, **last_train, **metrics})
        training[task] = {
            "latest_epoch": epoch,
            "log_records": len(batches),
            "median_samples_per_second": (
                float(np.median([b["ips"] for b in batches if "ips" in b]))
                if any("ips" in b for b in batches)
                else None
            ),
            "median_batch_ms": (
                float(
                    np.median([b["avg_batch_cost"] for b in batches if "avg_batch_cost" in b])
                    * 1000
                )
                if any("avg_batch_cost" in b for b in batches)
                else None
            ),
            "units": "images" if task == "det" else "text crops",
            "scope": "Last up to 1000 training log records; per-batch in old logs, epoch summaries in compact logs. Not inference speed.",
        }
        for metric in (
            ("precision", "recall", "hmean", "loss", "lr")
            if task == "det"
            else ("acc", "norm_edit_dis", "loss", "CTCLoss", "NRTRLoss", "lr")
        ):
            points = [
                (row["epoch"], row[metric]) for row in rows if row["task"] == task and metric in row
            ]
            if not points:
                continue
            xmax = max(1, max(x for x, _ in points))
            values = [value for _, value in points]
            percentage = metric in ("precision", "recall", "hmean", "acc", "norm_edit_dis")
            low, high = min(values), max(values)
            padding = max((high - low) * 0.15, 0.002 if percentage else max(abs(high) * 0.05, 1e-8))
            ymin, ymax = max(0, low - padding), high + padding
            if percentage:
                ymax = min(1, ymax)
            coords = " ".join(
                f"{80+x/xmax*650:.1f},{220-(y-ymin)/(ymax-ymin)*190:.1f}" for x, y in points
            )

            def number(value: float) -> str:
                return f"{value * 100:.2f}%" if percentage else f"{value:.5g}"

            delta = points[-1][1] - points[0][1]
            change = f"{delta * 100:+.2f} 个百分点" if percentage else f"{delta:+.5g}"
            best = (max if percentage else min)(points, key=lambda pair: pair[1])
            markers = "".join(
                f'<circle cx="{80+x/xmax*650:.1f}" cy="{220-(y-ymin)/(ymax-ymin)*190:.1f}" '
                f'r="3" fill="#2563eb"><title>epoch {x}: {number(y)}</title></circle>'
                for x, y in points
            )
            panels.append(
                f"<h2>{task} {metric}</h2><p>首个 {number(points[0][1])}；末次 {number(points[-1][1])}；"
                f"变化 {change}；{'最高' if percentage else '最低'} {number(best[1])}（epoch {best[0]}）</p>"
                '<svg viewBox="0 0 780 260" width="780" style="max-width:100%">'
                '<path d="M80 25 V220 H750" fill="none" stroke="#888"/>'
                f'<text x="2" y="35">{number(ymax)}</text><text x="2" y="220">{number(ymin)}</text>'
                f'<text x="80" y="245">epoch 0</text><text x="700" y="245">{xmax}</text>'
                f'<polyline fill="none" stroke="#2563eb" stroke-width="2" points="{coords}"/>{markers}</svg>'
            )
    (output / "history.json").write_text(json.dumps(rows, indent=2))
    (output / "curves.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>训练曲线</title>'
        '<body style="font-family:system-ui;margin:30px"><a href="index.html">返回图片报告</a>'
        "<h1>训练内验证指标</h1><p>acc 是标注裁剪整条识别准确率；norm_edit_dis 是编辑相似度。"
        "纵轴按各项实际范围缩放，注意刻度；悬停数据点可看数值。loss/CTCLoss/NRTRLoss 为验证时最近一次训练滑动窗口统计，非整轮均值。lr 沿用日志精度，显示 0 可能只是舍入。原始数值见 history.json；不包含训练前 epoch 0 的基线，运行中日志仅代表读取时结果。</p>"
        + "".join(panels)
        + "</body>",
        encoding="utf-8",
    )

    return training


def build_predictor(
    source: Path, output: Path, side_len: int, rec_batch: int = 16
) -> tuple[Any, Any, Any]:
    """Build official FP32 GPU predictors with explicit mobile-rec input dimensions."""
    sys.path.insert(0, str(source))
    import paddle
    from tools.infer import utility
    from tools.infer.predict_det import TextDetector
    from tools.infer.predict_rec import TextRecognizer

    class DynamicTextRecognizer(TextRecognizer):
        def resize_norm_img(self, img: Any, max_wh_ratio: float) -> Any:
            return resize_recognition(img, recognition_width(max_wh_ratio))[0]

    if not paddle.is_compiled_with_cuda() or paddle.device.cuda.device_count() < 1:
        raise RuntimeError("A CUDA GPU is required")
    paddle.set_device("gpu:0")
    args = utility.init_args().parse_args([])
    args.use_gpu = True
    args.gpu_id = 0
    args.det_model_dir = str(output / "models/det/inference")
    args.rec_model_dir = str(output / "models/rec/inference")
    args.rec_char_dict_path = str(source / "ppocr/utils/dict/ppocrv5_dict.txt")
    args.rec_image_shape = "3,48,320"
    args.rec_batch_num = rec_batch
    args.det_limit_side_len = side_len
    args.det_limit_type = DET_LIMIT_TYPE
    args.use_tensorrt = False
    args.enable_mkldnn = False
    return TextDetector(args), DynamicTextRecognizer(args), paddle


def infer(
    image: Any, detector: Any, recognizer: Any, paddle: Any, profile_stages: bool = False
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Time completed OCR; stage synchronization is optional profiling overhead."""
    if profile_stages:
        paddle.device.cuda.synchronize()
    start = time.perf_counter()
    boxes, _ = detector(image)
    if profile_stages:
        paddle.device.cuda.synchronize()
    det_end = time.perf_counter()
    polygons = [] if boxes is None else list(boxes)
    crops = [crop_text(image, box.tolist()) for box in polygons]
    crop_end = time.perf_counter()
    results = recognizer(crops)[0] if crops else []
    if profile_stages:
        paddle.device.cuda.synchronize()
    rec_end = time.perf_counter()
    if len(results) != len(polygons):
        raise RuntimeError("Recognizer returned a different number of results than crops")
    predictions = [
        {"points": box.tolist(), "text": text, "score": float(score)}
        for box, (text, score) in zip(polygons, results)
    ]
    end = time.perf_counter()
    seconds = {"pipeline": end - start}
    if profile_stages:
        seconds.update(det=det_end - start, crop=crop_end - det_end, rec=rec_end - crop_end)
    # Official GPU predictors copy their results back to CPU before returning.
    return predictions, seconds


def main(argv: list[str] | None = None) -> int:
    """Create a new offline report; this command runs models only when invoked by the user."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data/vendor/PaddleOCR-sparse",
    )
    parser.add_argument("--weights", choices=("best", "pretrained"), default="best")
    parser.add_argument(
        "--pretrained",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data/models/pretrained",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int, default=100, help="Validation images; 0 selects all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--det-side-len", type=int, default=DET_SIDE_LEN)
    parser.add_argument("--rec-batch", type=int, default=16)
    parser.add_argument("--profile-stages", action="store_true", help="Synchronize each GPU stage")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    output: Path | None = None
    try:
        if (
            args.limit < 0
            or args.warmup < 0
            or args.gpu < 0
            or args.det_side_len < 32
            or args.rec_batch < 1
        ):
            raise ValueError("Invalid limit, warmup, gpu or det-side-len")
        run, source = args.run.resolve(), args.source.resolve()
        output = (
            args.output.resolve()
            if args.output
            else run.parent
            / (run.name + f"-eval-{args.weights}-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
        )
        # Reports must not land inside read-only datasets, vendor source or training runs.
        protected = [
            run,
            source,
            args.pretrained.resolve(),
            Path(__file__).resolve().parents[2] / "data/photo-lsvt",
            Path(__file__).resolve().parents[2] / "data/models",
        ]
        if any(output == path or path in output.parents for path in protected):
            raise ValueError("Output must be outside the training run, raw data, models and source")
        data = run / "data"
        retained_examples = (data / "examples/manifest.json").is_file()
        if retained_examples:
            data /= "examples"
            logging.warning("此训练目录已精简：评估仅使用保留的约 1% 验证样例，非完整验证集")
        examples = []
        for line in (data / "det_val.txt").read_text().splitlines():
            path, labels = line.split("\t", 1)
            image_path = Path(path)
            if not image_path.is_absolute():
                image_path = data / image_path
            examples.append((image_path, json.loads(labels)))
        if not examples:
            raise ValueError("No validation images")
        if args.limit and args.limit < len(examples):
            examples = random.Random(args.seed).sample(examples, args.limit)
        output.mkdir(parents=True, exist_ok=False)
        (output / "images").mkdir()
        (output / "selected_ids.json").write_text(json.dumps([path.stem for path, _ in examples]))
        environment = {**os.environ, "CUDA_VISIBLE_DEVICES": str(args.gpu)}
        for name in ("OPENCV_FOR_THREADS_NUM", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            environment.setdefault(name, "1")
            os.environ.setdefault(name, "1")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        logging.info("Report directory: %s", output)
        logging.info("Python: %s", sys.executable)
        logging.info("Weights: %s", args.weights)
        snapshot_models(
            run, output, source, args.pretrained.resolve() if args.weights == "pretrained" else None
        )
        for task in ("det", "rec"):
            export_model(task, source, output, environment)
        training = training_curves(run, output)
        logging.info("Initializing inference predictors")
        detector, recognizer, paddle = build_predictor(
            source, output, args.det_side_len, args.rec_batch
        )
        if args.warmup:
            logging.info("Warming up: %d iterations", args.warmup)
            warm_image = None
            for path, _ in examples:
                try:
                    warm_image = read_image(path)
                    break
                except Exception:
                    continue
            if warm_image is None:
                raise RuntimeError("No images could be decoded for warmup")
            for _ in range(args.warmup):
                infer(warm_image, detector, recognizer, paddle, args.profile_stages)
        records = []
        skipped = []
        counts = dict.fromkeys(("targets", "predictions", "detected", "correct"), 0)
        for index, (path, labels) in enumerate(examples):
            try:
                started = time.perf_counter()
                image = read_image(path)
                read_seconds = time.perf_counter() - started
            except Exception as exc:
                skipped.append({"image": str(path), "error": str(exc)})
                logging.warning("Skipped %s: %s", path, exc)
                continue
            predictions, seconds = infer(image, detector, recognizer, paddle, args.profile_stages)
            seconds["read"] = read_seconds
            matched = match_predictions(labels, predictions)
            for key in counts:
                counts[key] += matched[key]
            ident = f"{index:04d}-" + re.sub(r"[^a-zA-Z0-9_-]", "_", path.stem)
            record = {
                "id": ident,
                "source": str(path),
                "labels": labels,
                "predictions": predictions,
                "seconds": seconds,
                "image_shape": list(image.shape),
                "counts": matched,
            }
            records.append(record)
            rgb = Image.fromarray(image[:, :, ::-1])
            rgb.save(output / f"images/{ident}-original.jpg")
            if sys.stderr.isatty():
                sys.stderr.write(f"\rValidation {index + 1}/{len(examples)}")
                sys.stderr.flush()
        if sys.stderr.isatty():
            sys.stderr.write("\n")
        if not records:
            raise RuntimeError("No images could be decoded")
        summary = {
            "weights": args.weights,
            "images": len(records),
            "selected": len(examples),
            "skipped": skipped,
            "scope": "Prepared detection validation images; all-ignore/invalid images excluded during preparation. Unknown/long texts retained. Sampled unless limit=0.",
            "retained_examples_only": retained_examples,
            "counts": counts,
            "detection": quality(counts, "detected"),
            "end_to_end": quality(counts, "correct"),
            "timing": timing_summary(records),
            "training": training,
            "benchmark": {
                "gpu": paddle.device.cuda.get_device_name(),
                "physical_gpu": args.gpu,
                "paddle": paddle.__version__,
                "precision": "FP32",
                "image_batch": 1,
                "rec_batch": args.rec_batch,
                "recognition_resize": "height=48, proportional, width>=320 aligned to 8; batch padding",
                "profile_stages": args.profile_stages,
                "det_max_side": args.det_side_len,
                "warmup": args.warmup,
                "confidence_filter": "none; all predicted boxes evaluated",
                "det_thresholds": "official defaults (DB thresh=0.3, box_thresh=0.6)",
                "max_gpu_allocated_bytes": paddle.device.cuda.max_memory_allocated(),
                "max_gpu_reserved_bytes": paddle.device.cuda.max_memory_reserved(),
                "max_host_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            },
            "timing_scope": "Completed CPU-result wall time; extra GPU synchronization only with profile_stages. Pipeline includes detection/pre/postprocessing, cropping and recognition; excludes read, export, init, warmup, matching, drawing and report. Memory peaks include init/warmup. Paddle allocator counters may omit inference-engine allocations and are not total VRAM. Other GPU jobs may affect speed; not a roadside benchmark.",
            "run": str(run),
            "seed": args.seed,
        }
        if retained_examples:
            summary["scope"] = (
                "Only the retained approximately 1% validation examples; NOT the full "
                "validation set, including when limit=0. Not a full validation benchmark."
            )
        for record in records:
            ident = record["id"]
            with Image.open(output / f"images/{ident}-original.jpg") as rgb:
                draw_boxes(rgb, record["labels"], output / f"images/{ident}-labels.jpg")
                draw_boxes(rgb, record["predictions"], output / f"images/{ident}-predictions.jpg")
        (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        (output / "predictions.jsonl").write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
        )
        write_report(output, records, summary)
        logging.info("%s", json.dumps(summary, ensure_ascii=False, indent=2))
        logging.info("Open %s", output / "index.html")
        return 0
    except KeyboardInterrupt:
        logging.info("\nEvaluation interrupted; partial output retained at %s", output)
        return 130
    except Exception as exc:
        logging.error("Evaluation failed: %s; output: %s", exc, output)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
