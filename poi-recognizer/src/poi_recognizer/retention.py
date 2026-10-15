"""Keep best weights and one percent of paired examples after a completed run."""

import argparse
import json
import logging
import math
import random
import shutil
import time
from pathlib import Path
from typing import Any

import yaml


def _validate_examples(archive: Path) -> None:
    """Check a published archive before resuming interrupted cleanup."""
    manifest = json.loads((archive / "manifest.json").read_text(encoding="utf-8"))
    for task in ("det", "rec"):
        for split in ("train", "val"):
            name = f"{task}_{split}.txt"
            rows = (archive / name).read_text(encoding="utf-8").splitlines()
            if len(rows) != manifest["counts"][name]["retained"]:
                raise ValueError(f"Incomplete example labels: {archive / name}")
            for row in rows:
                image, _ = row.split("\t", 1)
                if not (archive / image).is_file():
                    raise ValueError(f"Missing example image: {archive / image}")


def _archive_examples(data: Path) -> None:
    """Publish a complete sample archive before removing any training input."""
    archive = data / "examples"
    if (archive / "manifest.json").is_file():
        _validate_examples(archive)
        return
    pending = data / ".examples-pending"
    if pending.exists():
        shutil.rmtree(pending)
    pending.mkdir()
    counts = {}
    try:
        for task in ("det", "rec"):
            for split in ("train", "val"):
                name = f"{task}_{split}.txt"
                rows = (data / name).read_text(encoding="utf-8").splitlines()
                indices = sorted(
                    random.Random(42).sample(range(len(rows)), math.ceil(len(rows) / 100))
                )
                sampled = []
                for index in indices:
                    image, label = rows[index].split("\t", 1)
                    source = Path(image)
                    if not source.is_absolute():
                        source = data / source
                    folder = "images" if task == "det" else "crops"
                    relative = Path(folder) / split / source.name
                    destination = pending / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
                    sampled.append(f"{relative.as_posix()}\t{label}\n")
                (pending / name).write_text("".join(sampled), encoding="utf-8")
                counts[name] = {"original": len(rows), "retained": len(indices)}
        (pending / "manifest.json").write_text(
            json.dumps({"fraction": 0.01, "seed": 42, "counts": counts}, indent=2),
            encoding="utf-8",
        )
        pending.rename(archive)
    except BaseException:
        shutil.rmtree(pending, ignore_errors=True)
        raise


def compact_run(run: Path) -> None:
    """Compact a finished two-stage run; repeated calls keep the same sample."""
    stages = (run / "rects", run)
    # Check every stage before touching the first. Callers must ensure training stopped.
    for stage in stages:
        for task in ("det", "rec"):
            weight = stage / task / "best_accuracy.pdparams"
            if not weight.is_file() or not weight.stat().st_size:
                raise ValueError(f"Missing best weights: {weight}")
        data = stage / "data"
        if not (data / "examples/manifest.json").is_file():
            for task in ("det", "rec"):
                for split in ("train", "val"):
                    if not (data / f"{task}_{split}.txt").is_file():
                        raise ValueError(f"Missing training labels in {data}")
    # Copy both archives successfully before deleting full caches or checkpoints.
    for stage in stages:
        _archive_examples(stage / "data")
    for stage in stages:
        data = stage / "data"
        if (data / "crops").exists():
            shutil.rmtree(data / "crops")
        for task in ("det", "rec"):
            for split in ("train", "val"):
                (data / f"{task}_{split}.txt").unlink(missing_ok=True)
            model = stage / task
            if (model / "best_model").exists():
                shutil.rmtree(model / "best_model")
            for pattern in ("*.pdopt", "latest.*", "iter_epoch_*", "config.yml"):
                for path in model.glob(pattern):
                    if path.is_file():
                        path.unlink()
        logging.info("已精简 %s：最佳权重、配置、日志、数据报告及 1%% 图像/标签样例", stage)


def _process_token(pid: int) -> str | None:
    """Identify a live Linux process without confusing a reused PID with training."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except FileNotFoundError:
        return None
    return None if fields[0] == "Z" else fields[19]


def _ensure_finished(run: Path) -> None:
    """Reject live or failed old runs before destructive manual/deferred cleanup."""
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            command = (process / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if any(b"_paddle_train.py" in arg for arg in command) and any(
            str(run).encode() + b"/" in arg for arg in command
        ):
            raise ValueError(f"Training is still using {run}")
    for stage in (run / "rects", run):
        for task in ("det", "rec"):
            config: dict[str, Any] = yaml.safe_load((stage / f"{task}.yml").read_text())
            epochs = int(config["Global"]["epoch_num"])
            expected = str(stage / task / f"iter_epoch_{epochs}")
            with (stage / f"{task}.log").open() as log:
                completed = any(
                    "save model in " in line and line.rstrip().endswith(expected) for line in log
                )
            if not completed:
                raise ValueError(f"Training did not finish all epochs: {stage}/{task}")


def main() -> None:
    """Archive an old completed run, optionally waiting for its known parent PID."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--wait-pid", type=int)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run = args.run.resolve()
    if args.wait_pid is not None:
        if args.wait_pid <= 0:
            parser.error("--wait-pid must be positive")
        token = _process_token(args.wait_pid)
        logging.info("等待训练进程 %d 结束，再检查并精简 %s", args.wait_pid, run)
        while token is not None and _process_token(args.wait_pid) == token:
            time.sleep(30)
    try:
        _ensure_finished(run)
        compact_run(run)
    except Exception:
        logging.exception("未完成精简；训练未结束或失败时保留完整缓存和检查点")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
