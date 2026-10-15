"""Configure and invoke the two official training scripts sequentially."""

import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from .training_output import TrainingOutput


def training_config(
    source: Path,
    pretrained: Path,
    run: Path,
    task: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    initial_weights: Path | None = None,
) -> dict[str, Any]:
    """Adapt fixed PP-OCRv5 mobile YAMLs without changing model architecture."""
    model = f"PP-OCRv5_mobile_{task}"
    config = yaml.safe_load((source / f"configs/{task}/PP-OCRv5/{model}.yml").read_text())
    count = len((run / f"data/{task}_train.txt").read_text().splitlines())
    config["Global"].update(
        use_gpu=True,
        distributed=False,
        epoch_num=epochs,
        seed=42,
        save_model_dir=str(run / task),
        pretrained_model=str(initial_weights or pretrained / f"{model}_pretrained.pdparams"),
        checkpoints=None,
        save_epoch_step=epochs,
        eval_batch_step=[0, max(1, math.ceil(count / batch_size))],
        eval_batch_epoch=1,
        print_batch_step=1,
        use_visualdl=False,
    )
    config["Optimizer"]["lr"].update(learning_rate=learning_rate, warmup_epoch=0)
    for section, split in (("Train", "train"), ("Eval", "val")):
        dataset = config[section]["dataset"]
        dataset.update(
            data_dir=str(run / "data"), label_file_list=[str(run / f"data/{task}_{split}.txt")]
        )
        # Keep the official worker counts so CPU transforms can feed the GPU.
        config[section]["loader"].update(
            batch_size_per_card=batch_size if section == "Train" else 1,
            drop_last=False,
        )
        # CopyPaste assumes quads; concatenation may exceed the mobile CTC budget.
        dataset["transforms"] = [
            op for op in dataset["transforms"] if not ({"CopyPaste", "RecConAug"} & op.keys())
        ]
        for op in dataset["transforms"]:
            for settings in op.values():
                if isinstance(settings, dict) and "total_epoch" in settings:
                    settings["total_epoch"] = epochs
    if task == "rec":
        config["Global"]["character_dict_path"] = str(source / "ppocr/utils/dict/ppocrv5_dict.txt")
        config["Train"]["sampler"].update(scales=[[320, 48]], first_bs=batch_size, fix_bs=True)
    return config


def train(config: dict[str, Any], source: Path, run: Path, task: str, gpu: int) -> None:
    """Run PaddleOCR with live progress, compact logs and native checkpoints."""
    config_path = run / f"{task}.yml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    # Each loader worker should not create a full machine's worth of threads.
    # Set before the child imports OpenCV / numerical libraries; honor overrides.
    for name in ("OPENCV_FOR_THREADS_NUM", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        environment.setdefault(name, "1")
    with (run / f"{task}.log").open("w") as log:
        command = [
            sys.executable,
            "-u",
            str(Path(__file__).with_name("_paddle_train.py")),
            str(source),
            "-c",
            str(config_path),
        ]
        with subprocess.Popen(
            command,
            cwd=source,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        ) as process:
            output = TrainingOutput(task, log, sys.stderr)
            assert process.stdout is not None
            try:
                for line in process.stdout:
                    output.consume(line)
            except BaseException:
                process.terminate()
                raise
            finally:
                output.finish()
            returncode = process.wait()
            if returncode:
                raise subprocess.CalledProcessError(returncode, command)
    # The official script can exit successfully with an empty loader.
    for suffix in ("pdparams", "pdopt", "states"):
        if not (run / task / f"latest.{suffix}").is_file():
            raise RuntimeError(f"Training produced no latest.{suffix}; see {run / f'{task}.log'}")
