"""ReCTS det/rec training followed by supervised fine-tuning on train_images."""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Sequence

import yaml

from .dataset import prepare_data
from .training import train, training_config


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the linear training pipeline; --smoke bounds data and epochs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train.json"))
    parser.add_argument(
        "--smoke", action="store_true", help="Use four images per split and one epoch"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    started = time.monotonic()
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    stage = "检查配置和输入文件"

    try:
        config = json.loads(args.config.read_text())
        root = (args.config.resolve().parent / config.get("project_root", "..")).resolve()
        paths = {
            key: (root / config[key]).resolve()
            for key in (
                "rects_images",
                "rects_labels",
                "images",
                "labels",
                "paddleocr_source",
                "pretrained",
                "output",
            )
        }
        source = paths["paddleocr_source"]
        for key in ("images", "rects_images", "rects_labels"):
            if not paths[key].is_dir():
                raise FileNotFoundError(f"Directory does not exist ({key}): {paths[key]}")
        if len(list(paths["rects_labels"].glob("*.json"))) < 2:
            raise ValueError("rects_labels must contain per-image ReCTS JSON annotations")
        dictionary = source / "ppocr/utils/dict/ppocrv5_dict.txt"
        required = [paths["labels"], dictionary, source / "tools/train.py"]
        for task in ("det", "rec"):
            required.extend(
                [
                    source / f"configs/{task}/PP-OCRv5/PP-OCRv5_mobile_{task}.yml",
                    paths["pretrained"] / f"PP-OCRv5_mobile_{task}_pretrained.pdparams",
                ]
            )
        for path in required:
            if not path.is_file():
                raise FileNotFoundError(path)
        for key in (
            "images",
            "labels",
            "rects_images",
            "rects_labels",
            "paddleocr_source",
            "pretrained",
        ):
            if (
                paths["output"] == paths[key]
                or paths[key] in paths["output"].parents
                or paths["output"] in paths[key].parents
            ):
                raise ValueError(f"Output must be outside {key}")
        epochs = 1 if args.smoke else int(config.get("epochs", 50))
        rects_epochs = 1 if args.smoke else int(config.get("rects_epochs", epochs))
        batch = 2 if args.smoke else int(config.get("batch_size", 8))
        rate = float(config.get("learning_rate", 0.0001))
        gpu = int(config.get("gpu", 0))
        if min(epochs, rects_epochs) < 1 or batch < 1 or not 0 < rate < 1 or gpu < 0:
            raise ValueError("Invalid epochs, batch_size, learning_rate or gpu")
        run = paths["output"] / (("smoke-" if args.smoke else "train-") + run_id)
        run.mkdir(parents=True, exist_ok=False)
        (run / "settings.json").write_text(json.dumps(config, indent=2))
        rec = yaml.safe_load((source / "configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml").read_text())
        logging.info("本次输出：%s", run)
        logging.info(
            "训练顺序：ReCTS 检测 → ReCTS 识别 → LSVT 检测 → LSVT 识别\n"
            "每个 ReCTS 模型 %d 轮，每个 LSVT 模型 %d 轮；batch=%d，GPU=%d",
            rects_epochs,
            epochs,
            batch,
            gpu,
        )
        step = 0
        for dataset, stage_run, stage_epochs in (
            ("rects", run / "rects", rects_epochs),
            ("lsvt", run, epochs),
        ):
            stage_run.mkdir(exist_ok=True)
            step += 1
            stage = f"[{step}/6] 准备 {dataset.upper()} 数据"
            # Validate both handoffs before starting either target model.
            if dataset == "lsvt":
                for task in ("det", "rec"):
                    weights = run / "rects" / task / "best_accuracy.pdparams"
                    if not weights.is_file():
                        raise FileNotFoundError(f"Missing ReCTS {task} transfer weights: {weights}")
            logging.info("\n%s", stage)
            report = prepare_data(
                paths["rects_images" if dataset == "rects" else "images"],
                paths["rects_labels" if dataset == "rects" else "labels"],
                stage_run / "data",
                dictionary,
                rec["Global"]["max_text_length"],
                args.smoke,
                dataset=dataset,
            )
            logging.info(
                "数据就绪：训练 %d 张，验证 %d 张；统计：%s",
                report["train"]["images"],
                report["val"]["images"],
                stage_run / "data/report.json",
            )
            for task in ("det", "rec"):
                step += 1
                name = "文字检测（det）" if task == "det" else "文字识别（rec）"
                stage = f"[{step}/6] {dataset.upper()} {name}"
                initial = (
                    run / "rects" / task / "best_accuracy.pdparams" if dataset == "lsvt" else None
                )
                effective = training_config(
                    source,
                    paths["pretrained"],
                    stage_run,
                    task,
                    stage_epochs,
                    batch,
                    rate,
                    initial_weights=initial,
                )
                logging.info(
                    "\n%s，共 %d 轮\n初始权重：%s\n日志：%s",
                    stage,
                    stage_epochs,
                    effective["Global"]["pretrained_model"],
                    stage_run / f"{task}.log",
                )
                train(effective, source, stage_run, task, gpu)
                logging.info("完成：%s；模型目录：%s", stage, stage_run / task)
        logging.info(
            "\n训练全部完成，用时 %.0f 秒\n最终检测权重：%s\n最终识别权重：%s",
            time.monotonic() - started,
            run / "det/best_accuracy.pdparams",
            run / "rec/best_accuracy.pdparams",
        )
        return 0
    except KeyboardInterrupt:
        logging.info("\n已中断：%s", stage)
        return 130
    except Exception as exc:
        logging.error("\n停止于 %s：%s: %s", stage, type(exc).__name__, exc)
        return 1
