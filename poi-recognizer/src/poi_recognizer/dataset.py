"""Prepare only the train/validation inputs needed by official PaddleOCR."""

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import cv2

from .errors import POIError
from .geometry import crop_text, read_image, validate_polygon

SPLITS = {"train": range(0, 10500), "val": range(10500, 12000), "test": range(12000, 15000)}


def recognition_exclusion(text: str, vocabulary: set[str], max_length: int) -> str | None:
    """Reject whole labels instead of letting Paddle silently drop characters."""
    if not text.strip() or any(c in text for c in "\t\r\n"):
        return "empty_or_control_text"
    if any(c not in vocabulary for c in text):
        return "unknown_character"
    # Official NRTR reserves two positions; mobile CTC has 40 time steps.
    if len(text) > max_length - 2:
        return "long_text"
    if len(text) + sum(a == b for a, b in zip(text, text[1:])) > 40:
        return "ctc_length"
    return None


def prepare_data(
    images: Path,
    labels: Path,
    output: Path,
    dictionary: Path,
    max_length: int,
    smoke: bool = False,
    dataset: str = "lsvt",
) -> dict[str, Any]:
    """Prepare LSVT or native ReCTS line labels, splitting before checking images."""
    annotations = {}
    if dataset == "lsvt":
        annotations = json.loads(labels.read_text(encoding="utf-8"))
        splits = {key: [f"gt_{n}" for n in ids] for key, ids in SPLITS.items()}
    elif dataset == "rects":
        ids = sorted(path.stem for path in labels.glob("*.json"))
        if len(ids) < 2:
            raise ValueError(f"ReCTS needs at least two per-image JSON labels: {labels}")
        random.Random(42).shuffle(ids)
        boundary = min(len(ids) - 1, max(1, int(len(ids) * 0.9)))
        splits = {"train": ids[:boundary], "val": ids[boundary:]}
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    vocabulary = set(dictionary.read_text(encoding="utf-8").splitlines()) | {" "}
    output.mkdir(parents=True, exist_ok=False)
    crops = output / "crops"
    crops.mkdir()
    report: dict[str, Any] = {
        "dataset": dataset,
        "smoke": smoke,
        "planned": {k: len(v) for k, v in splits.items()},
    }
    reasons: Counter[str] = Counter()
    # Keep test IDs isolated. Test images and annotations do not enter training.
    for split, ids in splits.items():
        (output / f"{split}_ids.txt").write_text(
            "".join(f"{image_id}\n" for image_id in ids), encoding="utf-8"
        )
    with (output / "skipped.jsonl").open("w", encoding="utf-8") as skipped:

        def record(image_id: str, reason: str, index: int | None = None) -> None:
            reasons[reason] += 1
            skipped.write(
                json.dumps({"image_id": image_id, "instance": index, "reason": reason}) + "\n"
            )

        for split in ("train", "val"):
            used: list[str] = []
            det_count = rec_count = 0
            with (
                (output / f"det_{split}.txt").open("w", encoding="utf-8") as det_file,
                (output / f"rec_{split}.txt").open("w", encoding="utf-8") as rec_file,
            ):
                for image_id in splits[split]:
                    if dataset == "rects":
                        label_path = labels / f"{image_id}.json"
                        items = json.loads(label_path.read_text(encoding="utf-8"))["lines"]
                    else:
                        if image_id not in annotations:
                            raise ValueError(f"Missing label: {image_id}")
                        items = annotations[image_id]
                    image_path = images / f"{image_id}.jpg"
                    try:
                        image = read_image(image_path)
                    except POIError:
                        record(image_id, "unavailable_image")
                        continue
                    used.append(image_id)
                    detection = []
                    for index, item in enumerate(items):
                        try:
                            points = item["points"]
                            if dataset == "rects":
                                if len(points) != 8:
                                    raise POIError(
                                        "input_invalid", "geometry", "ReCTS needs eight coordinates"
                                    )
                                points = [points[i : i + 2] for i in range(0, 8, 2)]
                            polygon = validate_polygon(points, image.shape[1], image.shape[0])
                        except (POIError, KeyError, TypeError):
                            record(image_id, "invalid_polygon", index)
                            continue
                        text = item["transcription"]
                        ignored = (
                            item["ignore" if dataset == "rects" else "illegibility"]
                            or text in ("###", "*")
                            or not text.strip()
                        )
                        detection.append(
                            {"points": polygon, "transcription": "###" if ignored else text}
                        )
                        if ignored:
                            record(image_id, "ignored_text", index)
                            continue
                        reason = recognition_exclusion(text, vocabulary, max_length)
                        if reason:
                            record(image_id, reason, index)
                            continue
                        crop_path = crops / f"{image_id}_{index}.jpg"
                        try:
                            crop = crop_text(image, polygon)
                        except POIError:
                            record(image_id, "invalid_crop", index)
                            continue
                        if not cv2.imwrite(str(crop_path), crop):
                            raise OSError(f"Cannot save crop: {crop_path}")
                        rec_file.write(f"{crop_path}\t{text}\n")
                        rec_count += 1
                    if any(item["transcription"] != "###" for item in detection):
                        det_file.write(
                            f"{image_path}\t{json.dumps(detection, ensure_ascii=False)}\n"
                        )
                        det_count += 1
                    else:
                        record(image_id, "no_detection_target")
                    if smoke and len(used) >= 4:
                        break
            report[split] = {
                "images": len(used),
                "det": det_count,
                "rec": rec_count,
                "used_ids": used,
            }
    report["excluded"] = dict(reasons)
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for split in ("train", "val"):
        if not report[split]["det"] or not report[split]["rec"]:
            raise ValueError(f"No usable {split} inputs; see {output / 'report.json'}")
    return report
