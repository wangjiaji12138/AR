"""Small offline OCR report: matching, timing summaries and browsable examples."""

import html
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import Polygon

COLORS = {
    "correct": "#16803a",
    "wrong_text": "#d97706",
    "false_positive": "#dc2626",
    "missed": "#2563eb",
    "ignored": "#888888",
}


def match_predictions(
    labels: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> dict[str, int]:
    """Greedily match descending polygon IoU >= .5; ignore predictions covered > .5."""
    targets = [Polygon(label["points"]) for label in labels]
    boxes = [Polygon(pred["points"]) for pred in predictions]
    for label, target in zip(labels, targets):
        ignored = label.get("illegibility", False) or label["transcription"] == "###"
        label["status"] = "ignored" if ignored else "missed"
        if not target.is_valid or target.area <= 0:
            raise ValueError("Invalid ground-truth polygon")
    for pred, box in zip(predictions, boxes):
        if not box.is_valid or box.area <= 0:
            raise ValueError("Invalid predicted polygon")
        pred["status"] = "false_positive"
        if any(
            label["status"] == "ignored" and box.intersection(target).area / box.area > 0.5
            for label, target in zip(labels, targets)
        ):
            pred["status"] = "ignored"
    candidates = []
    for i, (label, target) in enumerate(zip(labels, targets)):
        if label["status"] == "ignored":
            continue
        for j, (pred, box) in enumerate(zip(predictions, boxes)):
            if pred["status"] == "ignored":
                continue
            iou = target.intersection(box).area / target.union(box).area
            if iou >= 0.5:
                candidates.append((-iou, i, j))
    used_targets: set[int] = set()
    used_boxes: set[int] = set()
    correct = 0
    for _, i, j in sorted(candidates):
        if i in used_targets or j in used_boxes:
            continue
        used_targets.add(i)
        used_boxes.add(j)
        same = labels[i]["transcription"].replace(" ", "") == predictions[j]["text"].replace(
            " ", ""
        )
        status = "correct" if same else "wrong_text"
        labels[i]["status"] = predictions[j]["status"] = status
        predictions[j]["target"] = labels[i]["transcription"]
        predictions[j]["target_index"] = i
        correct += int(same)
    return {
        "targets": sum(label["status"] != "ignored" for label in labels),
        "predictions": sum(pred["status"] != "ignored" for pred in predictions),
        "detected": len(used_boxes),
        "correct": correct,
    }


def quality(counts: dict[str, int], key: str) -> dict[str, float]:
    """Return micro precision/recall/F1 from explicit counts, including empty sets."""
    tp, predicted, targets = counts[key], counts["predictions"], counts["targets"]
    return {
        "precision": tp / predicted if predicted else 0.0,
        "recall": tp / targets if targets else 0.0,
        "f1": 2 * tp / (predicted + targets) if predicted + targets else 0.0,
    }


def timing_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate measured images only; total throughput includes detection and recognition."""
    result: dict[str, Any] = {}
    for key in ("read", "det", "crop", "rec", "pipeline"):
        values = [record["seconds"][key] for record in records]
        result[key] = {
            "mean_ms": float(np.mean(values) * 1000),
            "p50_ms": float(np.percentile(values, 50) * 1000),
            "p95_ms": float(np.percentile(values, 95) * 1000),
        }
    result["images_per_second"] = len(records) / sum(r["seconds"]["pipeline"] for r in records)
    result["mean_boxes_per_image"] = float(np.mean([len(r["predictions"]) for r in records]))
    return result


def draw_boxes(image: Image.Image, items: list[dict[str, Any]], path: Path) -> None:
    """Draw non-ignored regions with visible outlines; retain original row indices."""
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    width = max(5, round(max(image.size) / 180))
    font_size = max(22, round(max(image.size) / 45))
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except OSError:
        font = ImageFont.load_default(size=font_size)
    # Draw all outlines first so later polygons cannot cover the numbered badges.
    for index, item in enumerate(items):
        if item["status"] == "ignored":
            continue
        points = [tuple(point) for point in item["points"]]
        color = COLORS[item["status"]]
        draw.line(points + [points[0]], fill="white", width=width * 2, joint="curve")
        draw.line(points + [points[0]], fill=color, width=width, joint="curve")
    padding = max(3, font_size // 6)
    for index, item in enumerate(items):
        if item["status"] == "ignored":
            continue
        text = str(index + 1)
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        badge_width = right - left + padding * 2
        badge_height = bottom - top + padding * 2
        x = max(0, min(min(p[0] for p in item["points"]), image.width - badge_width))
        y = max(
            0, min(min(p[1] for p in item["points"]) - badge_height, image.height - badge_height)
        )
        draw.rectangle(
            (x, y, x + badge_width, y + badge_height),
            fill=COLORS[item["status"]],
            outline="white",
            width=2,
        )
        draw.text((x + padding - left, y + padding - top), text, font=font, fill="white")
    canvas.save(path, quality=95, subsampling=0)


def write_report(output: Path, records: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    """Write an offline HTML gallery with escaped text and local images."""
    escape = html.escape
    cards = []
    names = {
        "correct": "正确",
        "wrong_text": "错字",
        "false_positive": "误检",
        "missed": "漏检",
    }
    for record in records:
        ident = record["id"]
        statuses = {
            item["status"]
            for item in record["labels"] + record["predictions"]
            if item["status"] != "ignored"
        }
        rows = []
        for kind, items in (("预测", record["predictions"]), ("标注", record["labels"])):
            for index, item in enumerate(items):
                status = item["status"]
                if status == "ignored":
                    continue
                if kind == "标注" and status != "missed":
                    continue
                predicted = item.get("text", "—（未检出）")
                target = item.get("target", "忽略区域" if status == "ignored" else "—（未匹配）")
                if kind == "标注":
                    target = item["transcription"]
                score = f'{item["score"]:.3f}' if "score" in item else "—"
                crop = ""
                if "image_shape" in record:
                    height, width = record["image_shape"][:2]
                    xs, ys = zip(*item["points"])
                    x, y = max(0, min(xs) - 4), max(0, min(ys) - 4)
                    w = max(1, min(width, max(xs) + 4) - x)
                    h = max(1, min(height, max(ys) + 4) - y)
                    crop = (
                        f'<svg class="crop" viewBox="{x} {y} {w} {h}" role="img" aria-label="文字区域">'
                        f'<image href="images/{ident}-original.jpg" width="{width}" height="{height}"/></svg>'
                    )
                verdict = {
                    "correct": "✓ 识别正确",
                    "wrong_text": "✗ 识别错误",
                    "false_positive": "未匹配到标注",
                    "missed": "✗ 漏检",
                    "ignored": "忽略（不计分）",
                }[status]
                rows.append(
                    f'<tr class="result-{status}"><td>{kind} {index + 1}</td><td>{crop}</td>'
                    f'<td class="ocr-text">{escape(predicted)}</td>'
                    f'<td class="ocr-text">{escape(target)}</td>'
                    f'<td style="color:{COLORS[status]};font-weight:700">{verdict}</td><td>{score}</td></tr>'
                )
        cards.append(
            f'<article data-status="{" ".join(sorted(statuses))}" data-name="{escape(ident, quote=True)}">'
            f'<h2>{escape(ident)} · {record["seconds"]["pipeline"] * 1000:.1f} ms</h2>'
            f'<div class="images"><figure><img loading="lazy" src="images/{ident}-original.jpg"><figcaption>原图</figcaption></figure>'
            f'<figure><img loading="lazy" src="images/{ident}-labels.jpg"><figcaption>真实标注</figcaption></figure>'
            f'<figure><img loading="lazy" src="images/{ident}-predictions.jpg"><figcaption>预测结果</figcaption></figure></div>'
            "<h3>逐条识别对照</h3><div class=results><table><tr><th>框编号</th><th>原图文字区域</th><th>预测文字</th><th>真实标注</th><th>判定</th><th>置信度</th></tr>"
            + "".join(rows)
            + ("" if rows else '<tr><td colspan="6">没有需要展示的可辨认文字结果</td></tr>')
            + "</table></div></article>"
        )
    options = "".join(f'<option value="{key}">{name}</option>' for key, name in names.items())
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>OCR 效果验证</title>
<style>body{font-family:system-ui;margin:24px;background:#f5f6f8;color:#182230}article,header{background:white;padding:18px;margin:18px 0;border-radius:10px}.images{display:grid;grid-template-columns:repeat(3,minmax(360px,1fr));gap:12px;overflow-x:auto}figure{margin:0;flex:1;min-width:0}img{width:100%;cursor:zoom-in}table{border-collapse:collapse;width:100%}td,th{text-align:left;border-bottom:1px solid #ddd;padding:8px}pre{white-space:pre-wrap}select,input{padding:8px}svg{max-width:100%}.results{overflow-x:auto}.crop{width:180px;height:85px;background:#edf0f4;display:block}.ocr-text{font-size:18px;min-width:160px;overflow-wrap:anywhere}.result-wrong_text,.result-false_positive{background:#fff5ef}.result-missed{background:#eff6ff}</style>
<header><h1>OCR 真实图片验证</h1><p>FILTER_NOTICE</p><p>抽样验证结果不能代表全量质量；项目匹配口径：多边形 IoU ≥ 0.5，按 IoU 降序一对一匹配，去空格后整条文字匹配。忽略区覆盖预测面积 > 0.5 时不计。未删除未知字或超长标注，不等同于训练内裁剪识别指标或官方排行榜。</p>
<p>绿：正确；橙：错字；红：误检；蓝：漏检。按人工标注过滤：illegibility=true 的不可辨认区域（准备数据时记为 ###）及其对应预测不画框、不展示、不计分。下方逐条展示预测文字与真实标注，✓ 表示整条匹配，✗ 标出错字或漏检。置信度不代表一定识别正确。点击整图可查看大图。</p><details><summary>指标与速度</summary><pre>SUMMARY</pre></details><p><a href="curves.html">查看训练曲线</a> · <a href="summary.json">完整统计</a> · <a href="predictions.jsonl">逐图结果</a></p></header>
<label>文件名 <input id="search"></label> <label>包含状态 <select id="status"><option value="">全部</option>OPTIONS</select></label>
CARDS
<script>function filter(){for(const a of document.querySelectorAll('article'))a.hidden=!(a.dataset.name.includes(document.querySelector('#search').value)&&a.dataset.status.includes(document.querySelector('#status').value));}for(const e of document.querySelectorAll('input,select'))e.addEventListener('input',filter);for(const im of document.images)im.onclick=()=>window.open(im.src,'_blank');</script></html>"""
    page = page.replace(
        "FILTER_NOTICE", "保留所有可辨认文字，不按 POI、电话号码、价格或促销内容筛选。"
    )
    page = page.replace("SUMMARY", escape(json.dumps(summary, ensure_ascii=False, indent=2)))
    page = page.replace("OPTIONS", options).replace("CARDS", "".join(cards))
    (output / "index.html").write_text(page, encoding="utf-8")
