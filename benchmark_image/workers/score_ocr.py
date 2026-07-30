#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from importlib.metadata import version
from pathlib import Path

import numpy as np
from PIL import Image


QUOTE_RE = re.compile(r'"([^"]+)"')


def read_rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def normalize(text: str) -> str:
    return str(text or "").replace(" ", "").lower()


def levenshtein(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, 1):
        current = [i]
        for j, right_char in enumerate(right, 1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def flowopd_score(prediction: str, prompt: str) -> float:
    matches = QUOTE_RE.findall(prompt)
    if not matches:
        return 0.0
    target = normalize(matches[0])
    prediction = normalize(prediction)
    if target in prediction:
        return 1.0
    distance = min(levenshtein(prediction, target), len(target))
    return max(0.0, 1.0 - float(distance) / float(max(len(target), 1)))


def patch_paddlex_headless_cv() -> None:
    try:
        from paddlex.utils import deps
    except Exception:
        return
    original = deps.is_dep_available

    def available(dep):
        if dep == "opencv-contrib-python":
            return (
                original(dep)
                or original("opencv-contrib-python-headless")
                or original("opencv-python-headless")
            )
        return original(dep)

    deps.is_dep_available = available


def build_ocr(args):
    if args.require_paddleocr_3_3_3:
        installed = version("paddleocr")
        if installed != "3.3.3":
            raise RuntimeError(f"Expected PaddleOCR 3.3.3, found {installed}.")
    os.environ.setdefault("DISABLE_MODEL_SOURCE_CHECK", "True")
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    patch_paddlex_headless_cv()
    from paddleocr import PaddleOCR

    kwargs = {
        "lang": "en",
        "device": "gpu:0",
        "text_recognition_batch_size": int(args.batch_size),
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
    }
    try:
        return PaddleOCR(engine="paddle", **kwargs)
    except (TypeError, ValueError) as exc:
        if "engine" not in str(exc).lower():
            raise
        return PaddleOCR(**kwargs)


def plain_result(raw):
    if hasattr(raw, "json"):
        payload = raw.json() if callable(raw.json) else raw.json
        if isinstance(payload, dict):
            return payload.get("res", payload)
    if isinstance(raw, dict):
        return raw.get("res", raw)
    return raw


def recognized_text(raw) -> str:
    payload = plain_result(raw)
    if isinstance(payload, dict):
        texts = payload.get("rec_texts") or payload.get("texts") or []
        return "".join(str(text) for text in texts if str(text).strip())
    texts = []
    for item in payload or []:
        detections = item if isinstance(item, list) else [item]
        for detection in detections:
            if not isinstance(detection, (list, tuple)) or len(detection) < 2:
                continue
            rec = detection[1]
            text = rec[0] if isinstance(rec, (list, tuple)) else rec
            if str(text).strip():
                texts.append(str(text))
    return "".join(texts)


def score(args) -> None:
    rows = read_rows(Path(args.results))[args.rank :: args.world_size]
    ocr = build_ocr(args)
    scored = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        paths = [str(Path(row["image_path"])) for row in batch]
        if hasattr(ocr, "predict"):
            raw_results = list(ocr.predict(input=paths))
        else:
            raw_results = [ocr.ocr(path, cls=False) for path in paths]
        for row, raw in zip(batch, raw_results):
            prediction = recognized_text(raw)
            scored.append(
                {
                    "artifact_id": row["artifact_id"],
                    "score": flowopd_score(prediction, row["prompt"]),
                    "recognized_text": prediction,
                }
            )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"rank_{args.rank:05d}.jsonl"
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in scored)
        + ("\n" if scored else ""),
        encoding="utf-8",
    )


def merge(args) -> None:
    scores = {}
    for rank in range(args.world_size):
        path = Path(args.output_dir) / f"rank_{rank:05d}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        for row in read_rows(path):
            scores.setdefault(row["artifact_id"], float(row["score"]))
    if not args.allow_partial and len(scores) != 1018:
        raise RuntimeError(f"Incomplete Flow-OPD OCR benchmark: {len(scores)} images, expected 1018.")
    summary = {
        "metrics": {
            "flowopd_ocr_acc": sum(scores.values()) / len(scores),
        },
        "num_images": len(scores),
        "complete": len(scores) == 1018,
    }
    path = Path(args.output_dir) / "summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--results")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--require-paddleocr-3-3-3", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    if args.merge:
        merge(args)
    else:
        if not args.results:
            parser.error("--results is required for scoring.")
        score(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

