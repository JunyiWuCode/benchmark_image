#!/usr/bin/env python
"""CVTG-2K (TextCrafter) scorer for FAR-RL outputs.

Reads a FAR-RL results.jsonl produced by scripts/benchmark_cvtg.py and computes
the CVTG-2K text metrics:

  * Word Accuracy  -- PaddleOCR word-level exact membership, GT-word weighted
  * NED            -- mean normalized (1 - Levenshtein/maxlen) over GT words
  * CLIPScore      -- official CLIP ViT-L/14, 2.5 * max(cos, 0), text =
                      "A photo depicts " + prompt

Logic mirrors NJU-PCALab/TextCrafter TextCrafter_Eval/unified_metrics_eval.py
(the PaddleOCR Word-Acc + NED and the official-CLIP CLIPScore paths only; the
VQAScore/Aesthetic metrics are intentionally omitted). The GT word list rides
in each row's metadata["text"] (quoted spans we pre-extracted at dataset build
time, equivalent to the official re.findall(r"'(.*?)'", prompt)).

PaddleOCR (paddle) and CLIP (torch) do not share one CUDA process. OCR uses the
project's PaddleOCR 3.3.3 environment and ``predict()`` result parser, while
CLIP uses a separate torch environment. Scoring is split into three stages,
writing partial JSON that a final merge combines:

    # stage 1 -- OCR (env paddleocr_gpu_official, paddle only, no torch).
    # 8 plain processes, one per GPU via CUDA_VISIBLE_DEVICES (NOT torchrun):
    for i in $(seq 0 7); do
      CUDA_VISIBLE_DEVICES=$i python scripts/cvtg_score.py --stage ocr \
        --results R.jsonl --output_dir OUT --num_shards 8 --shard_id $i &
    done; wait

    # stage 2 -- CLIP (env longtext_ocr, torch + openai-clip), torchrun:
    torchrun --nproc_per_node=8 scripts/cvtg_score.py --stage clip \
        --results R.jsonl --output_dir OUT

    # stage 3 -- merge (any env), single process:
    python scripts/cvtg_score.py --stage merge \
        --results R.jsonl --output_dir OUT
"""
import argparse
import difflib
import glob
import json
import logging
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.WARNING)
QUOTE_RE = re.compile(r"'(.*?)'")


def gt_words_from_row(row: dict) -> list:
    """Quoted GT spans -> lowercase words, matching the official extractor."""
    meta = row.get("metadata", row)
    spans = meta.get("text")
    if spans is None:
        spans = QUOTE_RE.findall(meta.get("prompt", ""))
    words = []
    for span in spans:
        words.extend(str(span).lower().split())
    return words


def get_ld(a: str, b: str) -> float:
    import Levenshtein
    edit_dist = Levenshtein.distance(a, b)
    return 1 - edit_dist / (max(len(a), len(b)) + 1e-5)


def clip_scores_for(clip_model, preprocess, device, image_paths, prompts):
    import torch
    import clip as clip_lib
    from PIL import Image

    images = []
    for p in image_paths:
        try:
            images.append(preprocess(Image.open(p).convert("RGB")).unsqueeze(0))
        except Exception:
            images.append(torch.zeros(1, 3, 224, 224))
    images = torch.cat(images, 0).to(device)
    texts = clip_lib.tokenize(["A photo depicts " + t for t in prompts], truncate=True).to(device)
    with torch.no_grad():
        img_f = clip_model.encode_image(images).cpu().numpy()
        txt_f = clip_model.encode_text(texts).cpu().numpy()
    img_f = img_f / np.sqrt(np.sum(img_f ** 2, axis=1, keepdims=True))
    txt_f = txt_f / np.sqrt(np.sum(txt_f ** 2, axis=1, keepdims=True))
    sims = np.sum(img_f * txt_f, axis=1)
    return (2.5 * np.clip(sims, 0, None)).tolist()


def load_rows(results_path: str) -> list:
    rows = []
    with open(results_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda r: r.get("image_path", ""))
    return rows


def stage_ocr(args) -> None:
    """PaddleOCR 3.x; sharded by --shard_id over --num_shards."""
    from importlib.metadata import version

    from score_ocr import build_ocr, recognized_text

    paddleocr_version = version("paddleocr")
    if paddleocr_version != "3.3.3":
        raise RuntimeError(f"CVTG requires PaddleOCR 3.3.3, found {paddleocr_version}.")
    rows = load_rows(args.results)
    my_rows = rows[args.shard_id::args.num_shards]
    paddle_args = argparse.Namespace(
        batch_size=args.ocr_batch,
        require_paddleocr_3_3_3=True,
    )
    ocr = build_ocr(paddle_args)
    paddle_info = {"version": paddleocr_version, "api": "paddleocr_3_predict"}

    match, total = 0, 0
    ned = []
    by_region = defaultdict(lambda: [0, 0])
    for start in range(0, len(my_rows), args.ocr_batch):
        chunk = my_rows[start:start + args.ocr_batch]
        paths = [str(Path(row["image_path"])) for row in chunk]
        raw_results = list(ocr.predict(input=paths)) if hasattr(ocr, "predict") else [
            ocr.ocr(path, cls=False) for path in paths
        ]
        for r, raw in zip(chunk, raw_results):
            gt = gt_words_from_row(r)
            if not gt:
                continue
            pred = str(recognized_text(raw)).lower().split() or [""]
            n_region = int(r.get("metadata", r).get("region_count", 0))
            for w in gt:
                total += 1
                by_region[n_region][1] += 1
                if w in pred:
                    match += 1
                    by_region[n_region][0] += 1
                best = difflib.get_close_matches(w, pred, n=1, cutoff=0)
                ned.append(get_ld(w, best[0]) if best else 0.0)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "num_rows": len(my_rows),
        "match": match,
        "total": total,
        "ned": ned,
        "by_region": {str(n): mt for n, mt in by_region.items()},
        "paddleocr": {"version": paddleocr_version, **paddle_info},
    }
    (out / f"ocr_shard_{args.shard_id}.json").write_text(
        json.dumps(payload), encoding="utf-8")
    print(f"[ocr shard {args.shard_id}/{args.num_shards}] rows={len(my_rows)} "
          f"match={match} total={total}")


def stage_clip(args) -> None:
    """torch + openai-clip; sharded by torchrun RANK/WORLD."""
    import torch
    import clip as clip_lib

    rank, world = int(args.shard_id), int(args.num_shards)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    rows = load_rows(args.results)
    my_rows = rows[rank::world]
    clip_model, preprocess = clip_lib.load("ViT-L/14", device=device, jit=False)
    clip_model.eval()

    clip_vals = []
    for i in range(0, len(my_rows), args.clip_batch):
        chunk = my_rows[i:i + args.clip_batch]
        paths = [r["image_path"] for r in chunk]
        prompts = [r.get("metadata", r).get("prompt", "") for r in chunk]
        clip_vals.extend(clip_scores_for(clip_model, preprocess, device, paths, prompts))

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"clip_shard_{rank}.json").write_text(
        json.dumps({"clip": clip_vals}), encoding="utf-8")
    print(f"[clip shard {rank}/{world}] images={len(my_rows)}")


def stage_merge(args) -> None:
    rows = load_rows(args.results)
    out = Path(args.output_dir)

    num_ocr_rows, match, total = 0, 0, 0
    ned, clip = [], []
    by_region = defaultdict(lambda: [0, 0])
    for p in sorted(glob.glob(str(out / "ocr_shard_*.json"))):
        g = json.loads(Path(p).read_text(encoding="utf-8"))
        num_ocr_rows += int(g["num_rows"])
        match += g["match"]
        total += g["total"]
        ned.extend(g["ned"])
        for n, (m, t) in g["by_region"].items():
            by_region[int(n)][0] += m
            by_region[int(n)][1] += t
    for p in sorted(glob.glob(str(out / "clip_shard_*.json"))):
        clip.extend(json.loads(Path(p).read_text(encoding="utf-8"))["clip"])
    if num_ocr_rows != len(rows):
        raise RuntimeError(f"Incomplete CVTG OCR coverage: {num_ocr_rows}/{len(rows)} images.")
    if len(clip) != len(rows):
        raise RuntimeError(f"Incomplete CVTG CLIP coverage: {len(clip)}/{len(rows)} images.")

    summary = {
        "num_images": len(rows),
        "word_accuracy": match / max(total, 1),
        "ned": float(np.mean(ned)) if ned else 0.0,
        "clip_score": float(np.mean(clip)) if clip else 0.0,
        "total_words": total,
        "match_words": match,
        "by_region": {str(n): {"word_accuracy": m / max(t, 1), "total_words": t}
                      for n, (m, t) in sorted(by_region.items())},
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["ocr", "clip", "merge"])
    ap.add_argument("--results", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--num_shards", type=int, default=8)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--clip_batch", type=int, default=32)
    ap.add_argument("--ocr_batch", type=int, default=32)
    ap.add_argument("--allow_partial", action="store_true")
    args = ap.parse_args()

    if args.stage == "ocr":
        stage_ocr(args)
    elif args.stage == "clip":
        stage_clip(args)
    else:
        stage_merge(args)


if __name__ == "__main__":
    main()
