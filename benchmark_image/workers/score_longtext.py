#!/usr/bin/env python
"""LongText-Bench scorer for FAR-RL outputs.

Reads a FAR-RL results.jsonl produced by scripts/benchmark_longtext.py, runs
Qwen2.5-VL-7B OCR on each generated image, and computes the official X-Omni
LongText-Bench word-level Text Score (sum(match)/sum(gt)). The GT text rides in
each row's metadata["text"], so no GT image is needed.

Run in an isolated env (transformers==4.52.0 + qwen_vl_utils) via torchrun:

    torchrun --nproc_per_node=8 scripts/longtext_score.py \
        --results /path/to/results.jsonl \
        --output_dir /path/to/eval_results

Adapted from X-Omni textbench/{evaluate_text_reward,summary_scores}.py.
"""
import argparse
import glob
import json
import os
import re
import warnings
from collections import Counter
from datetime import timedelta

warnings.filterwarnings("ignore")

import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

QWEN_MODEL_ID = os.environ.get("LONGTEXT_QWEN_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
QWEN_PROMPT = (
    "Recognize the text in the image, only reply with the text content, "
    "but avoid repeating previously mentioned content. "
    "If no text is recognized, please reply with 'No text recognized'."
)


def clean_and_remove_hallucinations(text):
    keywords_list = ["addCriterion", "No text recognized."]
    s = text
    for keyword in keywords_list:
        s = s.replace(keyword, "").replace(f"\n{keyword}", "").replace(f"{keyword}\n", "")
    return s


def preprocess_string(s, mode="en"):
    cleaned = re.sub(r"[^一-龥a-zA-Z0-9\sàâäéèêëîïôöùûüçÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ]", "", s)
    if mode == "en":
        return re.sub(r"\s+", " ", cleaned).strip().lower()
    pattern = re.compile(r"[一-龥a-zA-Z0-9àâäéèêëîïôöùûüçÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ]")
    return "".join(pattern.findall(s)).strip()


def counter2list(counter):
    return [item for item, count in counter.items() for _ in range(count)]


def calculate_char_match_ratio(text_gt, ocr_str, mode="en"):
    if mode == "en":
        gt_counter = Counter(text_gt.split())
        ocr_counter = Counter(ocr_str.split())
    else:
        gt_counter = Counter(text_gt)
        ocr_counter = Counter(ocr_str)
    match = counter2list(gt_counter & ocr_counter)
    unmatch = counter2list(gt_counter - ocr_counter)
    words_gt = text_gt.split() if mode == "en" else text_gt
    return match, words_gt, unmatch


def split_list(x, n):
    k, m = divmod(len(x), n)
    return [x[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def load_items(results_path):
    """Read FAR-RL results.jsonl into [{image, prompt, text, category, prompt_id}]."""
    items = []
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            image = row.get("image_path", "")
            if not image or not os.path.exists(image):
                continue
            meta = row.get("metadata", {})
            gt_text = meta.get("text", [])
            if isinstance(gt_text, str):
                gt_text = [gt_text]
            items.append(
                {
                    "image": image,
                    "prompt": row.get("prompt", ""),
                    "text": gt_text,
                    "category": meta.get("category", ""),
                    "prompt_id": meta.get("prompt_id", row.get("image_id", "")),
                }
            )
    return items


class ImageEvaluator:
    def __init__(self, device):
        self.device = torch.device("cuda", int(device))
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            QWEN_MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map=None,
        ).to(self.device).eval()
        self._assert_model_on_device()
        self.processor = AutoProcessor.from_pretrained(QWEN_MODEL_ID)

    def _assert_model_on_device(self):
        bad = [
            (name, tuple(param.shape), str(param.device))
            for name, param in self.model.named_parameters()
            if param.device != self.device
        ]
        bad.extend(
            (name, tuple(buf.shape), str(buf.device))
            for name, buf in self.model.named_buffers()
            if buf.device != self.device
        )
        if bad:
            sample = ", ".join(f"{name}:{dev}" for name, _, dev in bad[:8])
            raise RuntimeError(
                f"Qwen2.5-VL is not fully on {self.device}; "
                f"{len(bad)} tensors are elsewhere: {sample}"
            )

    def qwen_ocr(self, image):
        message = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": QWEN_PROMPT},
        ]}]
        texts = self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(message)
        inputs = self.processor(text=texts, images=image_inputs, padding=True, return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        outputs = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return clean_and_remove_hallucinations(outputs[0])

    def evaluate(self, chunk):
        out = []
        with torch.no_grad():
            for data in tqdm(chunk):
                out.append({
                    "image": data["image"],
                    "prompt": data["prompt"],
                    "ocr_gt": data["text"],
                    "category": data.get("category", ""),
                    "prompt_id": data.get("prompt_id", ""),
                    "ocr_results": self.qwen_ocr(data["image"]),
                })
        return out


def score_and_summarize(results, mode, output_dir):
    per_cat = {}
    for r in results:
        ocr_results = preprocess_string(r["ocr_results"], mode)
        ocr_gt = preprocess_string(" ".join(r["ocr_gt"]), mode)
        match, gt, _ = calculate_char_match_ratio(ocr_gt, ocr_results, mode)
        r["ocr_results"] = ocr_results
        r["ocr_gt"] = ocr_gt
        r["match_word_count"] = len(match)
        r["gt_word_count"] = len(gt)
        r["text_accuray"] = len(match) / len(gt) if len(gt) else 0.0
        cat = r.get("category", "") or "unknown"
        bucket = per_cat.setdefault(cat, [0, 0])
        bucket[0] += len(match)
        bucket[1] += len(gt)

    res_path = os.path.join(output_dir, "results.jsonl")
    with open(res_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total_match = sum(r["match_word_count"] for r in results)
    total_gt = sum(r["gt_word_count"] for r in results)
    text_score = total_match / total_gt if total_gt else 0.0

    lines = [f"Text Score: {text_score:.4f}", f"num_images: {len(results)}", ""]
    lines.append("Per-category Text Score:")
    for cat in sorted(per_cat):
        m, g = per_cat[cat]
        lines.append(f"  {cat}: {m / g:.4f}" if g else f"  {cat}: n/a")
    report = "\n".join(lines) + "\n"
    print(report)
    with open(os.path.join(output_dir, "scores.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "text_score": text_score,
                "num_images": len(results),
                "total_match": total_match,
                "total_gt": total_gt,
                "per_category": {c: (per_cat[c][0] / per_cat[c][1] if per_cat[c][1] else None) for c in per_cat},
            },
            f,
            indent=2,
            ensure_ascii=False,
        )


def score_shard(args):
    torch.set_grad_enabled(False)
    rank = int(args.rank)
    world_size = int(args.world_size)
    torch.cuda.set_device(0)
    device = 0
    torch.manual_seed(args.global_seed * world_size + rank)
    os.makedirs(args.output_dir, exist_ok=True)
    data = load_items(args.results)
    with open(args.results, "r", encoding="utf-8") as f:
        expected_count = sum(1 for line in f if line.strip())
    if len(data) != expected_count:
        raise RuntimeError(
            f"LongText manifest has {expected_count} rows but only {len(data)} readable images."
        )
    chunk = data[rank::world_size]

    evaluator = ImageEvaluator(device)
    print(f"=== rank {rank}: OCR {len(chunk)} images ===")
    chunk_results = evaluator.evaluate(chunk)
    chunk_path = os.path.join(args.output_dir, f"results_chunk{rank}.jsonl")
    with open(chunk_path, "w", encoding="utf-8") as f:
        for r in chunk_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")



def merge(args):
    merged = []
    for rank in range(args.world_size):
        path = os.path.join(args.output_dir, f"results_chunk{rank}.jsonl")
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        with open(path, "r", encoding="utf-8") as f:
            merged.extend(json.loads(line) for line in f if line.strip())
    if not args.allow_partial and len(merged) != 640:
        raise RuntimeError(f"Incomplete LongText EN benchmark: {len(merged)}/640.")
    score_and_summarize(merged, args.mode, args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="FAR-RL results.jsonl from benchmark_longtext.py")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mode", choices=["en", "zh"], default="en")
    parser.add_argument("--global_seed", type=int, default=42)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, required=True)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--allow_partial", action="store_true")
    args = parser.parse_args()
    merge(args) if args.merge else score_shard(args)
