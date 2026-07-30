#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


STYLES = ("anime", "concept-art", "paintings", "photo")


def read_rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def score(args) -> None:
    import torch
    from PIL import Image
    from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer

    pretrained = args.open_clip_pretrained
    if args.open_clip_pretrained_path and Path(args.open_clip_pretrained_path).expanduser().is_file():
        pretrained = str(Path(args.open_clip_pretrained_path).expanduser())
    model, _, preprocess = create_model_and_transforms(
        "ViT-H-14",
        pretrained,
        precision="amp",
        device="cpu",
        jit=False,
        force_quick_gelu=False,
        force_custom_text=False,
        force_patch_dropout=False,
        force_image_size=None,
        pretrained_image=False,
        image_mean=None,
        image_std=None,
        light_augmentation=True,
        aug_cfg={},
        output_dict=True,
        with_score_predictor=False,
        with_region_predictor=False,
    )
    checkpoint = torch.load(
        os.path.expanduser(args.checkpoint_path),
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.cuda().eval().requires_grad_(False)
    tokenizer = get_tokenizer("ViT-H-14")

    rows = read_rows(Path(args.results))[args.rank :: args.world_size]
    scored = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        images = torch.stack(
            [preprocess(Image.open(row["image_path"]).convert("RGB")) for row in batch]
        ).cuda()
        texts = tokenizer([row["prompt"] for row in batch]).cuda()
        with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True):
            output = model(images, texts)
            values = torch.diagonal(output["image_features"] @ output["text_features"].T)
        for row, value in zip(batch, values.detach().float().cpu().tolist()):
            scored.append(
                {
                    "artifact_id": row["artifact_id"],
                    "style": row["metadata"]["style"],
                    "score": float(value),
                }
            )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"rank_{args.rank:05d}.jsonl"
    path.write_text(
        "\n".join(json.dumps(row) for row in scored) + ("\n" if scored else ""),
        encoding="utf-8",
    )


def merge(args) -> None:
    buckets = {style: [] for style in STYLES}
    seen = set()
    for rank in range(args.world_size):
        path = Path(args.output_dir) / f"rank_{rank:05d}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        for row in read_rows(path):
            artifact_id = row["artifact_id"]
            if artifact_id in seen:
                continue
            seen.add(artifact_id)
            buckets[row["style"]].append(float(row["score"]) * 100.0)
    counts = {style: len(values) for style, values in buckets.items()}
    if not args.allow_partial and any(counts[style] != 800 for style in STYLES):
        raise RuntimeError(f"Incomplete official HPSv2 benchmark: {counts}")
    summary = {
        style: sum(values) / len(values)
        for style, values in buckets.items()
        if values
    }
    all_values = [value for values in buckets.values() for value in values]
    summary.update(
        {
            "average": sum(all_values) / len(all_values),
            "num_prompts": len(all_values),
            "complete": len(all_values) == 3200,
            "hps_version": "v2.1",
            "counts": counts,
        }
    )
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
    parser.add_argument("--checkpoint-path", default="")
    parser.add_argument("--open-clip-pretrained", default="laion2B-s32B-b79K")
    parser.add_argument("--open-clip-pretrained-path", default="")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    if args.merge:
        merge(args)
    else:
        if not args.results or not args.checkpoint_path:
            parser.error("--results and --checkpoint-path are required for scoring.")
        score(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

