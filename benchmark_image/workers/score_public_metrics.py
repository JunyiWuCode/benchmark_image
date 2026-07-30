#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


class AestheticMLP:
    @staticmethod
    def build():
        import torch.nn as nn

        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(768, 1024),
                    nn.Dropout(0.2),
                    nn.Linear(1024, 128),
                    nn.Dropout(0.2),
                    nn.Linear(128, 64),
                    nn.Dropout(0.1),
                    nn.Linear(64, 16),
                    nn.Linear(16, 1),
                )

            def forward(self, embeddings):
                return self.layers(embeddings)

        return MLP()


def score(args):
    import torch
    from PIL import Image
    from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
    from transformers import CLIPModel, CLIPProcessor

    hps_pretrained = args.hps_open_clip_pretrained_path or args.hps_open_clip_pretrained
    hps, _, hps_preprocess = create_model_and_transforms(
        "ViT-H-14", hps_pretrained, precision="amp", device="cpu", jit=False,
        force_quick_gelu=False, force_custom_text=False, force_patch_dropout=False,
        force_image_size=None, pretrained_image=False, image_mean=None, image_std=None,
        light_augmentation=True, aug_cfg={}, output_dict=True,
        with_score_predictor=False, with_region_predictor=False,
    )
    hps.load_state_dict(torch.load(args.hps_checkpoint_path, map_location="cpu", weights_only=False)["state_dict"])
    hps.cuda().eval().requires_grad_(False)
    hps_tokenizer = get_tokenizer("ViT-H-14")

    clip = CLIPModel.from_pretrained(args.clip_model).cuda().eval().requires_grad_(False)
    processor = CLIPProcessor.from_pretrained(args.clip_model)
    aesthetic = AestheticMLP.build().cuda().eval().requires_grad_(False)
    aesthetic.load_state_dict(torch.load(args.aesthetic_checkpoint_path, map_location="cpu", weights_only=True))

    output = []
    rows = read_rows(args.results)[args.rank :: args.world_size]
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        pil = [Image.open(row["image_path"]).convert("RGB") for row in batch]
        prompts = [row["prompt"] for row in batch]
        hps_images = torch.stack([hps_preprocess(image) for image in pil]).cuda()
        hps_text = hps_tokenizer(prompts).cuda()
        clip_inputs = processor(
            text=prompts, images=pil, padding="max_length", truncation=True, return_tensors="pt"
        )
        clip_inputs = {key: value.cuda() for key, value in clip_inputs.items()}
        with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True):
            hps_out = hps(hps_images, hps_text)
            hps_values = torch.diagonal(hps_out["image_features"] @ hps_out["text_features"].T)
            clip_out = clip(**clip_inputs)
            clip_values = clip_out.logits_per_image.diagonal() / 30.0
            image_features = clip.get_image_features(pixel_values=clip_inputs["pixel_values"])
            if hasattr(image_features, "pooler_output"):
                image_features = image_features.pooler_output
            image_features = image_features / torch.linalg.vector_norm(image_features, dim=-1, keepdim=True)
            aesthetic_values = aesthetic(image_features.float()).squeeze(1)
        for row, hps_value, aesthetic_value, clip_value in zip(
            batch,
            hps_values.float().cpu().tolist(),
            aesthetic_values.float().cpu().tolist(),
            clip_values.float().cpu().tolist(),
        ):
            output.append(
                {
                    "artifact_id": row["artifact_id"],
                    "hpsv2": hps_value,
                    "aesthetic": aesthetic_value,
                    "clipscore": clip_value,
                }
            )
    path = Path(args.output_dir) / f"aesthetic_quality_rank_{args.rank:05d}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in output) + ("\n" if output else ""), encoding="utf-8")


def merge(args):
    rows = {}
    for rank in range(args.world_size):
        for row in read_rows(Path(args.output_dir) / f"aesthetic_quality_rank_{rank:05d}.jsonl"):
            rows.setdefault(row["artifact_id"], row)
    if not args.allow_partial and len(rows) != 1024:
        raise RuntimeError(f"Incomplete aesthetic quality benchmark: {len(rows)}/1024.")
    summary = {
        name: sum(float(row[name]) for row in rows.values()) / len(rows)
        for name in ("hpsv2", "aesthetic", "clipscore")
    }
    summary.update({"num_inputs": len(rows), "complete": len(rows) == 1024})
    path = Path(args.output_dir) / "summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--results")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hps-checkpoint-path", default="")
    parser.add_argument("--hps-open-clip-pretrained", default="laion2B-s32B-b79K")
    parser.add_argument("--hps-open-clip-pretrained-path", default="")
    parser.add_argument("--aesthetic-checkpoint-path", default="")
    parser.add_argument("--clip-model", default="openai/clip-vit-large-patch14")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    merge(args) if args.merge else score(args)


if __name__ == "__main__":
    main()
