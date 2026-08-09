from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path

from benchmark_image.protocols import BENCHMARKS, normalize_benchmarks


ASSET_ROOT = files("benchmark_image").joinpath("assets")
HPS_STYLES = ("anime", "concept-art", "paintings", "photo")


def _read_json(path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _hpsv2_official_records() -> list[dict]:
    records = []
    prompt_index = 0
    for style in HPS_STYLES:
        prompts = _read_json(ASSET_ROOT.joinpath("hpsv2", f"{style}.json"))
        if len(prompts) != 800:
            raise ValueError(f"HPSv2 style {style!r} must contain 800 prompts, got {len(prompts)}.")
        for style_index, prompt in enumerate(prompts):
            records.append(
                {
                    "benchmark": "hpsv2_official",
                    "prompt_index": prompt_index,
                    "sample_index": 0,
                    "image_id": f"{style}_{style_index:05d}",
                    "artifact_id": f"hpsv2_official:{style}:{style_index}",
                    "prompt": str(prompt),
                    "metadata": {"style": style, "style_index": style_index},
                }
            )
            prompt_index += 1
    return records


def _geneval_records() -> list[dict]:
    metadata_rows = _read_jsonl(ASSET_ROOT.joinpath("geneval", "evaluation_metadata.jsonl"))
    if len(metadata_rows) != 553:
        raise ValueError(f"GenEval must contain 553 prompts, got {len(metadata_rows)}.")
    records = []
    for prompt_index, metadata in enumerate(metadata_rows):
        for sample_index in range(4):
            records.append(
                {
                    "benchmark": "geneval",
                    "prompt_index": prompt_index,
                    "sample_index": sample_index,
                    "image_id": f"{prompt_index:05d}",
                    "artifact_id": f"geneval:{prompt_index}:{sample_index}",
                    "prompt": str(metadata["prompt"]),
                    "metadata": metadata,
                }
            )
    return records


def _ocr_records() -> list[dict]:
    prompts = [
        line.strip()
        for line in ASSET_ROOT.joinpath("ocr", "test.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(prompts) != 1018:
        raise ValueError(f"Flow-OPD OCR must contain 1018 prompts, got {len(prompts)}.")
    return [
        {
            "benchmark": "ocr",
            "prompt_index": index,
            "sample_index": 0,
            "image_id": f"{index:05d}",
            "artifact_id": f"ocr:{index}:0",
            "prompt": prompt,
            "metadata": {},
        }
        for index, prompt in enumerate(prompts)
    ]


def _aesthetic_quality_records() -> list[dict]:
    prompts = [
        line.strip()
        for line in ASSET_ROOT.joinpath("public_metrics", "test.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(prompts) != 1024:
        raise ValueError(f"Aesthetic quality must contain 1024 prompts, got {len(prompts)}.")
    return [
        {
            "benchmark": "aesthetic_quality",
            "prompt_index": index,
            "sample_index": 0,
            "image_id": f"{index:05d}",
            "artifact_id": f"aesthetic_quality:{index}:0",
            "prompt": prompt,
            "metadata": {},
        }
        for index, prompt in enumerate(prompts)
    ]


def _jsonl_image_records(benchmark: str, asset_parts: tuple[str, ...], expected: int, samples: int) -> list[dict]:
    metadata_rows = _read_jsonl(ASSET_ROOT.joinpath(*asset_parts))
    if len(metadata_rows) != expected:
        raise ValueError(f"{benchmark} must contain {expected} prompts, got {len(metadata_rows)}.")
    records = []
    for prompt_index, source_metadata in enumerate(metadata_rows):
        metadata = dict(source_metadata)
        prompt = str(metadata["prompt"])
        metadata["original_prompt_id"] = metadata.get("prompt_id", prompt_index)
        metadata["prompt_id"] = prompt_index
        for sample_index in range(samples):
            records.append(
                {
                    "benchmark": benchmark,
                    "prompt_index": prompt_index,
                    "sample_index": sample_index,
                    "image_id": f"{prompt_index:05d}",
                    "artifact_id": f"{benchmark}:{prompt_index}:{sample_index}",
                    "prompt": prompt,
                    "metadata": metadata,
                }
            )
    return records


def _cvtg_records() -> list[dict]:
    return _jsonl_image_records("cvtg", ("cvtg", "cvtg_prompts.jsonl"), 2000, 1)


def _longtext_records() -> list[dict]:
    return _jsonl_image_records("longtext_en", ("longtext", "text_prompts.jsonl"), 160, 4)


def _geneval2_records() -> list[dict]:
    return _jsonl_image_records("geneval2", ("geneval2", "geneval2_data.jsonl"), 800, 1)


def _qwen_image_bench_records() -> list[dict]:
    source_rows = _read_jsonl(
        ASSET_ROOT.joinpath("qwen_image_bench", "prompts_cn.jsonl")
    )
    if len(source_rows) != 1000:
        raise ValueError(
            f"Qwen-Image-Bench must contain 1000 prompts, got {len(source_rows)}."
        )
    records = []
    for prompt_index, source in enumerate(source_rows):
        benchmark_id = int(source["ID"])
        if benchmark_id != prompt_index + 1:
            raise ValueError(
                "Qwen-Image-Bench IDs must be contiguous and one-indexed: "
                f"row={prompt_index}, ID={benchmark_id}."
            )
        records.append(
            {
                "benchmark": "qwen_image_bench",
                "prompt_index": prompt_index,
                "sample_index": 0,
                "image_id": f"{benchmark_id:06d}",
                "artifact_id": f"qwen_image_bench:{benchmark_id}:0",
                "prompt": str(source["prompt_cn"]),
                "metadata": {
                    "ID": benchmark_id,
                    "prompt_cn": str(source["prompt_cn"]),
                    "prompt_en": str(source["prompt_en"]),
                    "dims_cn": str(source["dims_cn"]),
                    "dims_en": str(source["dims_en"]),
                    "language": "cn",
                },
            }
        )
    return records


LOADERS = {
    "aesthetic_quality": _aesthetic_quality_records,
    "hpsv2_official": _hpsv2_official_records,
    "geneval": _geneval_records,
    "ocr": _ocr_records,
    "cvtg": _cvtg_records,
    "longtext_en": _longtext_records,
    "geneval2": _geneval2_records,
    "qwen_image_bench": _qwen_image_bench_records,
}


class ImageBenchmarkDataset:
    """Image-level view of the selected full benchmark protocols.

    GenEval is expanded to four records per prompt so every dataloader item maps
    to exactly one deterministic image and one output artifact.
    """

    def __init__(
        self,
        benchmarks=(
            "aesthetic_quality",
            "hpsv2_official",
            "geneval",
            "geneval2",
            "ocr",
            "cvtg",
            "longtext_en",
        ),
        smoke_max_prompts_per_benchmark=None,
    ):
        self.benchmarks = normalize_benchmarks(benchmarks)
        self.records = []
        for benchmark in self.benchmarks:
            records = LOADERS[benchmark]()
            if smoke_max_prompts_per_benchmark is not None:
                limit = int(smoke_max_prompts_per_benchmark)
                if limit <= 0:
                    raise ValueError("smoke_max_prompts_per_benchmark must be positive.")
                records = [
                    record
                    for record in records
                    if int(record["prompt_index"]) < limit
                ]
            self.records.extend(records)

        for index, record in enumerate(self.records):
            record["index"] = index
            record["metadata_json"] = json.dumps(record.pop("metadata"), ensure_ascii=False)

        if smoke_max_prompts_per_benchmark is None:
            expected = sum(BENCHMARKS[name].image_count for name in self.benchmarks)
            if len(self.records) != expected:
                raise RuntimeError(f"Benchmark dataset has {len(self.records)} images, expected {expected}.")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return dict(self.records[index])
