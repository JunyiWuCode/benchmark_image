from __future__ import annotations

import argparse
import json

from benchmark_image.dataset import ImageBenchmarkDataset
from benchmark_image.protocols import BENCHMARKS, expected_image_count


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect benchmark-image protocols.")
    parser.add_argument(
        "--benchmarks",
        default="aesthetic_quality,hpsv2_official,geneval,geneval2,ocr,cvtg,longtext_en",
    )
    args = parser.parse_args()
    names = [name.strip() for name in args.benchmarks.split(",") if name.strip()]
    dataset = ImageBenchmarkDataset(names)
    payload = {
        "benchmarks": names,
        "protocols": {
            name: {
                "prompt_count": BENCHMARKS[name].prompt_count,
                "samples_per_prompt": BENCHMARKS[name].samples_per_prompt,
                "image_count": BENCHMARKS[name].image_count,
                "primary_metric": BENCHMARKS[name].primary_metric,
            }
            for name in names
        },
        "dataset_images": len(dataset),
        "expected_images": expected_image_count(names),
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
