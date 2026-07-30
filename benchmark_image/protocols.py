from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkProtocol:
    name: str
    prompt_count: int
    samples_per_prompt: int
    primary_metric: str

    @property
    def image_count(self) -> int:
        return self.prompt_count * self.samples_per_prompt


BENCHMARKS = {
    "aesthetic_quality": BenchmarkProtocol(
        name="aesthetic_quality",
        prompt_count=1024,
        samples_per_prompt=1,
        primary_metric="hpsv2",
    ),
    "hpsv2_official": BenchmarkProtocol(
        name="hpsv2_official",
        prompt_count=3200,
        samples_per_prompt=1,
        primary_metric="hpsv2_average",
    ),
    "geneval": BenchmarkProtocol(
        name="geneval",
        prompt_count=553,
        samples_per_prompt=4,
        primary_metric="geneval_overall",
    ),
    "ocr": BenchmarkProtocol(
        name="ocr",
        prompt_count=1018,
        samples_per_prompt=1,
        primary_metric="ocr_flowopd_acc",
    ),
    "cvtg": BenchmarkProtocol(
        name="cvtg",
        prompt_count=2000,
        samples_per_prompt=1,
        primary_metric="cvtg_word_accuracy",
    ),
    "longtext_en": BenchmarkProtocol(
        name="longtext_en",
        prompt_count=160,
        samples_per_prompt=4,
        primary_metric="longtext_en_text_score",
    ),
    "geneval2": BenchmarkProtocol(
        name="geneval2",
        prompt_count=800,
        samples_per_prompt=1,
        primary_metric="geneval2_soft_tifa_gm",
    ),
}


def normalize_benchmarks(benchmarks) -> tuple[str, ...]:
    names = tuple(str(name).strip().lower() for name in benchmarks if str(name).strip())
    if not names:
        raise ValueError("At least one image benchmark must be selected.")
    unknown = sorted(set(names) - set(BENCHMARKS))
    if unknown:
        raise ValueError(f"Unknown image benchmarks: {unknown}; supported={sorted(BENCHMARKS)}")
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate image benchmarks are not allowed: {names}")
    return names


def expected_image_count(benchmarks) -> int:
    return sum(BENCHMARKS[name].image_count for name in normalize_benchmarks(benchmarks))
