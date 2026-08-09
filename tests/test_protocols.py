from benchmark_image.dataset import ImageBenchmarkDataset
from benchmark_image.protocols import BENCHMARKS, expected_image_count


def test_full_suite_counts():
    names = (
        "aesthetic_quality",
        "hpsv2_official",
        "geneval",
        "geneval2",
        "ocr",
        "cvtg",
        "longtext_en",
    )
    dataset = ImageBenchmarkDataset(names)
    assert len(dataset) == 10894
    assert expected_image_count(names) == 10894
    assert BENCHMARKS["geneval"].image_count == 2212


def test_smoke_limit_is_prompt_level():
    dataset = ImageBenchmarkDataset(
        ("aesthetic_quality", "hpsv2_official", "geneval", "geneval2", "ocr", "cvtg", "longtext_en"),
        smoke_max_prompts_per_benchmark=2,
    )
    assert len(dataset) == 2 + 2 + 2 * 4 + 2 + 2 + 2 + 2 * 4


def test_qwen_image_bench_uses_official_chinese_prompt_protocol():
    dataset = ImageBenchmarkDataset(("qwen_image_bench",))
    assert len(dataset) == 1000
    assert expected_image_count(("qwen_image_bench",)) == 1000
    first = dataset[0]
    assert first["artifact_id"] == "qwen_image_bench:1:0"
    assert first["image_id"] == "000001"
    assert "机械手表" in first["prompt"]
    assert '"language": "cn"' in first["metadata_json"]
    assert '"dims_en"' in first["metadata_json"]
