from benchmark_image.workers.score_ocr import recognized_text


def test_recognized_text_can_preserve_detected_region_boundaries():
    raw = {"res": {"rec_texts": ["HELLO", "WORLD"]}}

    assert recognized_text(raw) == "HELLOWORLD"
    assert recognized_text(raw, separator=" ") == "HELLO WORLD"


def test_recognized_text_separator_supports_legacy_results():
    raw = [[
        [[0, 0, 1, 1], ("FIRST", 0.99)],
        [[1, 1, 2, 2], ("SECOND", 0.98)],
    ]]

    assert recognized_text(raw, separator=" ") == "FIRST SECOND"
