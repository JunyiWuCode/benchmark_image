import json
import os
import sys
from pathlib import Path

import pytest

from benchmark_image.q_judger_backends import (
    _openai_messages,
    _prepend_runtime_bin,
    _render_prompt,
    _vllm_model_compat_path,
)


def test_prepend_runtime_bin(monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin")
    _prepend_runtime_bin()
    assert os.environ["PATH"].split(os.pathsep) == [
        str(Path(sys.executable).resolve().parent),
        "/usr/bin",
    ]


def test_openai_messages_replace_exactly_one_image_marker():
    image = object()
    messages = _openai_messages(
        {
            "system_prompt": "system",
            "user_text": "before<image>after",
            "image": image,
        }
    )
    assert messages[1]["content"] == [
        {"type": "text", "text": "before"},
        {"type": "image_pil", "image_pil": image},
        {"type": "text", "text": "after"},
    ]


@pytest.mark.parametrize("text", ["missing", "a<image>b<image>c"])
def test_openai_messages_reject_invalid_image_markers(text):
    with pytest.raises(ValueError):
        _openai_messages(
            {"system_prompt": "system", "user_text": text, "image": object()}
        )


def test_render_prompt_enables_thinking_and_uses_one_image():
    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            self.messages = messages
            self.kwargs = kwargs
            return "rendered"

    processor = Processor()
    result = _render_prompt(
        processor,
        {
            "system_prompt": "system",
            "user_text": "before<image>after",
            "image": object(),
        },
    )
    assert result == "rendered"
    assert processor.kwargs == {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": True,
    }
    assert processor.messages[1]["content"][1] == {"type": "image"}


def test_vllm_model_compat_rewrites_metadata_without_copying_weights(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "TokenizersBackend", "foo": 1}),
        encoding="utf-8",
    )
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model / "weights.safetensors").write_text("weights", encoding="utf-8")
    (model / "processor_config.json").write_text(
        json.dumps(
            {
                "video_processor": {
                    "video_processor_type": "Qwen3VLVideoProcessor",
                    "fps": 2,
                }
            }
        ),
        encoding="utf-8",
    )

    compat = Path(_vllm_model_compat_path(str(model)))
    config = json.loads((compat / "tokenizer_config.json").read_text())
    assert config == {"tokenizer_class": "Qwen2TokenizerFast", "foo": 1}
    assert (compat / "tokenizer.json").is_symlink()
    assert (compat / "weights.safetensors").is_symlink()
    assert json.loads((compat / "video_preprocessor_config.json").read_text()) == {
        "video_processor_type": "Qwen3VLVideoProcessor",
        "fps": 2,
    }
    assert json.loads((model / "tokenizer_config.json").read_text()) == {
        "tokenizer_class": "TokenizersBackend",
        "foo": 1,
    }
