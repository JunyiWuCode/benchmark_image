import pytest

from benchmark_image.q_judger_backends import _openai_messages, _render_prompt


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
