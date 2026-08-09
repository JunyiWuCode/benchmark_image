from __future__ import annotations

from typing import Any


def _openai_messages(item: dict[str, Any]) -> list[dict[str, Any]]:
    user_text = str(item["user_text"])
    before, marker, after = user_text.partition("<image>")
    if not marker:
        raise ValueError("Q-Judger user prompt must contain exactly one <image> marker.")
    if "<image>" in after:
        raise ValueError("Q-Judger user prompt contains more than one <image> marker.")
    return [
        {"role": "system", "content": str(item["system_prompt"])},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": before},
                {"type": "image_pil", "image_pil": item["image"]},
                {"type": "text", "text": after},
            ],
        },
    ]


def _render_prompt(processor: Any, item: dict[str, Any]) -> str:
    messages = _openai_messages(item)
    template_messages = [
        messages[0],
        {
            "role": "user",
            "content": [
                {"type": "text", "text": messages[1]["content"][0]["text"]},
                {"type": "image"},
                {"type": "text", "text": messages[1]["content"][2]["text"]},
            ],
        },
    ]
    return processor.apply_chat_template(
        template_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


class VllmJudge:
    def __init__(
        self,
        model_path: str,
        *,
        max_batch_size: int = 24,
        max_new_tokens: int = 4096,
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.9,
    ) -> None:
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.engine = LLM(
            model=model_path,
            dtype="bfloat16",
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_seqs=max_batch_size,
            limit_mm_per_prompt={"image": 1},
            seed=42,
        )
        self.sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=0,
            top_k=1,
            top_p=1.0,
            repetition_penalty=1.05,
            seed=42,
        )

    def generate_batch(self, items: list[dict[str, Any]]) -> list[str]:
        prompts = [
            {
                "prompt": _render_prompt(self.processor, item),
                "multi_modal_data": {"image": item["image"]},
            }
            for item in items
        ]
        outputs = self.engine.generate(
            prompts,
            self.sampling_params,
            use_tqdm=False,
        )
        return [output.outputs[0].text for output in outputs]


class SglangJudge:
    def __init__(
        self,
        model_path: str,
        *,
        max_batch_size: int = 24,
        max_new_tokens: int = 4096,
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.9,
    ) -> None:
        from sglang import Engine
        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.engine = Engine(
            model_path=model_path,
            mem_fraction_static=gpu_memory_utilization,
            context_length=max_model_len,
        )
        self.max_batch_size = max_batch_size
        self.sampling_params = {
            "max_new_tokens": max_new_tokens,
            "temperature": 0,
            "top_k": 1,
            "top_p": 1.0,
            "repetition_penalty": 1.05,
            "seed": 42,
        }

    def generate_batch(self, items: list[dict[str, Any]]) -> list[str]:
        outputs = self.engine.generate(
            prompt=[_render_prompt(self.processor, item) for item in items],
            image_data=[item["image"] for item in items],
            sampling_params=self.sampling_params,
        )
        return [str(output["text"]) for output in outputs]


def build_judge(
    backend: str,
    *,
    model_path: str,
    max_batch_size: int,
    max_new_tokens: int,
    max_model_len: int,
    gpu_memory_utilization: float,
):
    normalized = backend.strip().lower()
    if normalized == "vllm":
        cls = VllmJudge
    elif normalized == "sglang":
        cls = SglangJudge
    else:
        raise ValueError(f"Unsupported accelerated Q-Judger backend: {backend!r}")
    return cls(
        model_path,
        max_batch_size=max_batch_size,
        max_new_tokens=max_new_tokens,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
    )
