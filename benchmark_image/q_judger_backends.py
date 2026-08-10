from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
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


def _load_qwen_tokenizer(model_path: str):
    from transformers import Qwen2TokenizerFast

    return Qwen2TokenizerFast.from_pretrained(model_path)


def _prepend_runtime_bin() -> None:
    runtime_bin = str(Path(sys.executable).resolve().parent)
    current_path = os.environ.get("PATH", "")
    os.environ["PATH"] = os.pathsep.join(
        part for part in (runtime_bin, current_path) if part
    )


def _vllm_model_compat_path(model_path: str) -> str:
    source = Path(model_path).resolve()
    digest = hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:12]
    target = Path(tempfile.gettempdir()) / f"q_judger_vllm_{digest}"
    target.mkdir(parents=True, exist_ok=True)
    for source_path in source.iterdir():
        target_path = target / source_path.name
        if source_path.is_file() and not target_path.exists():
            os.symlink(source_path, target_path)

    tokenizer_config = json.loads(
        source.joinpath("tokenizer_config.json").read_text(encoding="utf-8")
    )
    tokenizer_config["tokenizer_class"] = "Qwen2TokenizerFast"
    tokenizer_config_path = target / "tokenizer_config.json"
    tokenizer_config_path.unlink(missing_ok=True)
    tokenizer_config_path.write_text(
        json.dumps(tokenizer_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    processor_config_path = source / "processor_config.json"
    if processor_config_path.is_file():
        processor_config = json.loads(processor_config_path.read_text(encoding="utf-8"))
        video_config = processor_config.get("video_processor")
        if isinstance(video_config, dict):
            target.joinpath("video_preprocessor_config.json").write_text(
                json.dumps(video_config, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    return str(target)


class VllmJudge:
    def __init__(
        self,
        model_path: str,
        *,
        max_batch_size: int = 24,
        max_new_tokens: int = 4096,
        max_model_len: int = 8192,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        enforce_eager: bool = False,
        mm_encoder_attn_backend: str | None = None,
    ) -> None:
        _prepend_runtime_bin()
        from vllm import LLM, SamplingParams

        compat_path = _vllm_model_compat_path(model_path)
        self.processor = _load_qwen_tokenizer(compat_path)
        self.engine = LLM(
            model=compat_path,
            tokenizer=compat_path,
            dtype="bfloat16",
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_seqs=max_batch_size,
            tensor_parallel_size=tensor_parallel_size,
            limit_mm_per_prompt={"image": 1},
            enforce_eager=enforce_eager,
            mm_encoder_attn_backend=mm_encoder_attn_backend,
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
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
    ) -> None:
        _prepend_runtime_bin()
        from sglang import Engine

        self.processor = _load_qwen_tokenizer(model_path)
        self.engine = Engine(
            model_path=model_path,
            mem_fraction_static=gpu_memory_utilization,
            context_length=max_model_len,
            tp_size=tensor_parallel_size,
        )
        self.max_batch_size = max_batch_size
        self.sampling_params = {
            "max_new_tokens": max_new_tokens,
            "temperature": 0,
            "top_k": 1,
            "top_p": 1.0,
            "repetition_penalty": 1.05,
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
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    vllm_enforce_eager: bool = False,
    vllm_mm_encoder_attn_backend: str | None = None,
):
    normalized = backend.strip().lower()
    if normalized == "vllm":
        cls = VllmJudge
    elif normalized == "sglang":
        cls = SglangJudge
    else:
        raise ValueError(f"Unsupported accelerated Q-Judger backend: {backend!r}")
    common_kwargs = dict(
        max_batch_size=max_batch_size,
        max_new_tokens=max_new_tokens,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    if normalized == "vllm":
        common_kwargs.update(
            enforce_eager=vllm_enforce_eager,
            mm_encoder_attn_backend=vllm_mm_encoder_attn_backend,
        )
    return cls(model_path, **common_kwargs)
