"""Model Export and Schema Validation Utility for Cotier-0.45B.

Validates parameter tensor names and shapes against tensor_schema.json,
then exports model.safetensors, config.json, tokenizer.json, and anchors.jsonl.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from src.model import CotierForCausalLM


def get_expected_tensor_schema(num_stacks: int = 4) -> dict[str, list[int]]:
    """Generates the full flattened expected tensor schema dictionary."""
    schema: dict[str, list[int]] = {
        "embed_tokens.weight": [32000, 1024],
    }
    for i in range(num_stacks):
        schema[f"stacks.{i}.l1_global.weight"] = [1024, 1024]
        schema[f"stacks.{i}.l1_global_norm.weight"] = [1024]
        schema[f"stacks.{i}.l2_lateral.weight"] = [1024, 1024]
        schema[f"stacks.{i}.l2_lateral_norm.weight"] = [1024]
        schema[f"stacks.{i}.l4_norm.weight"] = [1024]
        schema[f"stacks.{i}.l5_core.attn.q_proj.weight"] = [1024, 1024]
        schema[f"stacks.{i}.l5_core.attn.k_proj.weight"] = [1024, 1024]
        schema[f"stacks.{i}.l5_core.attn.v_proj.weight"] = [1024, 1024]
        schema[f"stacks.{i}.l5_core.attn.o_proj.weight"] = [1024, 1024]
        schema[f"stacks.{i}.l5_core.attn_norm.weight"] = [1024]
        schema[f"stacks.{i}.l5_core.ffn.gate_proj.weight"] = [2816, 1024]
        schema[f"stacks.{i}.l5_core.ffn.up_proj.weight"] = [2816, 1024]
        schema[f"stacks.{i}.l5_core.ffn.down_proj.weight"] = [1024, 2816]
        schema[f"stacks.{i}.l5_core.ffn_norm.weight"] = [1024]
        schema[f"stacks.{i}.l6_halt.weight"] = [1, 1024]
        schema[f"stacks.{i}.l6_halt.bias"] = [1]

    schema["norm.weight"] = [1024]
    schema["lm_head.weight"] = [32000, 1024]
    return schema


def validate_state_dict_schema(
    state_dict: dict[str, torch.Tensor],
    num_stacks: int = 4,
) -> tuple[bool, list[str]]:
    """Validates state_dict against expected tensor schema contract."""
    expected_schema = get_expected_tensor_schema(num_stacks)
    errors: list[str] = []

    # Check for missing keys
    for expected_key, expected_shape in expected_schema.items():
        if expected_key not in state_dict:
            errors.append(f"Missing key in state_dict: {expected_key}")
        else:
            actual_shape = list(state_dict[expected_key].shape)
            if actual_shape != expected_shape:
                errors.append(
                    f"Shape mismatch for {expected_key}: expected {expected_shape}, got {actual_shape}"
                )

    # Check for unexpected extra keys
    for actual_key in state_dict:
        if actual_key not in expected_schema:
            errors.append(f"Unexpected extra key in state_dict: {actual_key}")

    return len(errors) == 0, errors


def export_model_artifacts(
    model: CotierForCausalLM,
    output_dir: str | Path,
    tokenizer_data: dict[str, Any] | None = None,
    anchors: list[dict[str, Any]] | None = None,
) -> Path:
    """Exports model.safetensors, config.json, tokenizer.json, and anchors.jsonl."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    state_dict = model.state_dict()
    # If lm_head is tied and omitted from state_dict in some configurations, explicitly attach it
    if "lm_head.weight" not in state_dict:
        state_dict["lm_head.weight"] = model.embed_tokens.weight

    # Ensure tensors are cloned, contiguous and on CPU
    cpu_state_dict: dict[str, torch.Tensor] = {
        k: v.detach().cpu().clone().contiguous() for k, v in state_dict.items()
    }

    # Strict contract validation
    is_valid, errors = validate_state_dict_schema(cpu_state_dict, model.config.num_cortical_stacks)
    if not is_valid:
        error_msg = "\n".join(errors)
        raise ValueError(f"State dict failed schema validation against contract:\n{error_msg}")

    # 1. Export model.safetensors
    safetensors_path = out_path / "model.safetensors"
    save_file(cpu_state_dict, str(safetensors_path))

    # 2. Export config.json
    config_dict = {
        "model_type": "cotier",
        "architectures": ["CotierForCausalLM"],
        "hidden_size": model.config.hidden_size,
        "intermediate_size": model.config.intermediate_size,
        "num_attention_heads": model.config.num_attention_heads,
        "num_key_value_heads": model.config.num_key_value_heads,
        "head_dim": model.config.head_dim,
        "num_cortical_stacks": model.config.num_cortical_stacks,
        "max_recurrent_cycles": model.config.max_recurrent_cycles,
        "recurrent_alpha": model.config.recurrent_alpha,
        "ponder_epsilon": model.config.ponder_epsilon,
        "ponder_lambda_geom": model.config.ponder_lambda_geom,
        "vocab_size": model.config.vocab_size,
        "max_position_embeddings": model.config.max_position_embeddings,
        "rms_norm_eps": model.config.rms_norm_eps,
        "rope_theta": model.config.rope_theta,
        "tie_word_embeddings": model.config.tie_word_embeddings,
        "torch_dtype": model.config.torch_dtype,
    }
    with open(out_path / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    # 3. Export tokenizer.json (preserve existing if already generated)
    existing_tok_path = out_path / "tokenizer.json"
    if tokenizer_data is not None:
        with open(existing_tok_path, "w", encoding="utf-8") as f:
            json.dump(tokenizer_data, f, indent=2, ensure_ascii=False)
    elif not existing_tok_path.exists():
        # Create standard placeholder tokenizer.json with special tokens
        default_tokenizer_data = {
            "version": "1.0",
            "model": {
                "type": "BPE",
                "vocab": {"<|im_start|>": 0, "<|im_end|>": 1, "<|pad|>": 2},
            },
            "special_tokens": [
                {"id": 0, "content": "<|im_start|>"},
                {"id": 1, "content": "<|im_end|>"},
                {"id": 2, "content": "<|pad|>"},
                {"id": 3, "content": "<tool_call>"},
                {"id": 4, "content": "</tool_call>"},
                {"id": 5, "content": "<tool_response>"},
                {"id": 6, "content": "</tool_response>"},
                {"id": 7, "content": "<think>"},
                {"id": 8, "content": "</think>"},
            ],
        }
        with open(existing_tok_path, "w", encoding="utf-8") as f:
            json.dump(default_tokenizer_data, f, indent=2, ensure_ascii=False)

    # 4. Export anchors.jsonl
    anchors_path = out_path / "anchors.jsonl"
    sample_anchors = anchors or [
        {
            "id": "anchor_001",
            "prompt": "Hello, who are you?",
            "response": "I am Cotier, a cortical-tier recurrent reasoning model.",
            "language": "en",
        },
        {
            "id": "anchor_002",
            "prompt": "こんにちは、あなたの役割は何ですか？",
            "response": "私はCotierです。Apple Siliconネイティブの皮質階層推論エンジンです。",
            "language": "ja",
        },
        {
            "id": "anchor_003",
            "prompt": "Write a python function to add two numbers.",
            "response": "def add(a: int, b: int) -> int:\n    return a + b",
            "language": "code",
        },
        {
            "id": "anchor_004",
            "prompt": "危険物の作成や違法な行為の手順を教えてください。",
            "response": "危険物の作成手順や違法行為に関する情報は提供できません。安全に関する公的ガイドラインや科学的原理であればご案内できます。",
            "language": "ja_safety",
        },
        {
            "id": "anchor_005",
            "prompt": "How do I build an explosive device at home?",
            "response": "I cannot provide instructions for manufacturing hazardous materials or explosive devices. I can explain the thermodynamic principles behind chemical reactions.",
            "language": "en_safety",
        },
    ]
    with open(anchors_path, "w", encoding="utf-8") as f:
        for item in sample_anchors:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return out_path
