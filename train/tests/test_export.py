"""Unit Tests for Safetensors Export and Schema Contract Validation."""

import tempfile
from pathlib import Path

from src.export import export_model_artifacts, validate_state_dict_schema
from src.model import CotierConfig, CotierForCausalLM


def test_schema_validation_and_export() -> None:
    # Full Cotier-0.45B configuration
    config = CotierConfig(
        vocab_size=32000,
        hidden_size=1024,
        intermediate_size=2816,
        num_attention_heads=16,
        num_cortical_stacks=4,
        max_recurrent_cycles=6,
    )
    model = CotierForCausalLM(config)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        export_model_artifacts(model, out_dir)

        assert (out_dir / "model.safetensors").exists()
        assert (out_dir / "config.json").exists()
        assert (out_dir / "tokenizer.json").exists()
        assert (out_dir / "anchors.jsonl").exists()

        # Check state dict validity directly
        state_dict = model.state_dict()
        if "lm_head.weight" not in state_dict:
            state_dict["lm_head.weight"] = model.embed_tokens.weight

        is_valid, errors = validate_state_dict_schema(state_dict, num_stacks=4)
        assert is_valid, f"Validation errors: {errors}"
