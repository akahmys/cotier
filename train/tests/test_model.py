"""Unit Tests for Cotier-0.45B PyTorch Cortical Model."""

import torch

from src.model import (
    CorticalStack,
    CotierConfig,
    CotierForCausalLM,
    RMSNorm,
    RotaryEmbedding,
)


def test_rmsnorm_forward_shape_and_dtype() -> None:
    norm = RMSNorm(1024)
    x = torch.randn(2, 16, 1024, dtype=torch.bfloat16)
    out = norm(x)
    assert out.shape == (2, 16, 1024)
    assert out.dtype == torch.bfloat16


def test_rotary_embedding() -> None:
    rope = RotaryEmbedding(dim=64, max_position_embeddings=2048)
    dummy_x = torch.randn(2, 16, 32, 64)
    cos, sin = rope(dummy_x, seq_len=32)
    assert cos.shape == (32, 64)
    assert sin.shape == (32, 64)


def test_cortical_stack_forward_and_halting_distribution() -> None:
    config = CotierConfig(
        hidden_size=256,
        intermediate_size=512,
        num_attention_heads=4,
        head_dim=64,
        max_recurrent_cycles=6,
        num_cortical_stacks=1,
    )
    stack = CorticalStack(config)
    x = torch.randn(2, 8, 256)

    z_final, step_states, p_mass, z_l1 = stack(x, max_cycles=6)

    assert z_final.shape == (2, 8, 256)
    assert len(step_states) == 6
    assert p_mass.shape == (2, 8, 6)
    assert z_l1.shape == (2, 1, 256)

    # Sum of halting probability mass p_k across K should be 1.0 everywhere
    prob_sums = p_mass.sum(dim=-1)
    assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5)


def test_cotier_causal_lm_forward() -> None:
    config = CotierConfig(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_attention_heads=4,
        head_dim=64,
        num_cortical_stacks=2,
        max_recurrent_cycles=4,
    )
    model = CotierForCausalLM(config)
    input_ids = torch.randint(0, 1000, (2, 12))

    output = model(input_ids, max_cycles=4)

    assert output.logits.shape == (2, 12, 1000)
    assert len(output.all_step_logits) == 4
    for step_logits in output.all_step_logits:
        assert step_logits.shape == (2, 12, 1000)
    assert output.halting_probs.shape == (2, 12, 4)
    assert output.max_cycles == 4
