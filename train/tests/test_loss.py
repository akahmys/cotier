"""Unit Tests for Cotier Joint 3-Loss Objective."""

import torch

from src.loss import CotierJointLoss
from src.model import CotierConfig, CotierForCausalLM


def test_cotier_joint_loss_computation_and_backward() -> None:
    config = CotierConfig(
        vocab_size=1000,
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=2,
        head_dim=64,
        num_cortical_stacks=2,
        max_recurrent_cycles=3,
    )
    model = CotierForCausalLM(config)
    loss_fn = CotierJointLoss(lambda_pred_error=0.1, lambda_ponder=0.01, lambda_geom=0.2)

    input_ids = torch.randint(0, 1000, (2, 8))
    targets = input_ids.clone()

    output = model(input_ids, max_cycles=3)
    losses = loss_fn(output, targets)

    assert not torch.isnan(losses.total_loss)
    assert not torch.isinf(losses.total_loss)
    assert losses.total_loss.item() > 0.0
    assert losses.task_loss.item() > 0.0
    assert losses.pred_error_loss.item() >= 0.0

    # Test gradient backward pass
    losses.total_loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient missing for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN in gradient for {name}"
