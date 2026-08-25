"""Cotier Joint 3-Loss Objective Function.

Implements task expected cross-entropy, predictive coding error, and
PonderNet geometric regularization KL divergence with numerical stability safeguards.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model import ModelOutput


@dataclass
class LossComponents:
    """Breakdown of loss terms for logging and metrics."""

    total_loss: torch.Tensor
    task_loss: torch.Tensor
    pred_error_loss: torch.Tensor
    ponder_loss: torch.Tensor


class CotierJointLoss(nn.Module):
    """3-Loss Composite Objective for Cotier-0.45B.

    L_total = L_task + lambda_pred * L_pred_error + lambda_ponder * L_ponder
    """

    def __init__(
        self,
        lambda_pred_error: float = 0.1,
        lambda_ponder: float = 0.01,
        lambda_geom: float = 0.2,
        eps: float = 1e-7,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        self.lambda_pred_error = lambda_pred_error
        self.lambda_ponder = lambda_ponder
        self.lambda_geom = lambda_geom
        self.eps = eps
        self.ignore_index = ignore_index

    def compute_geometric_prior(self, max_k: int, device: torch.device) -> torch.Tensor:
        """Constructs target geometric distribution q_1..K."""
        # q_k = lambda * (1 - lambda)^(k - 1)
        k_indices = torch.arange(1, max_k + 1, dtype=torch.float32, device=device)
        probs = self.lambda_geom * ((1.0 - self.lambda_geom) ** (k_indices - 1))
        # Assign remainder mass to step K so sum(q) == 1.0
        if max_k > 1:
            probs[-1] = 1.0 - probs[:-1].sum()
        return probs.clamp(min=self.eps)

    def forward(
        self,
        model_output: ModelOutput,
        targets: torch.Tensor,
    ) -> LossComponents:
        """Computes composite loss given model output and target labels.

        Args:
            model_output: ModelOutput containing all_step_logits, halting_probs,
                step_hidden_states, and z_l1_context.
            targets: Target token IDs tensor [B, L].
        """
        all_step_logits = model_output.all_step_logits
        halting_probs = model_output.halting_probs  # [B, L, K]
        step_hidden_states = model_output.step_hidden_states  # List of [B, L, D]
        z_l1 = model_output.z_l1_context  # [B, 1, D]
        k_steps = len(all_step_logits)

        # 1. Expected Task Cross-Entropy Loss: sum_{k=1}^K p_k * CE(y_hat^(k), y)
        # Shift logits and targets for next-token prediction
        batch_size, seq_len = targets.shape
        shift_targets = targets[..., 1:].contiguous()  # [B, L-1]

        # halting_probs for shifted tokens: [B, L-1, K]
        shift_p = halting_probs[:, :-1, :].contiguous()  # [B, L-1, K]

        step_ce_losses: list[torch.Tensor] = []
        for step_logits in all_step_logits:
            shift_logits = step_logits[:, :-1, :].contiguous()  # [B, L-1, V]
            # per-token CE loss [B, L-1]
            ce = F.cross_entropy(
                shift_logits.view(-1, shift_logits.shape[-1]),
                shift_targets.view(-1),
                ignore_index=self.ignore_index,
                reduction="none",
            ).view(batch_size, seq_len - 1)
            step_ce_losses.append(ce)

        # [B, L-1, K]
        stacked_ce = torch.stack(step_ce_losses, dim=-1)

        # Valid token mask: ignore tokens where target is ignore_index
        valid_mask = (shift_targets != self.ignore_index).float()  # [B, L-1]
        valid_count = valid_mask.sum().clamp(min=1.0)

        # Weighted CE sum per token: sum_k p_k * CE_k
        weighted_ce = (shift_p * stacked_ce).sum(dim=-1)  # [B, L-1]
        task_loss = (weighted_ce * valid_mask).sum() / valid_count

        # 2. Predictive Coding Error Loss: (1/K) * sum_{k=1}^K || z^(k) - z_L1 ||_2^2
        pred_errors: list[torch.Tensor] = []
        for z_k in step_hidden_states:
            # MSE between z_k [B, L, D] and z_L1 [B, 1, D]
            mse = (z_k - z_l1).pow(2).mean(dim=-1)  # [B, L]
            pred_errors.append(mse)

        stacked_pred_error = torch.stack(pred_errors, dim=-1).mean(dim=-1)  # [B, L]
        # Mask with full token sequence mask (if available) or mean
        pred_error_loss = stacked_pred_error.mean()

        # 3. PonderNet Geometric KL Divergence Loss: D_KL(p_1..K || Geom(lambda))
        all_p_stacks = (
            model_output.all_stacks_halting_probs
            if model_output.all_stacks_halting_probs
            else [halting_probs]
        )
        q_prior = self.compute_geometric_prior(k_steps, device=halting_probs.device)  # [K]
        q_clamped = q_prior.view(1, 1, k_steps).clamp(min=self.eps)

        stack_ponder_losses: list[torch.Tensor] = []
        for p_stack in all_p_stacks:
            p_stack_shift = p_stack[:, :-1, :].contiguous().clamp(min=self.eps)
            kl_div = (p_stack_shift * (p_stack_shift.log() - q_clamped.log())).sum(dim=-1)
            stack_ponder_loss = (kl_div * valid_mask).sum() / valid_count
            stack_ponder_losses.append(stack_ponder_loss)

        ponder_loss = torch.stack(stack_ponder_losses).mean()

        # Composite Loss
        total_loss = (
            task_loss
            + (self.lambda_pred_error * pred_error_loss)
            + (self.lambda_ponder * ponder_loss)
        )

        return LossComponents(
            total_loss=total_loss,
            task_loss=task_loss,
            pred_error_loss=pred_error_loss,
            ponder_loss=ponder_loss,
        )
