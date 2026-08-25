"""Cotier-0.45B: 4-Stack Cortical Column Recurrent Reasoning Model.

Implements the Layer I-VI cortical column architecture, latent recurrence (k=1..6),
PonderNet halting unit, and weight tying according to ARCHITECTURE.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CotierConfig:
    """Configuration class for Cotier-0.45B model."""

    vocab_size: int = 32000
    hidden_size: int = 1024
    intermediate_size: int = 2816
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    head_dim: int = 64
    num_cortical_stacks: int = 4
    max_recurrent_cycles: int = 6
    recurrent_alpha: float = 0.1
    ponder_epsilon: float = 0.05
    ponder_lambda_geom: float = 0.2
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = True
    torch_dtype: str = "bfloat16"


class ModelOutput(NamedTuple):
    """Output tuple for Cotier forward pass."""

    logits: torch.Tensor  # Final step logits [B, L, V]
    all_step_logits: list[torch.Tensor]  # Logits for each k: list of [B, L, V]
    halting_probs: torch.Tensor  # Step probability distribution p_k [B, L, K]
    step_hidden_states: list[torch.Tensor]  # Latent hidden states z^(k) [B, L, D]
    z_l1_context: torch.Tensor  # Layer I global context z_L1 [B, 1, D]
    max_cycles: int  # Max cycles executed
    all_stacks_halting_probs: list[torch.Tensor] = []  # Halting probs across all cortical stacks


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization with FP32 upcasting."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS in float32 for numerical stability
        return x * torch.rsqrt(x.to(torch.float32).pow(2).mean(-1, keepdim=True) + self.eps).to(
            x.dtype
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        return self._norm(x) * self.weight.to(x.dtype)


class RotaryEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE)."""

    def __init__(
        self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0
    ) -> None:
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=torch.float32)
        inv_freq: torch.Tensor = self.inv_freq  # type: ignore[assignment]
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, H, L, head_dim]
        cos_cached: torch.Tensor = self.cos_cached  # type: ignore[assignment]
        sin_cached: torch.Tensor = self.sin_cached  # type: ignore[assignment]
        if seq_len > cos_cached.shape[0]:
            self._set_cos_sin_cache(seq_len)
            cos_cached = self.cos_cached  # type: ignore[assignment]
            sin_cached = self.sin_cached  # type: ignore[assignment]
        return cos_cached[:seq_len].to(x.dtype), sin_cached[:seq_len].to(x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Position Embedding to query and key tensors."""
    # cos, sin: [L, head_dim] -> [1, 1, L, head_dim]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class CausalSelfAttention(nn.Module):
    """Multi-Head Causal Self-Attention with RoPE."""

    def __init__(self, config: CotierConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # hidden_states: [B, L, D]
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # [B, L, D] -> [B, H, L, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(v, seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Scaled dot-product attention
        is_causal = attention_mask is None
        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, is_causal=is_causal
        )

        # [B, H, L, head_dim] -> [B, L, D]
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        )
        return cast(torch.Tensor, self.o_proj(attn_output))


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network."""

    def __init__(self, config: CotierConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D] -> [B, L, D_ffn] -> [B, L, D]
        return cast(torch.Tensor, self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class LayerVCore(nn.Module):
    """Layer V: Recurrent Attention & SwiGLU Core Engine."""

    def __init__(self, config: CotierConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn = SwiGLUFFN(config)
        self.alpha = config.recurrent_alpha

    def forward(
        self,
        z_prev: torch.Tensor,
        z_l1: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # z_prev: [B, L, D], z_l1: [B, 1, D]
        # a^(k) = CausalSelfAttention(z^(k-1))
        a_k = self.attn(z_prev, attention_mask=attention_mask)
        # u^(k) = RMSNorm(z^(k-1) + a^(k) + z_L1)
        u_k = self.attn_norm(z_prev + a_k + z_l1)
        # z_tilde^(k) = z^(k-1) + alpha * SwiGLU(ffn_norm(u^(k)))
        ffn_out = self.ffn(self.ffn_norm(u_k))
        z_tilde = z_prev + self.alpha * ffn_out
        # z^(k) = RMSNorm(z_tilde^(k))
        return cast(torch.Tensor, self.ffn_norm(z_tilde))


class LayerVIHalting(nn.Module):
    """Layer VI: PonderNet Halting Unit (Early Exit / Adaptive Computation)."""

    def __init__(self, config: CotierConfig) -> None:
        super().__init__()
        # Matches tensor_schema: weight [1, 1024], bias [1]
        self.weight = nn.Parameter(torch.empty(1, config.hidden_size))
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, z_k: torch.Tensor) -> torch.Tensor:
        # z_k: [B, L, D]
        # h_k = sigma(W_halt * z^(k) + b_halt) in FP32
        logits = F.linear(
            z_k.to(torch.float32), self.weight.to(torch.float32), self.bias.to(torch.float32)
        )
        return torch.sigmoid(logits).squeeze(-1)  # [B, L]


class CorticalStack(nn.Module):
    """Single Cortical Column Stack encompassing Layers I through VI."""

    def __init__(self, config: CotierConfig) -> None:
        super().__init__()
        self.config = config
        self.l1_global = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.l1_global_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.l2_lateral = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.l2_lateral_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.l4_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.l5_core = LayerVCore(config)
        self.l6_halt = LayerVIHalting(config)

    def compute_l1_context(self, context_tokens: torch.Tensor) -> torch.Tensor:
        # context_tokens: [B, L, D] -> mean pool to [B, 1, D]
        bar_context = context_tokens.mean(dim=1, keepdim=True)
        return cast(torch.Tensor, self.l1_global_norm(self.l1_global(bar_context)))

    def forward(
        self,
        z_in: torch.Tensor,
        z_l1_override: torch.Tensor | None = None,
        max_cycles: int | None = None,
        is_prefill: bool = False,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor, torch.Tensor]:
        # z_in: [B, L, D]
        # Layer IV: Sensory Gateway Normalization
        z_l4 = self.l4_norm(z_in)

        # Layer I: Top-Down Invariants & Modulator
        z_l1 = z_l1_override if z_l1_override is not None else self.compute_l1_context(z_in)

        # Layer II/III: Lateral Integration
        z_k = self.l2_lateral_norm(self.l2_lateral(z_l4 + z_l1))

        cycles = 1 if is_prefill else (max_cycles or self.config.max_recurrent_cycles)
        step_hidden_states: list[torch.Tensor] = []
        halting_unit_outputs: list[torch.Tensor] = []

        # Layer V Recurrent Loop (k = 1..K)
        for _ in range(cycles):
            z_k = self.l5_core(z_k, z_l1, attention_mask=attention_mask)
            step_hidden_states.append(z_k)
            # Layer VI Halting Unit
            h_k = self.l6_halt(z_k)  # [B, L]
            halting_unit_outputs.append(h_k)

        # Compute PonderNet Step Probability Mass p_k
        # h_stack: [B, L, K]
        h_stack = torch.stack(halting_unit_outputs, dim=-1)
        k_steps = h_stack.shape[-1]
        p_list: list[torch.Tensor] = []
        accum_not_halt = torch.ones_like(h_stack[..., 0])  # [B, L]

        for k in range(k_steps):
            h_k = h_stack[..., k]
            if k == k_steps - 1:
                # Remainder assigned to the final step to guarantee sum(p_k) == 1.0
                p_k = accum_not_halt
            else:
                p_k = h_k * accum_not_halt
                accum_not_halt = accum_not_halt * (1.0 - h_k)
            p_list.append(p_k)

        p_mass = torch.stack(p_list, dim=-1)  # [B, L, K]
        z_final = step_hidden_states[-1]  # Final step hidden state [B, L, D]

        return z_final, step_hidden_states, p_mass, z_l1


class CotierForCausalLM(nn.Module):
    """Cotier-0.45B Autoregressive Language Model with 4 Cortical Stacks."""

    def __init__(self, config: CotierConfig | None = None) -> None:
        super().__init__()
        self.config = config or CotierConfig()

        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.stacks = nn.ModuleList(
            [CorticalStack(self.config) for _ in range(self.config.num_cortical_stacks)]
        )
        self.norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        max_cycles: int | None = None,
        is_prefill: bool = False,
    ) -> ModelOutput:
        # input_ids: [B, L]
        # Token Embedding: [B, L, D]
        hidden_states = cast(torch.Tensor, self.embed_tokens(input_ids))

        last_stack_hidden_states: list[torch.Tensor] = []
        all_stacks_p_mass: list[torch.Tensor] = []
        last_stack_p_mass = torch.empty(0)
        z_l1_context = torch.empty(0)

        # Forward pass across 4 Cortical Stacks
        for stack_idx, stack in enumerate(self.stacks):
            stack_module: CorticalStack = stack  # type: ignore[assignment]
            z_final, step_states, p_mass, z_l1 = stack_module(
                hidden_states,
                max_cycles=max_cycles,
                is_prefill=is_prefill,
                attention_mask=attention_mask,
            )
            hidden_states = z_final
            all_stacks_p_mass.append(p_mass)
            if stack_idx == len(self.stacks) - 1:
                last_stack_hidden_states = step_states
                last_stack_p_mass = p_mass
                z_l1_context = z_l1

        # Compute LM logits for each recurrent step
        all_step_logits: list[torch.Tensor] = []
        for step_state in last_stack_hidden_states:
            normed_step = self.norm(step_state)
            step_logits = cast(torch.Tensor, self.lm_head(normed_step))
            all_step_logits.append(step_logits)

        final_logits = all_step_logits[-1]

        return ModelOutput(
            logits=final_logits,
            all_step_logits=all_step_logits,
            halting_probs=last_stack_p_mass,
            step_hidden_states=last_stack_hidden_states,
            z_l1_context=z_l1_context,
            max_cycles=len(last_stack_hidden_states),
            all_stacks_halting_probs=all_stacks_p_mass,
        )
