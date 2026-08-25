# 💻 Cotier Coding Standards & Hardening Protocol

This document defines the coding conventions, safety standards, and architectural patterns required across the Python training pipeline and Rust inference/lifelong learning engine in the Cotier repository.

---

## 🛡️ 1. Rust Server Hardening Rules (`server/`)

Derived from aerospace safety and deterministic runtime principles, these rules guarantee memory safety, crash prevention, and optimal Apple Silicon Metal execution.

### Rule Summary Matrix

| Rule | Area | Requirement | Enforcement |
| :--- | :--- | :--- | :--- |
| **Rule 1** | Function Length | Max 50 lines for standard functions.<br>Max 150 lines for cortical dispatcher / forward passes. | Linter / Review |
| **Rule 2** | Panic Prevention | `unwrap()` and `expect()` are forbidden in production code. Use `?`, `unwrap_or()`, or explicit error matching. | Automated grep / Clippy |
| **Rule 3** | Unsafe Ban | `unsafe` blocks are forbidden (`#![forbid(unsafe_code)]`). | Rustc lint |
| **Rule 4** | Match Exhaustiveness | Wildcard arms (`_ =>`) are forbidden when matching domain enums (e.g. `LayerType`, `ToolStatus`, `FeedbackKind`). | `clippy::wildcard_enum_match_arm` |
| **Rule 5** | Error Transparency | Return typed `thiserror` enums (`CotierError`). String-based errors (`Result<T, String>`) are forbidden. | Automated grep check |
| **Rule 6** | No Error Swallowing | `filter_map(Result::ok)` and silent error dropping in model forward loops are forbidden. | Automated grep check |
| **Rule 7** | Global State | `static mut` is strictly forbidden. Shared mutable weights must use `ArcSwap` or `Arc<RwLock<T>>`. | Automated grep check |
| **Rule 8** | Deterministic Collections | `HashMap` iteration is forbidden where order affects tensor generation or KV Cache lookup. Use `BTreeMap` or contiguous indexed `Vec`. | Code review / Clippy |
| **Rule 9** | Zero-Allocation Inference | In-place tensor manipulation or pre-allocated KV Buffers must be prioritized during the token decode loop. | Benchmark / Profiling |
| **Rule 10** | Preemption Safety | Background sleep-learning threads must yield immediately upon receiving high-priority inference requests. | Tokio cancellation token |
| **Rule 11** | Tensor Schema Contract | Safetensors tensor keys and shapes MUST strictly follow `models/cotier-0.5b/tensor_schema.json`. Arbitrary renaming is forbidden. | Parity contract test |
| **Rule 12** | Numerical Upcasting | Weights are stored in `bfloat16`/`float32`, but RMSNorm variance, Softmax, and PonderNet halting sums MUST be computed in FP32. | Code review / Metal shader |

---

## 🐍 2. Python Training Standards (`train/`)

The Python training system builds the base representation, recurrent reasoning capability, and tool-use alignment.

### Python Hardening Guidelines

1. **Strict Static Typing**:
   - All modules under `train/src/` must be strictly typed using `typing` / Python 3.11+ type hints and pass `mypy --strict`.
2. **Explicit Tensor Dimensions**:
   - Every tensor transformation must be documented and asserted with shape comments:
     ```python
     # x: [B, L, D]
     # z_L1: [B, 1, D]
     # z_L4: [B, L, D]
     ```
3. **Reproducibility & Seed Discipline**:
   - All random number generators (`torch.manual_seed`, `numpy.random.seed`, `random.seed`) must be deterministically initialized in all training scripts.
4. **Numerical Stability in PonderNet & KL Loss**:
   - Probability distributions $p_k$ and geometric targets $\text{Geom}(\lambda)$ must enforce epsilon clamping ($\epsilon = 10^{-7}$) before computing KL divergence or Cross Entropy to prevent `NaN` gradients.
5. **Clean Export Schema**:
   - The export utility (`export.py`) must validate the generated `safetensors` against `models/cotier-0.5b/tensor_schema.json` before saving.
6. **LoRA Safety & Snapshot Rollback**:
   - Dynamic adapter updates must maintain rolling versioned snapshots (`plastic_adapter.v1..v3`) and evaluate against `anchors.jsonl` before activation.

---

## 🏛️ 3. Layer Architecture Boundaries

Code must respect the architectural layers defined in [`ARCHITECTURE.md`](ARCHITECTURE.md):

- **Layer I (Top-Down Invariants)**: Must not be mutated by recurrent loops; serves as the conditioning context.
- **Layer IV (Sensory Gateway)**: Handles input token embeddings and cross-layer normalization only.
- **Layer V (Recurrent Attention & SwiGLU)**: Core iterative engine ($k=1\dots K_{\text{max}}$). Must remain stateless between tokens except via the explicit KV Cache and hidden states.
- **Layer VI (PonderNet Halting Unit)**: Computes stopping probability $h_k$ and early exits cleanly without leaking memory or gradients.
