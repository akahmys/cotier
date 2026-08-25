# 💻 Cotier Coding Standards & Hardening Protocol (CR-20 Protocol)

This document defines the coding conventions, safety standards (**CR-20 Protocol: Cotier Reliability-20**), and architectural patterns required across the Python training pipeline, Rust inference engine, and Apple Silicon Metal shaders in the Cotier repository.

---

## 🛡️ 1. The CR-20 Hardening Rules

Derived from aerospace safety and high-assurance real-time systems, the **CR-20 Protocol** guarantees mathematical determinism, zero-panic memory safety, and optimal Apple Silicon Metal execution across the entire PyTorch/Rust lifecycle.

### Rule Summary Matrix

| Rule | Area | Requirement | Enforcement |
| :--- | :--- | :--- | :--- |
| **Rule 1** | **Function Length** | Max 50 lines for standard functions.<br>Max 150 lines for `// CR-20 Limit: Dispatcher` (e.g. forward pass or CLI loop). | `./scripts/audit/verify_compliance.sh` |
| **Rule 2** | **Panic Prevention** | `unwrap()` and `expect()` are forbidden in production code. Use `?`, `unwrap_or()`, or explicit error matching. | Automated grep / Clippy |
| **Rule 3** | **Unsafe Ban** | `unsafe` blocks are forbidden (`workspace.lints.rust.unsafe_code = "forbid"` and `#![forbid(unsafe_code)]`). | Rustc compiler lint |
| **Rule 4** | **Control Flow** | Avoid deep nesting (`if let` / `match`). Prefer early return with `?`. | Code review / Clippy |
| **Rule 5** | **Match Exhaustiveness** | Wildcard arms (`_ =>`) are forbidden when matching **domain enums** (`LayerType`, `FeedbackKind`, `ToolStatus`). | `clippy::wildcard_enum_match_arm` |
| **Rule 6** | **Bounded Recurrence** | Unbounded recursion and infinite loops are forbidden. Layer V recurrent loops MUST be bounded ($k \le K_{\text{max}} = 6$). | Unit tests / Architecture review |
| **Rule 7** | **Global State** | `static mut` and global mutable state are forbidden. Shared state must use `Arc<Mutex<T>>`, `ArcSwap`, or atomics. | Automated grep check |
| **Rule 8** | **Invalid State** | Use type-safe `enum` states instead of boolean flags or nested `Option`s (Make Illegal States Unrepresentable). | Architecture review |
| **Rule 9** | **Pure Rust & Native Metal** | No dependency may compile C/C++ source or bind third-party native C libraries (No `cc` crate). GPU compute is strictly native Apple Metal. | `./scripts/audit/verify_compliance.sh` |
| **Rule 10** | **Determinism** | `HashMap` and `HashSet` are forbidden in core inference and tensor pipelines. Use `BTreeMap`, `BTreeSet`, or contiguous indexed `Vec`. | Automated grep check |
| **Rule 11** | **Error Transparency** | Return typed `thiserror` enums (`CotierError`). String-based errors (`Result<T, String>`) are forbidden in public APIs. | Automated grep check |
| **Rule 12** | **No Error Swallowing** | `filter_map(Result::ok)` and silent error dropping in model forward loops are forbidden. | Automated grep check |
| **Rule 13** | **Zero-Allocation Inference** | Minimize allocations in the autoregressive decode loop. Prioritize pre-allocated KV Buffers and in-place tensor updates. | Memory profiling / Benchmark |
| **Rule 14** | **Test Code Separation** | Integration and unit tests MUST be placed in `server/tests/` or `train/tests/`. Do NOT pollute `src/` with standalone test files. | Directory structure check |
| **Rule 15** | **Tensor Schema Contract** | Safetensors tensor keys, shapes, and dtypes MUST strictly adhere to `models/cotier-0.5b/tensor_schema.json`. Renaming is forbidden. | `scripts/test/verify_tensor_schema.py` |
| **Rule 16** | **PyTorch-Rust Parity** | The Rust `candle` inference engine must output logits numerically identical to PyTorch within $L_\infty < 10^{-4}$. | `scripts/test/verify_parity.py` |
| **Rule 17** | **FP32 Numerical Upcasting** | Weights are stored in `bfloat16`/`float32`, but RMSNorm variance, Softmax, PonderNet halting sums, and Surprise MUST be computed in FP32. | Code review / Parity test |
| **Rule 18** | **Secrets and PII** | No credential, API key, personal name, or confidential data may be committed. | `betterleaks` via `verify_compliance.sh` |
| **Rule 19** | **Licences** | Every Rust and Python dependency licence must be compliant (No GPL/AGPL copyleft violations). | `cargo deny` & `check_python_licenses.sh` |
| **Rule 20** | **Preemption & User Priority** | Hippocampal sleep consolidation must yield immediately (`< 50ms`) upon receiving user inference requests. | Preemption integration test |

---

### Rule 9 in detail: Pure Rust and Metal Native Policy

Cotier targets Apple Silicon Unified Memory without external C/C++ runtime bloat or opaque FFI bridges.

| Category | Example | Policy |
| :--- | :--- | :--- |
| **Forbidden** | `llama.cpp` bindings, `onnxruntime-sys`, `libtorch-sys`, `openssl-sys` | Compiles vendored C/C++; pulls `cc` build dependency. |
| **Allowed** | `candle-core` (with pure Metal backend), `objc`, `core-foundation-sys` | Pure Rust or platform API declarations provided by macOS. |

**Enforcement**: `cargo tree` must verify that no crate named `cc` exists in the dependency tree for `aarch64-apple-darwin`.

---

### Rule 5 in detail: Domain Enum Exhaustiveness

When matching domain enums (such as `FeedbackKind`, `ModelStage`, or `ToolExecutionStatus`), wildcard arms (`_ =>`) are strictly forbidden. Adding a new state or tool type must break the build at every location that processes it, preventing silent fallback bugs.

---

## 🐍 2. Python Training Standards (`train/`)

1. **Strict Static Typing**:
   - All modules under `train/src/` must be fully typed using Python 3.11+ type hints and pass `mypy --strict src/`.
2. **Explicit Tensor Dimensions**:
   - Every tensor transformation must be documented and asserted with shape comments:
     ```python
     # hidden_states: [B, L, D]
     # z_L1: [B, 1, D]
     # halting_probs: [B, L, K]
     ```
3. **Epsilon Clamping for PonderNet**:
   - Probability distributions $p_k$ and geometric targets $\text{Geom}(\lambda)$ must enforce epsilon clamping ($\epsilon = 10^{-7}$) before computing KL divergence or Cross Entropy to prevent `NaN` gradients.
4. **Reproducibility & Seed Discipline**:
   - All random number generators (`torch.manual_seed`, `random.seed`) must be deterministically initialized in all training scripts.

---

## 🏛️ 3. Layer Architecture Boundaries

Code must strictly respect the 4-stack Cortical Column architecture defined in [`ARCHITECTURE.md`](ARCHITECTURE.md):

* **Layer I (Top-Down Invariants)**: Must not be mutated by recurrent loops; serves as the conditioning context ($z_{\text{L1}}$).
* **Layer IV (Sensory Gateway)**: Handles input token embeddings and cross-layer normalization only.
* **Layer V (Recurrent Attention & SwiGLU)**: Core iterative engine ($k=1\dots K_{\text{max}}$). Must remain stateless between tokens except via the explicit KV Cache and hidden states.
* **Layer VI (PonderNet Halting Unit)**: Computes stopping probability $h_k$ and early exits cleanly without leaking memory or gradients.

---

## 🛠️ 4. Automated Enforcement & Compliance Verification

Run the master compliance suite to verify all 20 rules:

```bash
# Run master compliance suite
bash scripts/audit/verify_compliance.sh

# Run end-to-end integration and parity tests
python scripts/test/verify_parity.py --model-dir models/cotier-0.5b
python scripts/test/test_e2e_integration.py
```
