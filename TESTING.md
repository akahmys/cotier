# 🧪 Cotier Testing & Validation Strategy (Verification Rules)

This document details the testing methodology, validation suites, numerical parity verifications, and quality assurance processes for Cotier.

*(Note: Coding standards live in [`CODING.md`](CODING.md); security and license audits live in [`AUDITING.md`](AUDITING.md).)*

---

## 🎯 1. Test Hierarchy & Verification Rules

Cotier enforces strict verification rules across both the Python training subsystem and the Rust inference engine:

```
                  ┌──────────────────────────────┐
                  │ 1. E2E Tool & Agent Tests    │ (Cursor, Open-WebUI, MCP clients)
                  ├──────────────────────────────┤
                  │ 2. Server & Lifelong Growth  │ (axum SSE, SQLite buffer, Sleep LoRA)
                  ├──────────────────────────────┤
                  │ 3. Numerical Parity & Metal  │ (PyTorch vs Rust candle, Metal speed)
                  ├──────────────────────────────┤
                  │ 4. Unit & Shape Assertions   │ (Layer I–VI tests, loss functions)
                  └──────────────────────────────┘
```

### Verification Rule Matrix

| Rule | Area | Requirement | Enforcement |
| :--- | :--- | :--- | :--- |
| **Rule T1** | **Test Code Separation** | Standalone unit and integration tests MUST be placed in `server/tests/` or `train/tests/`. Do NOT pollute `src/` with dedicated test files. | Directory structure check |
| **Rule T2** | **Tensor Schema Contract** | Exported Safetensors weights MUST match the exact keys, shapes, and dtypes defined in `models/cotier-0.5b/tensor_schema.json`. | `scripts/test/verify_tensor_schema.py` |
| **Rule T3** | **Numerical Parity** | PyTorch and Rust `candle` inference engines MUST produce numerically identical logits ($L_\infty < 10^{-4}$). | `scripts/test/verify_parity.py` |
| **Rule T4** | **E2E Agent & Tool Contract** | The API server must support `/v1/models`, `/v1/chat/completions` (streaming & non-streaming), and tool-call JSON parsing. | `scripts/test/test_e2e_integration.py` |
| **Rule T5** | **Preemption & User Priority** | Hippocampal sleep consolidation must yield immediately (`< 50ms`) when a user inference request arrives. | Preemption integration test |
| **Rule T6** | **Zero Catastrophic Forgetting**| Lifelong sleep consolidation must maintain $\le 5\%$ performance delta on base benchmarks via 30% anchor replay. | Anchor replay test suite |

---

## 🔬 2. Key Subsystem Test Suites

### 1. Python Model & Training Tests (`train/tests/`)
- **Cortical Forward Pass**: Verify shapes across Layer I to VI ($B \times L \times D$).
- **PonderNet Halting Unit**: Verify $\sum p_k \approx 1.0$ and early exit mechanisms.
- **3-Loss Computation**: Numerical stability check for $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda_1 \mathcal{L}_{\text{pred\_error}} + \lambda_2 \mathcal{L}_{\text{ponder}}$.
- **Safetensors Export**: Ensure exported tensor keys match configuration names exactly.
- **Run**: `cd train && uv run pytest`

### 2. Rust Inference & Metal Engine Tests (`server/tests/`)
- **Zero-Copy Safetensors Loader**: Verify memory-mapped model weights load without allocation failures.
- **Prefill vs. Decode Attention**: Ensure KV Cache indexing produces identical results to non-cached sequential forward passes.
- **Metal Acceleration**: Verify inference on Apple Silicon GPU produces valid logits without NaN or float overflow.
- **Run**: `cd server && cargo test --workspace`

### 3. PyTorch $\leftrightarrow$ Rust Numerical Parity Cross-Check
To ensure the Rust inference engine faithfully replicates the trained PyTorch model:
- Generate reference outputs from PyTorch given a fixed input token sequence.
- Run the identical weights and token sequence through the Rust engine.
- **Tolerance Requirement**:
  $$\max \vert \text{Logits}_{\text{PyTorch}} - \text{Logits}_{\text{Rust}} \vert < 10^{-4} \quad (\text{bfloat16 / float32})$$
- **Run**: `uv run python scripts/test/verify_parity.py --model-dir models/cotier-0.5b --tolerance 1e-4`

### 4. Tensor Schema Contract Audit
- Validate that all 67 tensor weights match `models/cotier-0.5b/tensor_schema.json`:
- **Run**: `uv run python scripts/test/verify_tensor_schema.py --model-dir models/cotier-0.5b --schema-file models/cotier-0.5b/tensor_schema.json`

### 5. E2E Server, MCP & Hippocampal Integration Tests
- Verify API server startup, `/v1/models`, Chat Completions with Cortical Load metrics, +1 user feedback, and SQLite memory updates.
- **Run**: `uv run python scripts/test/test_e2e_integration.py`

---

## 📊 3. Performance & Memory Targets (KPIs)

| Metric | Target | Verification Method |
| :--- | :--- | :--- |
| **Unified Memory Footprint** | **$< 1.5$ GB** (KV Cache included) | Activity Monitor / Memory profiler |
| **Inference Throughput** | **$\ge 120$ tokens/sec** | `cotier bench` (Metal Native) |
| **Numerical Parity Error** | **$L_\infty < 10^{-4}$** | `verify_parity.py` |
| **Preemption Yield Latency** | **$< 50$ ms** | E2E concurrent request benchmark |
| **MCP Tool Argument Accuracy**| **$\ge 90\%$** | Glaive evaluation dataset |

---

## 🛠️ 4. Pre-Merge Quality Checklist

Before completing a task, milestone, or Pull Request:

- [ ] Python unit tests pass: `cd train && uv run pytest`.
- [ ] Rust compliance & tests pass: `cargo clippy --workspace --all-targets -- -D warnings` and `cargo test --workspace`.
- [ ] Master compliance suite passes: `bash scripts/audit/verify_compliance.sh`.
- [ ] Tensor schema contract passes: `verify_tensor_schema.py`.
- [ ] Numerical parity passes: `verify_parity.py`.
- [ ] E2E integration test passes: `test_e2e_integration.py`.

