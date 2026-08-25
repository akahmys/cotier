# 🧪 Cotier Testing & Validation Strategy

This document details the testing methodology, validation suites, numerical parity verifications, and quality assurance processes for Cotier.

---

## 🎯 1. Test Pyramid & Validation Hierarchy

Cotier employs a 4-tier testing hierarchy across both the Python training subsystem and the Rust inference engine:

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

---

## 🔬 2. Key Subsystem Test Suites

### 1. Python Model & Training Tests (`train/tests/`)
- **Cortical Forward Pass**: Verify shapes across Layer I to VI ($B \times L \times D$).
- **PonderNet Halting Unit**: Verify $\sum p_k \approx 1.0$ and early exit mechanisms.
- **3-Loss Computation**: Numerical stability check for $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda_1 \mathcal{L}_{\text{pred\_error}} + \lambda_2 \mathcal{L}_{\text{ponder}}$.
- **Safetensors Export**: Ensure exported tensor keys match configuration names exactly.

### 2. Rust Inference & Metal Engine Tests (`server/tests/`)
- **Zero-Copy Safetensors Loader**: Verify memory-mapped model weights load without allocation failures.
- **Prefill vs. Decode Attention**: Ensure KV Cache indexing produces identical results to non-cached sequential forward passes.
- **Metal Acceleration**: Verify inference on Apple Silicon GPU produces valid logits without NaN or float overflow.

### 3. Numerical Parity Cross-Check (PyTorch $\leftrightarrow$ Rust)
To ensure the Rust inference engine faithfully replicates the trained model:
- Generate reference outputs from PyTorch given a fixed input token sequence.
- Run the identical weights and token sequence through the Rust engine.
- **Tolerance Requirement**:
  $$\max \vert \text{Logits}_{\text{PyTorch}} - \text{Logits}_{\text{Rust}} \vert < 10^{-4} \quad (\text{bfloat16 / float32})$$
- Run: `python scripts/test/verify_parity.py`

### 4. Tokenizer & Schema Parity Tests
- **Tokenizer Parity**: Verify identical token IDs across Python HuggingFace and Rust `tokenizers` for Japanese, English, Code, and Tool-calling JSON:
  - Run: `python scripts/test/verify_tokenizer_parity.py`
- **Tensor Schema Contract**: Validate that all weights match `models/cotier-0.5b/tensor_schema.json`:
  - Run: `python scripts/test/verify_tensor_schema.py`

### 5. Lifelong Growth & Anti-Forgetting Tests (`server/src/learner.rs`)
- **Episode Ingestion**: Verify high-surprise dialogues and positive user feedback (+1) are saved to SQLite.
- **Guardrail Filter**: Negative feedback (-1) or syntax errors must not trigger LoRA fine-tuning.
- **Replay Mixture**: Validate that 30% anchor data (`anchors.jsonl`) is blended into every sleep consolidation step.
- **Preemption**: Assert sleep-learning yields immediately when a new chat completion request arrives.

### 6. MCP / Tool-Use Benchmarks
- **JSON Schema Strictness**: Verify output format matches `<tool_call>{"name": ..., "arguments": ...}</tool_call>`.
- **Glaive Eval Benchmark**: Target $\ge 90\%$ accuracy on tool parameter extraction.

---

## 📊 3. Performance & Memory Benchmarks

| Metric | Target | Verification Command |
| :--- | :--- | :--- |
| **Inference Throughput** | $\ge 120$ tokens/sec | `cargo bench --package cotier-server` |
| **Unified Memory Footprint** | $< 1.5$ GB (KV Cache included) | Activity Monitor / Memory profiler |
| **Time to First Token (TTFT)** | $< 50$ ms for 512 tokens prefill | `cotier bench --prefill 512` |
| **Numerical Parity** | $\Delta < 10^{-4}$ | `python scripts/verify_parity.py` |

---

## 🛠️ 4. Pre-Merge Quality Checklist

Before completing a task, milestone, or Pull Request:

- [ ] Python lint & typing passes: `uv run ruff check .` and `uv run mypy src/`.
- [ ] Rust compliance passes: `cargo clippy --workspace --all-targets -- -D warnings`.
- [ ] Rust formatting is clean: `cargo fmt --all --check`.
- [ ] All unit & integration tests pass: `cargo test --workspace --features metal`.
- [ ] Model weights export test succeeds with numerical parity verified.
- [ ] `cargo deny check licenses` passes with no license violations.
- [ ] Secret scan passes: `betterleaks dir .`.
