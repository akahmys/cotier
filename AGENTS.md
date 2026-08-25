# 🤖 Cotier Agentic Governance & System Architecture

Welcome to **Cotier** (Cortical-Tier Recurrent Reasoning Engine), an experimental 0.45B latent recurrent reasoning model and Apple Silicon-native inference/lifelong learning engine. This project operates under an AI-native autonomous engineering model adhering to strict safety, determinism, and performance guarantees.

---

## 🏛️ Governance Architecture & Document Structure

The project rules, architecture specifications, and operational protocols are modularized into the following documents.

| Document | Focus & Scope | Description |
| :--- | :--- | :--- |
| 📘 **[AGENTS.md](AGENTS.md)** | **Constitution & Governance** | System vision, truth hierarchy, decision framework, and entry point. |
| 🏛️ **[ARCHITECTURE.md](ARCHITECTURE.md)** | **System Design & Layering Rules** | 4-stack cortical column architecture, Layer I–VI formulation, PonderNet early exit, PyTorch training & Rust candle/Metal server specs. |
| 🗺️ **[ROADMAP.md](ROADMAP.md)** | **Roadmap & Milestones** | v1.0 MVP to v2.0 roadmap, sprint breakdown, deliverables, and KPI targets. |
| 📋 **[PLANNING.md](PLANNING.md)** | **Planning & Discovery** | Implementation plans, architecture design, exploration protocols, and task breakdown. |
| 💻 **[CODING.md](CODING.md)** | **Coding Rules & Standards** | Hardening protocol for Rust (`candle`/`axum`) and Python (`PyTorch`/`MLX`), safety constraints, and memory rules. |
| 🛡️ **[AUDITING.md](AUDITING.md)** | **Security, Compliance & Audit** | Static auditing, `cargo-deny` license checks, secret protection (`betterleaks`), and linting suites. |
| 🧪 **[TESTING.md](TESTING.md)** | **Testing & Validation** | Numerical parity tests (PyTorch vs Rust), Metal inference benchmarks, MCP tool-use evaluation, and lifelong learning verification. |

---

## ⚖️ Hierarchy of Truth

When directives or specifications conflict, resolve in this order:

```
1. Verified Measurements (Metal execution benchmark, PyTorch-Rust tensor parity, memory profiling)
   └── 2. ARCHITECTURE.md and CODING.md (System design and hard safety rules)
        └── 3. ROADMAP.md and PLANNING.md
             └── 4. The remaining governance documents
```

**Measurement outranks documentation.**
A document that disagrees with a verified measurement (e.g. inference speed, memory footprint, or tensor numerical equivalence) is wrong and must be corrected, not argued from.

---

## 📚 Which Document Owns What

Each document answers one primary question to prevent conflicting specifications:

| Document | Answers | Does **not** contain |
| :--- | :--- | :--- |
| **[README.md](README.md)** | What is Cotier, how do I install and run it? | Internal module math or deep rules |
| **[AGENTS.md](AGENTS.md)** | How is the project governed? Where does everything live? | Specific implementation code |
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | What is the exact mathematical and architectural design? | Task history, sprint schedules |
| **[ROADMAP.md](ROADMAP.md)** | What is next, what are the sprint tasks and KPIs? | Mathematical formulas, coding rules |
| **[CODING.md](CODING.md)** | What standards and rules must the code satisfy? | High-level system architecture |
| **[AUDITING.md](AUDITING.md)** | How are compliance, licenses, and secrets checked? | The implementation code itself |
| **[TESTING.md](TESTING.md)** | What must be verified before merging or releasing? | Test execution logs |
| **[PLANNING.md](PLANNING.md)** | How are changes planned and explored? | Completed post-mortems |

---

## 🎯 Core Operating Principles

1. **Hardware Native & Efficiency First**:
   - Target Apple Silicon Unified Memory with zero-copy `mmap`, Metal shaders, and sub-1.5 GB memory footprints.
2. **Latent Reasoning Over Token Spillage**:
   - Internal recurrent loops ($k=1\dots6$) in Layer V rather than generating extensive textual `<think>` tokens.
3. **Strict Parity & Determinism**:
   - PyTorch-trained weights exported to `safetensors` must produce numerically identical outputs ($L_\infty < 10^{-4}$) in the Rust `candle` inference engine.
4. **Safety & Zero-Unsafe Rust**:
   - `unsafe_code = "forbid"` across the Rust codebase. Preemption locks ensure sleep-learning never blocks user inference.
5. **Continuous Verification**:
   - Every change must pass lints, license auditing, secret scanning, unit tests, and cross-platform benchmarks.

---

## 🚀 Quick Verification Commands

```bash
# Python Training Environment check
cd train && uv run ruff check . && uv run mypy src/

# Rust Server Compliance & Test Suite
cd server && cargo clippy --workspace --all-targets -- -D warnings
cd server && cargo test --workspace --features metal

# License Audit
cargo deny check licenses

# Secret Scan
betterleaks dir .
```
