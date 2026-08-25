# 📋 Cotier Planning & Discovery Protocol

This document governs task planning, codebase exploration, feature design, and decision-making workflows within the Cotier project.

---

## 🎯 1. Planning Workflow

Before introducing structural changes to the cortical architecture, training objectives, or the Rust inference engine, formalize the plan. For architectural decisions with trade-offs (e.g. KV Cache layout, Metal shader optimizations, LoRA rank configurations), document rationale and empirical measurements.

### Implementation Plan Structure
1. **Goal Description**: Clear scope, rationale, target outcomes, and target latency/memory impact.
2. **Governance & Hardening Compliance**:
   - Explicit check against [`CODING.md`](CODING.md) (**CR-15 Protocol**): line limits, panic ban, pure Rust, zero-allocation.
   - Explicit check against [`AUDITING.md`](AUDITING.md) (**Rules A1〜A8**): secrets, licenses, static checks.
   - Explicit check against [`TESTING.md`](TESTING.md) (**Rules T1〜T6**): tensor parity, test separation, schema contract.
3. **User Review Required**: Breaking schema changes, API endpoint modifications, or hyperparameter updates.
4. **Open Questions**: Unresolved requirements, Metal API limitations, or dataset nuances.
5. **Proposed Changes**: Grouped logically by subsystem (`train/`, `server/`, `models/`, `data/`) with `[NEW]`, `[MODIFY]`, or `[DELETE]` annotations.
6. **Verification Plan**:
   - Master compliance audit (`bash scripts/audit/verify_compliance.sh`).
   - PyTorch forward test vs Rust candle tensor parity check ($L_\infty < 10^{-4}$).
   - Metal throughput benchmark (`tokens/sec`).
   - Memory footprint validation (< 1.5 GB).
   - E2E integration test suite (`test_e2e_integration.py`).

---

## 🔍 2. Codebase Discovery Protocol

Never guess implementation details, tensor dimensions, or configuration schemas. Follow this exploration protocol:

1. **Measure, do not assume**:
   - Verify tensor shapes, Metal memory allocations, and execution times by running focused probes/benchmarks rather than estimating.
2. **Safetensors Key & Dimension Verification**:
   - Before modifying model loading in Rust, inspect actual tensor names and shapes in `model.safetensors` using `safetensors` inspection tools.
3. **Complete Type & Struct Inspection**:
   - Inspect full PyTorch `nn.Module` and Rust `struct` definitions, especially parameter fields and lifecycle locks (`ArcSwap`, `RwLock`).
4. **Dependency & Environment Audit**:
   - Check `Cargo.toml` for feature flags (specifically `metal`), and `train/pyproject.toml` for PyTorch/Hugging Face version constraints.

---

## 🔄 3. Task & Session Tracking

- Task states and progress are tracked sequentially against the milestones defined in [`ROADMAP.md`](ROADMAP.md).
- After completing work or sprints, document:
  - Summary of implemented components.
  - Verification results (benchmark numbers, numerical parity deltas, audit status).
