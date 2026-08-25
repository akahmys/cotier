# 📋 Cotier Planning & Discovery Protocol

This document governs task planning, codebase exploration, feature design, and decision-making workflows within the Cotier project.

---

## 🎯 1. Planning Workflow

Before introducing structural changes to the cortical architecture, training objectives, or the Rust inference engine, formalize the plan. For architectural decisions with trade-offs (e.g. KV Cache layout, Metal shader optimizations, LoRA rank configurations), document rationale and empirical measurements.

### Implementation Plan Structure
1. **Goal Description**: Clear scope, rationale, target outcomes, and target latency/memory impact.
2. **User Review Required**: Breaking schema changes, API endpoint modifications, or hyperparameter updates.
3. **Open Questions**: Unresolved requirements, Metal API limitations, or dataset nuances.
4. **Proposed Changes**: Grouped logically by subsystem (`train/`, `server/`, `models/`, `data/`) with `[NEW]`, `[MODIFY]`, or `[DELETE]` annotations.
5. **Verification Plan**:
   - PyTorch forward test vs Rust candle tensor parity check.
   - Metal throughput benchmark (`tokens/sec`).
   - Memory footprint validation (< 1.5 GB).
   - Tool-calling JSON schema accuracy on Glaive eval test.

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
