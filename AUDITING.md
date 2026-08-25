# 🛡️ Cotier Security, License & Compliance Auditing Protocol

This document defines the automated audit checks, license policy, security vulnerability management, and secret protection standards for the Cotier repository.

---

## 🔍 1. Audit Framework Overview

Cotier enforces a multi-tier automated compliance pipeline covering both the Python training subsystem and the Rust inference engine:

```
                          ┌──────────────────────────┐
                          │   Compliance Pipeline    │
                          └────────────┬─────────────┘
                                       │
     ┌──────────────────┬──────────────┴──────────────┬──────────────────┐
     ▼                  ▼                             ▼                  ▼
1. Safety Rules    2. Lint & Type Check       3. License Audit    4. Secret Scan
(Line limits,      (Clippy, ruff,             (cargo-deny via     (betterleaks via
 panic, unsafe)     mypy strict)               deny.toml)          pre-commit)
```

---

## 📜 2. License Compliance Protocol (`cargo-deny`)

All workspace crates and dependencies are continuously audited using **`cargo-deny`** against the project's license policy configured in `deny.toml`.

### Allowed License List (Permissive & Open Source)
- **Primary**: `MIT`, `Apache-2.0`, `Apache-2.0 WITH LLVM-exception`
- **BSD Family**: `BSD-3-Clause`, `BSD-2-Clause`, `0BSD`
- **Public Domain / Permissive**: `CC0-1.0`, `Unlicense`, `ISC`, `Zlib`, `MIT-0`

### Forbidden Licenses
- Strong copyleft licenses (e.g., `GPL-2.0`, `GPL-3.0`, `AGPL-3.0`) are strictly **denied** (`copyleft = "deny"`).

### License Audit Commands
```bash
# 1. Rust Server License Audit (via cargo-deny)
cargo deny check licenses

# 2. Python Dependencies License Audit (via pip-licenses & uv)
./scripts/audit/check_python_licenses.sh
# または単体実行:
uvx pip-licenses --fail-on="GPL;AGPL;LGPL" --order=license
```

---

## 🔐 3. Secret & PII Protection Protocol (`betterleaks`)

To prevent accidental leaks of Hugging Face tokens, OpenAI API keys, cloud credentials, and Personally Identifiable Information (PII):

### Git Pre-commit Hook
Security scanning is automatically enforced before every git commit via `.git/hooks/pre-commit` using **`betterleaks`**.

### Custom Leak Prevention Rules (`.betterleaks.toml`)
- **Hugging Face / OpenAI / Anthropic API Keys**: Standard high-entropy and service token patterns.
- **Private Keys & SSH Credentials**: RSA, EC, OpenSSH private keys.
- **Model Weights in Git**: Guard against committing raw `.safetensors` or `.bin` files directly into git history (must be managed via storage / releases).

### Secret Audit Commands
```bash
# Scan repository for secrets
betterleaks dir .

# Run pre-commit staged scan manually
betterleaks git --pre-commit --staged
```

---

## 🛠️ 4. Static Compliance Checklist & Script

### Rust Server Audit Matrix
| Check | Rule / Tool | Description |
| :--- | :--- | :--- |
| **No Unsafe** | `unsafe_code = "forbid"` | Zero `unsafe` blocks across all Rust modules. |
| **No Panics** | Grep / Clippy | `unwrap()` and `expect()` are forbidden in production server paths. |
| **No Wildcards** | `clippy::wildcard_enum_match_arm` | Explicit matching on all domain enums (e.g. Layer types, Tool types). |
| **Pure Rust / Metal** | Cargo dependencies | No foreign C/C++ compilation except platform-native Metal/CoreML bindings. |
| **Clippy Lints** | `cargo clippy -D warnings` | Pedantic and nursery lints clean across workspace. |
| **Code Formatting** | `cargo fmt --check` | Standard Rustfmt style. |

### Python Training Subsystem Audit Matrix
| Check | Tool | Description |
| :--- | :--- | :--- |
| **Code Formatting** | `ruff format --check` | Black-compatible formatting enforced. |
| **Linting** | `ruff check .` | High-standard Python linting with zero warnings. |
| **Type Checking** | `mypy --strict src/` | Full static typing on model definitions, loss functions, and data pipelines. |
| **Shape Assertions** | Torch assertions | Tensor shapes explicitly verified on layer boundaries ($B, L, D$). |
