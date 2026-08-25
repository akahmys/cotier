# 🛡️ Cotier Security, License & Compliance Auditing Protocol

This document defines the automated audit checks, license policy, security vulnerability management, and secret protection standards for the Cotier repository.

*(Note: Coding standards live in [`CODING.md`](CODING.md); tests, parity benchmarks, and E2E verifications live in [`TESTING.md`](TESTING.md).)*

---

## 🔍 1. Audit Framework & Static Rules

Cotier enforces an automated compliance pipeline covering both the Python training subsystem and the Rust inference engine:

```
                          ┌──────────────────────────┐
                          │   Compliance Pipeline    │
                          └────────────┬─────────────┘
                                       │
     ┌──────────────────┬──────────────┴──────────────┬──────────────────┐
     ▼                  ▼                             ▼                  ▼
1. Safety Audits    2. Type & Static Check     3. License Audit    4. Secret Scan
(Line limits,       (Clippy, ruff,             (cargo-deny via     (betterleaks via
 panic, unsafe)      mypy strict)               deny.toml)          pre-commit)
```

### Static Audit Rule Matrix

| Rule | Area | Requirement | Enforcement |
| :--- | :--- | :--- | :--- |
| **Rule A1** | **Secret & PII Protection** | No API keys, credentials, private keys, or personal data may be committed. | `betterleaks` via Git pre-commit & CI |
| **Rule A2** | **License Compliance** | All Rust and Python dependencies must belong to the allow-list in `deny.toml` (No GPL/AGPL copyleft). | `cargo deny` & `check_python_licenses.sh` |
| **Rule A3** | **Foreign C/C++ Ban** | No crate may pull `cc` or compile unapproved C/C++ libraries. | `cargo tree -i cc` in `verify_compliance.sh` |
| **Rule A4** | **Panic Prevention** | Production code must be free of `.unwrap()` and `.expect()`. | Grep AST check in `verify_compliance.sh` |
| **Rule A5** | **Zero Unsafe** | `unsafe` is forbidden across all Rust workspace members. | Rustc `#![forbid(unsafe_code)]` |
| **Rule A6** | **Global Mutable State** | `static mut` is strictly forbidden. | Grep check in `verify_compliance.sh` |
| **Rule A7** | **Zero Silent Swallowing**| `filter_map(Result::ok)` and silent error dropping are forbidden. | Grep check in `verify_compliance.sh` |
| **Rule A8** | **Formatting Standard** | Code must satisfy `cargo fmt --all --check` and `ruff check`. | Linter checks in `verify_compliance.sh` |

---

## 📜 2. License Compliance Protocol (`cargo-deny`)

All workspace crates and dependencies are continuously audited using **`cargo-deny`** against the project's license policy configured in `deny.toml`.

### Allowed License List (Permissive & Open Source)
- **Primary**: `MIT`, `Apache-2.0`, `Apache-2.0 WITH LLVM-exception`
- **BSD Family**: `BSD-3-Clause`, `BSD-2-Clause`, `0BSD`
- **Public Domain / Permissive**: `CC0-1.0`, `Unlicense`, `ISC`, `Zlib`, `MIT-0`, `MPL-2.0`

### Forbidden Licenses
- Strong copyleft licenses (e.g., `GPL-2.0`, `GPL-3.0`, `AGPL-3.0`) are strictly **denied** (`copyleft = "deny"`).

### License Audit Commands
```bash
# 1. Rust Server License Audit (via cargo-deny)
cargo deny check licenses

# 2. Python Dependencies License Audit (via pip-licenses & uv)
./scripts/audit/check_python_licenses.sh
```

---

## 🔐 3. Secret & PII Protection Protocol (`betterleaks`)

To prevent accidental leaks of Hugging Face tokens, OpenAI API keys, cloud credentials, and Personally Identifiable Information (PII):

### Git Pre-commit Hook
Security scanning is automatically enforced before every git commit via `scripts/hooks/pre-commit` using **`betterleaks`**.

### Custom Leak Prevention Rules (`.betterleaks.toml`)
- **Hugging Face / OpenAI / Anthropic API Keys**: Standard high-entropy and service token patterns.
- **Private Keys & SSH Credentials**: RSA, EC, OpenSSH private keys.
- **Model Weights in Git**: Guard against committing raw `.safetensors` or `.bin` files directly into git history.

### Secret Audit Commands
```bash
# Scan repository for secrets
betterleaks dir .

# Run pre-commit staged scan manually
betterleaks git --pre-commit --staged
```

---

## 🛠️ 4. Master Automated Audit Execution

To run the complete static audit suite:

```bash
bash scripts/audit/verify_compliance.sh
```
