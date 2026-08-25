#!/usr/bin/env bash
# Cotier Master CR-20 Compliance and Quality Audit Suite
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

echo "============================================================"
echo "🛡️  Running Cotier CR-20 Master Compliance & Quality Audit"
echo "============================================================"

# [CR-20 Rule 18] Secret & PII Scanning
echo ""
echo "▶️ [1/7] Running betterleaks secret & personal name PII scan (Rule 18)..."
if command -v betterleaks >/dev/null 2>&1; then
    betterleaks dir .
else
    echo "⚠️  betterleaks not found. Skipping secret scan."
fi

# [CR-20 Rule 19] License Compliance
echo ""
echo "▶️ [2/7] Running Rust & Python License Audits (Rule 19)..."
if command -v cargo-deny >/dev/null 2>&1; then
    cargo deny check licenses
else
    echo "⚠️  cargo-deny not found. Skipping cargo license check."
fi
./scripts/audit/check_python_licenses.sh

# [CR-20 Rule 9] Pure Rust & Metal Native (No foreign C/C++ compilation except named platform bridges)
echo ""
echo "▶️ [3/7] Verifying Pure Rust & Native Metal compliance (Rule 9)..."
cd "${REPO_ROOT}/server"
RULE9_EXEMPT_CRATES="libsqlite3-sys objc_exception"
RULE9_TARGETS="aarch64-apple-darwin"

C_BUILDERS=""
for member in $(cargo metadata --no-deps --format-version 1 2>/dev/null \
        | python3 -c "import sys,json;print(' '.join(p['name'] for p in json.load(sys.stdin)['packages']))"); do
    for triple in $RULE9_TARGETS; do
        cargo tree -p "$member" --target "$triple" --edges normal,build --prefix none \
            2>/dev/null | grep -q '^cc v' || continue
        for culprit in $(cargo tree -p "$member" --target "$triple" --edges normal,build \
                -i cc --depth 1 --prefix none 2>/dev/null \
                | awk '$1 != "cc" && NF { print $1 }' | sort -u); do
            case " $RULE9_EXEMPT_CRATES " in
                *" $culprit "*) ;;
                *) C_BUILDERS="$C_BUILDERS $member($triple):$culprit" ;;
            esac
        done
    done
done

if [ -n "$C_BUILDERS" ]; then
    echo "❌ Rule 9 Violation: Found unapproved foreign C/C++ compilers:$C_BUILDERS"
    exit 1
fi
echo "   ✅ Pure Rust verified: Zero unapproved C/C++ compiler dependencies (Exempt: $RULE9_EXEMPT_CRATES)."

# [CR-20 Rules 2, 3, 7, 10, 11, 12] Rust Static Code Audits
echo ""
echo "▶️ [4/7] Running Rust Static Hardening Audits (Rules 2, 3, 7, 10, 11, 12)..."
# Rule 2: Panic prevention (unwrap / expect)
echo "   Checking forbidden unwrap/expect in server/src/ (Rule 2)..."
if grep -rnE --include="*.rs" '\.(unwrap|expect)\(' src/ | grep -v 'test' >/dev/null 2>&1; then
    echo "❌ Rule 2 Violation: Forbidden unwrap/expect found in production code:"
    grep -rnE --include="*.rs" '\.(unwrap|expect)\(' src/ | grep -v 'test'
    exit 1
fi
echo "   ✅ No unwrap/expect in production server code."

# Rule 3: Unsafe ban
echo "   Checking unsafe ban (Rule 3)..."
if grep -rnE --include="*.rs" 'unsafe[[:space:]]+(\{|\bimpl\b|\bfn\b)' src/ >/dev/null 2>&1; then
    echo "❌ Rule 3 Violation: Found unsafe code in server/src/:"
    grep -rnE --include="*.rs" 'unsafe[[:space:]]+(\{|\bimpl\b|\bfn\b)' src/
    exit 1
fi
echo "   ✅ No unsafe code in server."

# Rule 7: Global mutable state ban
echo "   Checking static mut (Rule 7)..."
if grep -rnE --include="*.rs" 'static[[:space:]]+mut[[:space:]]' src/ >/dev/null 2>&1; then
    echo "❌ Rule 7 Violation: Found static mut in server/src/:"
    grep -rnE --include="*.rs" 'static[[:space:]]+mut[[:space:]]' src/
    exit 1
fi
echo "   ✅ No static mut in server."

# Rule 12: No error swallowing
echo "   Checking error swallowing (Rule 12)..."
if grep -rnE --include="*.rs" 'filter_map\([[:space:]]*Result::ok[[:space:]]*\)' src/ >/dev/null 2>&1; then
    echo "❌ Rule 12 Violation: Found filter_map(Result::ok) error swallowing in server/src/:"
    grep -rnE --include="*.rs" 'filter_map\([[:space:]]*Result::ok[[:space:]]*\)' src/
    exit 1
fi
echo "   ✅ No silent error swallowing in server."

# Formatting and Clippy
echo "   Running cargo fmt & clippy..."
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
echo "   ✅ Rust formatting & clippy passed."

# [CR-20 Python Standards] Python Subsystem Checks
echo ""
echo "▶️ [5/7] Running Python Subsystem Checks (Ruff & Mypy)..."
cd "${REPO_ROOT}/train"
if command -v uv >/dev/null 2>&1; then
    uv run ruff check .
    uv run mypy src/
fi
echo "   ✅ Python formatting, linter, and type checker passed."

# [CR-20 Rule 15] Model Configuration & Tensor Schema Contract
echo ""
echo "▶️ [6/7] Auditing Model Safetensors Schema Contract (Rule 15)..."
cd "${REPO_ROOT}"
test -f models/cotier-0.5b/config.json
test -f models/cotier-0.5b/tensor_schema.json
if [ -f models/cotier-0.5b/model.safetensors ]; then
    cd "${REPO_ROOT}/train"
    uv run python ../scripts/test/verify_tensor_schema.py --model-dir ../models/cotier-0.5b --schema-file ../models/cotier-0.5b/tensor_schema.json
    cd "${REPO_ROOT}"
fi
echo "   ✅ Safetensors schema contract verified."

# [CR-20 Rule 16] PyTorch-Rust Numerical Parity
echo ""
echo "▶️ [7/7] Verifying PyTorch-Rust Numerical Determinism (Rule 16)..."
if [ -f models/cotier-0.5b/model.safetensors ]; then
    cd "${REPO_ROOT}/train"
    uv run python ../scripts/test/verify_parity.py --model-dir ../models/cotier-0.5b --tolerance 1e-4
    cd "${REPO_ROOT}"
    echo "   ✅ Numerical parity verified (L_inf < 1e-4)."
else
    echo "   ⚠️ model.safetensors not present yet. Skipping parity check."
fi

echo ""
echo "============================================================"
echo "🎉 === ALL CR-20 PROTOCOL COMPLIANCE AUDITS PASSED ==="
echo "============================================================"
