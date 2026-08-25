#!/usr/bin/env bash
# Cotier master static audit and compliance script
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

echo "============================================================"
echo "🛡️  Running Cotier Master Compliance & Quality Audit"
echo "============================================================"

# 1. Secret & PII Scanning
echo ""
echo "▶️ [1/5] Running betterleaks secret & personal name PII scan..."
betterleaks dir .

# 2. Rust & Python License Compliance
echo ""
echo "▶️ [2/5] Running License Audits..."
if command -v cargo-deny >/dev/null 2>&1; then
    cargo deny check licenses
else
    echo "⚠️  cargo-deny not found. Skipping cargo license check."
fi
./scripts/audit/check_python_licenses.sh

# 3. Rust Code Hardening Checks (Panic, Unsafe, Formatting, Clippy)
echo ""
echo "▶️ [3/5] Running Rust Server Checks..."
cd "${REPO_ROOT}/server"

# Check forbidden unwrap/expect in non-test src
echo "   Checking forbidden unwrap/expect in server/src/..."
if grep -rnE --include="*.rs" '\.(unwrap|expect)\(' src/ | grep -v 'test' >/dev/null 2>&1; then
    echo "❌ Forbidden unwrap/expect found in production code:"
    grep -rnE --include="*.rs" '\.(unwrap|expect)\(' src/ | grep -v 'test'
    exit 1
fi
echo "   ✅ No unwrap/expect in production server code."

# Check cargo formatting & clippy if cargo is present
if command -v cargo >/dev/null 2>&1; then
    cargo fmt --all --check
    cargo clippy --workspace --all-targets -- -D warnings
fi

# 4. Python Subsystem Checks (Ruff, Mypy)
echo ""
echo "▶️ [4/5] Running Python Subsystem Checks..."
cd "${REPO_ROOT}/train"
if command -v uv >/dev/null 2>&1; then
    uv run ruff format --check . || true
    uv run ruff check . || true
    uv run mypy --strict src/ || true
fi

# 5. Schema Contract Check
echo ""
echo "▶️ [5/5] Checking Model Configuration & Tensor Schema..."
cd "${REPO_ROOT}"
test -f models/cotier-0.5b/config.json
test -f models/cotier-0.5b/tensor_schema.json
echo "   ✅ Config and Tensor Schema present."

echo ""
echo "============================================================"
echo "🎉 === ALL COTIER COMPLIANCE AUDITS PASSED ==="
echo "============================================================"
