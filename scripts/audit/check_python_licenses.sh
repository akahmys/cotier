#!/usr/bin/env bash
# Python license compliance audit script using pip-licenses & uv
set -euo pipefail

echo "🔍 Checking Python dependencies license compliance..."

# Permitted licenses list for MIT compatibility
ALLOWED_LICENSES=(
    "MIT License"
    "MIT"
    "Apache Software License"
    "Apache 2.0"
    "Apache-2.0"
    "BSD License"
    "BSD"
    "BSD-3-Clause"
    "BSD-2-Clause"
    "Python Software Foundation License"
    "PSF License"
    "ISC License (ISCL)"
    "ISC"
    "CC0 1.0 Universal (CC0 1.0) Public Domain Dedication"
    "Unlicense"
    "Mozilla Public License 2.0 (MPL 2.0)"
)

# Forbidden copyleft patterns
FORBIDDEN_PATTERNS="GPL|AGPL|LGPL|Copyleft"

if ! command -v uv >/dev/null 2>&1; then
    echo "⚠️  uv is not installed. Please install uv (https://astral.sh/uv)."
    exit 1
fi

echo "📋 Generating dependency license summary..."
(cd train && uv run pip-licenses --order=license --format=plain-vertical)

echo ""
echo "⚖️  Verifying copyleft violations (Forbidden: ${FORBIDDEN_PATTERNS})..."
# Check for copyleft licenses
if (cd train && uv run pip-licenses --format=json) | grep -iE "${FORBIDDEN_PATTERNS}" > /dev/null 2>&1; then
    echo "❌ License Violation: Copyleft or restricted licenses detected in Python dependencies!"
    (cd train && uv run pip-licenses --format=plain-vertical) | grep -iE "${FORBIDDEN_PATTERNS}" || true
    exit 1
fi

echo "✅ Python license compliance check passed cleanly."
