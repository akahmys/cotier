#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "🔧 Installing Git hooks for Cotier..."
mkdir -p "${REPO_ROOT}/.git/hooks"
cp "${REPO_ROOT}/scripts/hooks/pre-commit" "${REPO_ROOT}/.git/hooks/pre-commit"
chmod +x "${REPO_ROOT}/.git/hooks/pre-commit"

echo "✅ Git pre-commit hook installed successfully!"
