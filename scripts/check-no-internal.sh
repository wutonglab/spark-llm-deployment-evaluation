#!/usr/bin/env bash
# Sanitization lint: fails (non-zero exit) if any internal hostname / IP /
# customer business reference / credential pattern appears in the repository.
#
# Run locally before PR:
#   bash scripts/check-no-internal.sh
#
# CI runs this on every push and PR (see .github/workflows/check-no-internal.yml).
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Patterns that are NEVER allowed in this repo.
# Each entry: "pattern|description". Use Perl-compatible regex.
PATTERNS=(
    '\bspark215\b|internal hostname spark215'
    '\bspark-4d6b\b|internal hostname spark-4d6b'
    '\bjetson223\b|internal hostname jetson223'
    '\bagent213\b|internal hostname agent213'
    '\bcursor9\b|internal hostname cursor9'
    '\bbig192\b|internal hostname big192'
    '\bsmall69\b|internal hostname small69'
    '\bclaw90\b|internal hostname claw90'
    '\bali16\b|internal hostname ali16'
    '\bus150\b|internal hostname us150'
    '\b172\.21\.[0-9]+\.[0-9]+\b|RFC1918 IP from internal lab'
    '\b10\.19\.[0-9]+\.[0-9]+\b|RFC1918 IP from internal lab'
    '\bnvidia\.com/email|likely internal email reference'
    '/home/nvidia/|hardcoded internal home path'
    '\bpanpan_beijing[a-z_0-9]*|internal account name'
    '\bwxid_[a-z0-9]+|internal account name'
    'hf_[A-Za-z0-9]{20,}|leaked HuggingFace token'
    'sk-(proj-|ant-)?[A-Za-z0-9_-]{30,}|leaked OpenAI/Anthropic API key'
    '-----BEGIN [A-Z ]+PRIVATE KEY-----|leaked private key'
    'AKIA[0-9A-Z]{16}|leaked AWS access key'
    '\b节能 45\.[0-9]+%|internal business metric (energy saving)'
    '\b波束|internal business term'
    '\bRAN [aA]gent\b|internal customer agent name'
    '\bIntent 稳态|internal business KPI'
)

# Files/dirs to skip from scanning (themselves contain these patterns by design)
EXCLUDES=(
    ".git"
    "node_modules"
    ".venv"
    "venv"
    "site"
    "evaluation-runs"
    "scripts/check-no-internal.sh"
    "__pycache__"
    ".pytest_cache"
    ".ruff_cache"
)

# Build grep --exclude args
EXCLUDE_ARGS=()
for e in "${EXCLUDES[@]}"; do
    EXCLUDE_ARGS+=(--exclude-dir="$e")
done
# Exclude this script itself from grep (it contains the patterns by definition)
EXCLUDE_ARGS+=(--exclude="check-no-internal.sh")

HITS=0
for entry in "${PATTERNS[@]}"; do
    pattern="${entry%%|*}"
    description="${entry#*|}"
    matches=$(grep -rEHn "${EXCLUDE_ARGS[@]}" -e "$pattern" . 2>/dev/null || true)
    if [[ -n "$matches" ]]; then
        echo "❌ FOUND ($description):"
        echo "$matches" | sed 's/^/    /'
        echo
        HITS=$((HITS + 1))
    fi
done

if [[ $HITS -gt 0 ]]; then
    echo "FAIL: $HITS sanitization pattern(s) hit. Remove or redact before committing."
    exit 1
fi

echo "✅ Sanitization clean: 0 patterns hit"
exit 0
