#!/usr/bin/env bash
# sterilize-check.sh — scan for credentials and private paths before committing.
#
# Usage:
#   ./scripts/sterilize-check.sh          # scan staged files (pre-commit mode)
#   ./scripts/sterilize-check.sh --all    # scan all tracked files
#
# Exit 0 = clean. Exit 1 = violations found.

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ERRORS=0

fail() { echo -e "${RED}[FAIL]${NC} $*"; ERRORS=$((ERRORS + 1)); }
info() { echo -e "${YELLOW}[INFO]${NC} $*"; }

# ── Collect files to scan ────────────────────────────────────────────────────

if [ "${1:-}" = "--all" ]; then
    SCAN_FILES=$(git ls-files)
else
    SCAN_FILES=$(git diff --cached --name-only --diff-filter=ACM 2>/dev/null || true)
    if [ -z "$SCAN_FILES" ]; then
        echo "No staged files to scan."
        exit 0
    fi
fi

# Filter to text files — skip binaries, images, lock files, pickles
TEXT_FILES=""
for f in $SCAN_FILES; do
    [ -f "$f" ] || continue
    case "$f" in
        *.pkl|*.faiss|*.png|*.jpg|*.jpeg|*.gif|*.ico|*.whl|*.zip|*.tar.gz) continue ;;
        *.pyc|*.pyo) continue ;;
        uv.lock|poetry.lock|package-lock.json|yarn.lock) continue ;;
        scripts/sterilize-check.sh) continue ;;   # skip self — script contains the patterns
    esac
    TEXT_FILES="$TEXT_FILES $f"
done

[ -z "$TEXT_FILES" ] && { echo "No text files to scan."; exit 0; }
N=$(echo "$TEXT_FILES" | wc -w | tr -d ' ')
echo "Scanning $N file(s)..."

# ── 1. Credential patterns ────────────────────────────────────────────────────
# Real API keys — long alphanumeric strings, not placeholder sk-... or sk-<...>

for f in $TEXT_FILES; do
    # OpenAI / compatible keys: sk- followed by 20+ alphanumeric chars
    # Excludes placeholders: sk-..., sk-<key>, sk-YOUR_KEY
    if grep -qE 'sk-[a-zA-Z0-9]{20,}' "$f" 2>/dev/null; then
        fail "Possible API key in $f (pattern: sk-[a-zA-Z0-9]{20,})"
    fi

    # Anthropic keys
    if grep -qE 'sk-ant-api[0-9]+-[a-zA-Z0-9_-]{20,}' "$f" 2>/dev/null; then
        fail "Possible Anthropic key in $f"
    fi

    # Groq keys
    if grep -qE 'gsk_[a-zA-Z0-9]{30,}' "$f" 2>/dev/null; then
        fail "Possible Groq key in $f"
    fi

    # GitHub personal access tokens
    if grep -qE 'ghp_[a-zA-Z0-9]{36,}|github_pat_[a-zA-Z0-9_]{82,}' "$f" 2>/dev/null; then
        fail "Possible GitHub token in $f"
    fi
done

# ── 2. Private paths — absolute personal paths (all file types) ──────────────

ABSOLUTE_PRIVATE=(
    '/Users/selizondo'
    'Dropbox/projects/vscode'
    'vscode/career'
)

for pattern in "${ABSOLUTE_PRIVATE[@]}"; do
    for f in $TEXT_FILES; do
        if grep -qF "$pattern" "$f" 2>/dev/null; then
            fail "Private absolute path '$pattern' in $f"
        fi
    done
done

# ── 3. Internal names — only in docs/markdown (not source code) ──────────────
# These are internal project/venv names that shouldn't appear in user-facing docs.

DOC_FILES=""
for f in $TEXT_FILES; do
    case "$f" in *.md|*.rst|*.txt) DOC_FILES="$DOC_FILES $f" ;; esac
done

if [ -n "$DOC_FILES" ]; then
    INTERNAL_DOC_PATTERNS=(
        'newline_labs'       # internal monorepo folder name
        '\.venvs/newline'    # internal shared venv name
        'newline_stuff'      # old internal repo name
    )
    for pattern in "${INTERNAL_DOC_PATTERNS[@]}"; do
        for f in $DOC_FILES; do
            if grep -qE "$pattern" "$f" 2>/dev/null; then
                fail "Internal name '$pattern' in doc file $f"
            fi
        done
    done
fi

# ── Result ────────────────────────────────────────────────────────────────────

echo ""
if [ "$ERRORS" -gt 0 ]; then
    echo -e "${RED}✗ $ERRORS violation(s) found — commit blocked.${NC}"
    exit 1
else
    echo -e "${GREEN}✓ Clean — no credentials or private paths found.${NC}"
    exit 0
fi
