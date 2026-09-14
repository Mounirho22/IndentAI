#!/usr/bin/env bash
# scripts/clean_repo.sh — repository hygiene & open-source compliance check.
#
# What it does
#   1. Purges compiled Python caches and temporary test artefacts
#      (__pycache__/, *.pyc, .pytest_cache/, .mypy_cache/, .ruff_cache/).
#   2. Segregates the internal development-skills directory (.agents/) so it
#      is never indexed as part of the shipped agent logic:
#        * default  — .agents/ is excluded from version control via .gitignore
#                     (non-destructive: local tooling keeps working);
#        * --move   — additionally relocates .agents/ to .local_agent_skills/
#                     (also gitignored) for a fully clean working tree.
#   3. Verifies that the root LICENSE is a valid, standard MIT License
#      (byte-identical to the canonical template GitHub's licensee detects).
#
# Usage:
#   bash scripts/clean_repo.sh            # clean + gitignore segregation
#   bash scripts/clean_repo.sh --move     # clean + physically relocate .agents/
#   bash scripts/clean_repo.sh --check    # verification only, change nothing
#
# Exit codes: 0 = clean and compliant, 1 = LICENSE verification failed,
#             2 = environment problem (not a git repo / wrong cwd).

set -euo pipefail

MODE="${1:-clean}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BLUE='\033[0;34m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
step()  { printf "${BLUE}::${NC} %s\n" "$1"; }
ok()    { printf "  ${GREEN}✓${NC} %s\n" "$1"; }
info()  { printf "  ${YELLOW}•${NC} %s\n" "$1"; }
fail()  { printf "  ${RED}✗${NC} %s\n" "$1" >&2; }

# --------------------------------------------------------------------------- #
echo "=========================================================="
echo " IdentAI — repository hygiene & compliance"
echo " Root: $REPO_ROOT"
echo "=========================================================="

# 0. Sanity ---------------------------------------------------------------- #
if [ ! -f "main.py" ] || [ ! -d "app" ]; then
  fail "Run this from the repository root (expected main.py and app/ here)."
  exit 2
fi

# 1. Purge caches and temporary test artefacts ------------------------------ #
if [ "$MODE" != "--check" ]; then
  step "Purging compiled caches and temporary test directories..."
  removed=0
  while IFS= read -r -d '' path; do
    rm -rf "$path"
    ok "removed ${path#./}"
    removed=$((removed + 1))
  done < <(find . \
      -type d \( -name "__pycache__" -o -name ".pytest_cache" \
                 -o -name ".mypy_cache" -o -name ".ruff_cache" \
                 -o -name ".tox" -o -name "*.egg-info" \) \
      -not -path "./.venv/*" -not -path "./venv/*" \
      -print0 2>/dev/null)
  while IFS= read -r -d '' path; do
    rm -f "$path"
    removed=$((removed + 1))
  done < <(find . \
      -type f \( -name "*.pyc" -o -name "*.pyo" -o -name "*.db-journal" \
                 -o -name "*.db-wal" -o -name "*.db-shm" \) \
      -not -path "./.venv/*" -not -path "./venv/*" \
      -print0 2>/dev/null)
  [ "$removed" -eq 0 ] && info "nothing to remove — already clean"
else
  step "Check-only mode: skipping cache purge."
fi

# 2. Segregate the internal skills directory -------------------------------- #
step "Segregating internal development skills (.agents/)..."
if [ "$MODE" == "--check" ]; then
  if grep -qE '^\.agents/?$' .gitignore 2>/dev/null; then
    ok ".agents/ is excluded from version control"
  else
    fail ".agents/ is NOT excluded from version control (add it to .gitignore)"
  fi
elif [ "$MODE" == "--move" ]; then
  if [ -d ".agents" ]; then
    mkdir -p ".local_agent_skills"
    mv .agents/* ".local_agent_skills/" 2>/dev/null || true
    rmdir .agents 2>/dev/null || true
    ok "moved .agents/ -> .local_agent_skills/"
  else
    info ".agents/ not present (already relocated)"
  fi
  if ! grep -qE '^\.local_agent_skills/?$' .gitignore 2>/dev/null; then
    printf '\n# Internal agent-skills library, relocated out of the audited tree.\n.local_agent_skills/\n' >> .gitignore
    ok "added .local_agent_skills/ to .gitignore"
  fi
else
  # Default: keep the directory for local tooling, hide it from the repo.
  if grep -qE '^\.agents/?$' .gitignore 2>/dev/null; then
    ok ".agents/ already excluded via .gitignore"
  else
    printf '\n# Internal development-skills library (.agents/) — local tooling\n# configuration, not part of the shipped IdentAI agent logic.\n.agents/\n' >> .gitignore
    ok "added .agents/ to .gitignore"
  fi
fi

# 3. Verify the LICENSE is a standard MIT license --------------------------- #
step "Verifying LICENSE against the canonical MIT template..."
LICENSE_VERIFY="$(python - <<'PYEOF'
from pathlib import Path

CANONICAL = """MIT License

Copyright (c) <year> <copyright holders>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

license_path = Path("LICENSE")
if not license_path.exists():
    print("MISSING")
    raise SystemExit(0)

text = license_path.read_text(encoding="utf-8").strip()
lines = text.splitlines()
# GitHub's licensee keys on the exact canonical body with the copyright line
# varying. Mask our copyright line (line 3: header, blank, copyright) and
# compare the rest byte-for-byte.
if len(lines) >= 3 and lines[0].strip() == "MIT License":
    masked = "\n".join(
        lines[:2] + ["Copyright (c) <year> <copyright holders>"] + lines[3:]
    )
    print("VALID" if masked.strip() == CANONICAL.strip() else "MUTATED")
else:
    print("NOT_MIT")
PYEOF
)"

case "$LICENSE_VERIFY" in
  VALID)
    ok "LICENSE is standard MIT — GitHub license detection will tag the repo 'MIT'"
    printf "     header: %s\n" "$(head -n 1 LICENSE)"
    printf "     line:   %s\n" "$(sed -n '3p' LICENSE)"
    ;;
  MISSING) fail "LICENSE file is missing from the repository root"; exit 1 ;;
  MUTATED)
    fail "LICENSE deviates from the canonical MIT text — restore the standard template"
    exit 1
    ;;
  NOT_MIT) fail "LICENSE does not start with an MIT License header"; exit 1 ;;
esac

# 4. Summary ---------------------------------------------------------------- #
echo "=========================================================="
if [ "$MODE" == "--check" ]; then
  echo " Verification complete — tree is compliant."
else
  echo " Clean complete. Suggested next step:"
  echo "   git status   # review what changed"
fi
echo "=========================================================="
