#!/usr/bin/env bash
# SessionStart hook — guarantee a session can run the test suite.
#
# Builds .venv with CPU-only torch plus requirements-dev.txt. Idempotent: once
# the venv imports torch and pytest this exits in well under a second. It never
# fails the session — an install failure prints the manual command and exits 0,
# because a session with no venv is still more useful than no session.
#
# Not gated on CLAUDE_CODE_REMOTE: the early-exit path is cheap enough that
# running it locally costs nothing.

set -uo pipefail

REPO="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
VENV="$REPO/.venv"
PY="$VENV/bin/python"
TORCH_CPU_INDEX="https://download.pytorch.org/whl/cpu"

# Put the venv on PATH for the rest of the session, whether or not we built it.
export_venv() {
  if [ -n "${CLAUDE_ENV_FILE:-}" ] && [ -x "$PY" ]; then
    {
      echo "export VIRTUAL_ENV=$VENV"
      echo "export PATH=$VENV/bin:\$PATH"
    } >> "$CLAUDE_ENV_FILE"
  fi
}

summary() {
  echo
  echo "Tests:        .venv/bin/pytest            # fast suite, no model download"
  echo "Slow tests:   .venv/bin/pytest -m slow    # downloads a HuggingFace model"
  echo "Config:       pyproject.toml [tool.pytest.ini_options]"
}

# --- fast path: already set up ------------------------------------------------
if [ -x "$PY" ] && "$PY" -c "import torch, pytest" >/dev/null 2>&1; then
  echo "session-start: .venv ready (torch + pytest import cleanly)."
  export_venv
  summary
  exit 0
fi

# --- build --------------------------------------------------------------------
if [ ! -x "$PY" ]; then
  echo "session-start: creating .venv"
  if ! python3 -m venv "$VENV"; then
    echo "session-start: python3 -m venv failed. Create the venv by hand:"
    echo "  python3 -m venv .venv"
    exit 0
  fi
fi

"$PY" -m pip install --quiet --upgrade pip setuptools wheel

failed=0

echo "session-start: installing CPU-only torch"
"$PY" -m pip install torch --index-url "$TORCH_CPU_INDEX" || failed=1

if [ -f "$REPO/requirements-dev.txt" ]; then
  echo "session-start: installing requirements-dev.txt"
  "$PY" -m pip install -r "$REPO/requirements-dev.txt" || failed=1
fi

if [ "$failed" -ne 0 ] || ! "$PY" -c "import torch, pytest" >/dev/null 2>&1; then
  echo
  echo "session-start: setup incomplete — the session will start anyway."
  echo "Retry by hand, and read the pip output for the reason:"
  echo "  python3 -m venv .venv"
  echo "  .venv/bin/pip install torch --index-url $TORCH_CPU_INDEX"
  echo "  .venv/bin/pip install -r requirements-dev.txt"
  echo "Tests that need torch will fail until this succeeds."
  exit 0
fi

echo "session-start: .venv ready."
export_venv
summary
exit 0
