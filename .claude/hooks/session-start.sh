#!/bin/bash
# SessionStart hook: prepare the JaxMARL dev environment for Claude Code on the web.
#
# Installs jaxmarl with its dev + algs extras into a project-local virtualenv so
# that tests and tooling work out of the box. A venv is used because the system
# Python ships a Debian-patched setuptools that fails to build some source-only
# dependencies (e.g. antlr4-python3-runtime, pulled in via hydra-core/omegaconf).
set -euo pipefail

# Only run in the remote (Claude Code on the web) environment.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "$CLAUDE_PROJECT_DIR"

VENV_DIR="$CLAUDE_PROJECT_DIR/.venv"

# Create the virtualenv if it doesn't already exist (idempotent).
if [ ! -x "$VENV_DIR/bin/python" ]; then
  python -m venv "$VENV_DIR"
fi

# Modern pip/setuptools/wheel avoid the patched-setuptools build failure.
"$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel

# Install jaxmarl (editable) with the extras the CI uses.
"$VENV_DIR/bin/pip" install -e ".[dev,algs]"

# Put the venv on PATH for the rest of the session so `python`/`pytest`
# resolve to the installed environment.
echo "export PATH=\"$VENV_DIR/bin:\$PATH\"" >> "$CLAUDE_ENV_FILE"
echo "export VIRTUAL_ENV=\"$VENV_DIR\"" >> "$CLAUDE_ENV_FILE"
