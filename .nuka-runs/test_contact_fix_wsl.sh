#!/bin/bash
# Run contact fix verification in WSL for unified environment
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Ensure we're using the WSL Python environment
export PYTHONPATH="$PROJECT_ROOT/build-win-editor/python:$PYTHONPATH"

echo "Testing contact fix in WSL environment..."
python3 .nuka-runs/verify_contact_fix.py
