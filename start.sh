#!/usr/bin/env bash
# RagMate Unified Linux/macOS Launcher
set -e
cd "$(dirname "$0")"

echo "=========================================="
echo "Starting RagMate Unified Launcher..."
echo "=========================================="

if [ -f "backend/.venv/bin/python" ]; then
    backend/.venv/bin/python run.py "$@"
else
    python3 run.py "$@"
fi
