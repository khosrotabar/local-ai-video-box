#!/usr/bin/env bash
set -Eeuo pipefail

tmux kill-session \
    -t ai-movie-api \
    2>/dev/null || true

echo "AI Movie API stopped."