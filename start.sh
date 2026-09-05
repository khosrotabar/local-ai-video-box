#!/usr/bin/env bash
set -Eeuo pipefail

if tmux has-session -t ai-movie-api 2>/dev/null; then
    echo "AI Movie API is already running."
    exit 0
fi

tmux new-session \
    -d \
    -s ai-movie-api \
    "/opt/ai-movie/server/run.sh"

echo "AI Movie API started."