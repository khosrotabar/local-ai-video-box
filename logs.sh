#!/usr/bin/env bash

echo "=== API LOG ==="

tmux capture-pane \
    -pt ai-movie-api \
    -S -100 \
    2>/dev/null || true

echo
echo "=== LATEST GENERATION LOG ==="

LATEST="$(
    ls -t \
    /opt/ai-movie/logs/api/*.log \
    2>/dev/null \
    | head -1
)"

if [[ -n "$LATEST" ]]; then
    echo "$LATEST"
    echo
    tail -n 100 "$LATEST"
else
    echo "No generation logs yet."
fi