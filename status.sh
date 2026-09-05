#!/usr/bin/env bash

echo "=== API ==="

if curl -fsS \
    http://127.0.0.1:11434/api/health \
    2>/dev/null
then
    echo
else
    echo "API offline"
fi

echo
echo "=== TMUX ==="

if tmux has-session -t ai-movie-api 2>/dev/null; then
    echo "ai-movie-api: running"
else
    echo "ai-movie-api: stopped"
fi

echo
echo "=== GPU ==="

nvidia-smi \
    --query-gpu=name,memory.used,memory.free,memory.total,utilization.gpu \
    --format=csv,noheader