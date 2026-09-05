#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$SCRIPT_DIR/stop.sh"

sleep 1

"$SCRIPT_DIR/start.sh"

sleep 2

"$SCRIPT_DIR/status.sh"