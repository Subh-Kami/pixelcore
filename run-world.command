#!/bin/zsh
set -eu
cd "$(dirname "$0")"
export UV_CACHE_DIR="$PWD/.weightworld/uv-cache"
mode="${1:-reference}"
if [[ "$mode" == "model" ]]; then
  exec uv run --locked python -m qgpu.world_server serve --backend model --merged build/merged-2b
elif [[ "$mode" == "reference" ]]; then
  exec uv run --locked python -m qgpu.world_server serve --backend reference
else
  printf 'Usage: ./run-world.command [reference|model]\n' >&2
  exit 1
fi
