#!/bin/zsh
set -eu
cd "$(dirname "$0")"
export UV_CACHE_DIR="$PWD/.weightworld/uv-cache"
mode="${1:-model}"
if [[ "$mode" == "model" ]]; then
  checkpoint="build/pixelcore-2b"
  port=8766
elif [[ "$mode" == "core" ]]; then
  checkpoint="build/pixelcore-tiny"
  port=8767
else
  printf 'Usage: ./run-pixelcore.command [model|core]\n' >&2
  exit 1
fi
if [[ ! -f "$checkpoint/pixelcore.json" ]]; then
  printf 'Build %s first; see PIXELCORE.md.\n' "$checkpoint" >&2
  exit 1
fi
printf 'Open http://127.0.0.1:%s\n' "$port"
exec uv run --locked python -m qgpu.pixelcore.server --checkpoint "$checkpoint" --port "$port"
