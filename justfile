model := "Qwen3.5-2B"
merged := "build/merged-2b"

# list the recipes
default:
    @just --list

# install the dependencies
setup:
    uv sync

# download Qwen3.5-2B and compile the GPU into it
build:
    uv run hf download Qwen/{{model}} --local-dir models/{{model}}
    uv run python -m qgpu.merge models/{{model}} {{merged}}

# chat with Qwen and its GPU (e.g. just chat --mode advanced)
chat *args:
    uv run qgpu {{merged}} {{args}}

# check the build: Qwen's text unchanged, kernels exact
check:
    uv run python scripts/merged_check.py models/{{model}} {{merged}}

# run the tests
test *args:
    uv run pytest {{args}}
