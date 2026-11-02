# PixelCore / WeightPong

A small game engine compiled into eight transformer layers and merged into
Qwen3.5-2B. The model emits its new game state and every pixel. This project uses
Qwenputer's compiler and adapts qwenGPU's model-merge conventions; it does not claim
to have invented transformer compilation, recurrent memory, or in-model games.

## What works

- An 8×8 paddle-and-ball game: movement, side/ceiling reflections, paddle contact
  steering, scoring, loss, a three-save win, terminal-state freezing, and reset.
- Four input tokens: left, right, stay, reset. An action produces 18 generated
  tokens: one state word, 16 packets of four two-bit pixels, and a frame-end token.
- A 14-bit state in the first compiled layer's Gated DeltaNet memory. Only the
  initial state is host-supplied. Subsequent states are model-generated tokens.
- Full merged-Qwen inference and a separately labeled standalone circuit model.
- A playable local viewer, manual controls, optional external paddle assistance,
  trace export, independent lock checking, and explicitly labeled 4× replay.

The initial frame emits 17 tokens because the initial state is an input. The
standalone model has eight game layers plus a zero full-attention helper layer
that lets the stock Transformers cache count positions. It contains no
pretrained Qwen language backbone.

## Measured on a 24 GB Apple Silicon Mac

| Check | Result |
| --- | --- |
| Exact circuit evaluator | All 1,064 transitions from 266 reachable states matched |
| Persistent circuit sequence | 100 ticks without host state re-upload matched |
| Full Qwen game | 29 ticks, three saves, 539/539 generated tokens matched |
| Recurrent state | Checked against each model-emitted state word |
| Small text regression check | 40/40 top-token agreement; mean KL 5.49e-4; no game-token leakage |
| Full-Qwen tick time | Mean 1.862 s; median 1.842 s over the recorded 29 ticks |
| Standalone tick time | Mean 0.165 s; median 0.152 s over the same controller's game |
| Full merged checkpoint | 5,805,412,992 bytes of tensor data, three shards |
| Standalone checkpoint | About 54 MiB before compression |
| Browser checks | Model input, held movement, pause, loss, reset, export; no console errors observed |

These are prototype measurements, not general model-quality or hardware
benchmarks. The earlier WeightWorld particle experiment uses a different
workload; its 489 tokens/frame are not a controlled speed comparison with this
game. The standalone model's speed is not the full Qwen model's speed.

## Run

```sh
uv sync --locked

# Emit the compact standalone transformer weights:
uv run python -m qgpu.pixelcore.merge build/pixelcore-tiny
uv run python -m qgpu.pixelcore.server --checkpoint build/pixelcore-tiny --port 8767

# Build the actual merged Qwen model:
uv run hf download Qwen/Qwen3.5-2B --revision 15852e8c16360a2fea060d615a32b45270f8a8fc --local-dir models/Qwen3.5-2B
uv run python -m qgpu.pixelcore.merge build/pixelcore-2b --base models/Qwen3.5-2B
uv run python -m qgpu.pixelcore.server --checkpoint build/pixelcore-2b --port 8766
```

Open `http://127.0.0.1:8766` for full Qwen or `http://127.0.0.1:8767` for the
standalone circuit. The UI labels the execution source. The model never silently
falls back to the reference interpreter. On macOS, `./run-pixelcore.command model`
and `./run-pixelcore.command core` launch the corresponding viewer.

The reference preview is available by omitting `--checkpoint`. It is explicitly
not model inference. Stock PyTorch/Transformers are sufficient on MPS; this
project does not require CUDA-only optional acceleration libraries on the Mac.

Hold left/right while the game is running, or use the on-screen buttons. Space
pauses. New game sends the compiled reset action rather than computing a new
world on the host. Pausing finishes the tick already in progress. Paddle assist
selects buttons from the last model-emitted ball position; it is an external
controller, not learned model strategy. Three saves wins; missing ends the game.

## Validate and record

```sh
uv run pytest tests/test_pixelcore.py -q
uv run python scripts/validate_pixelcore.py build/pixelcore-2b --device mps --output recordings/pixelcore-qwen.json
uv run python scripts/validate_pixelcore.py build/pixelcore-tiny --device mps --output recordings/pixelcore-standalone.json
uv run python scripts/check_pixelcore_text.py models/Qwen3.5-2B build/pixelcore-2b --output recordings/text-check.json
```

To verify an exported trace independently:

```python
import json
from qgpu.pixelcore.runtime import verify
print(verify(json.load(open("recordings/pixelcore-qwen.json"))))
```

Actual model mode greedily decodes its own logits and reads its recurrent head.
The host runs a separate Python specification as a checker. Any token or state
mismatch stops execution; reference values never replace a model result. Pixel
packets are unpacked into colors for display, with no host collision or
rasterization code supplying model-mode frames.

A trace's backend label is not cryptographic attestation that inference happened.
The included run logs and witnessed local runs establish the provenance of the
provided recordings. Numerical circuit checks and sampled bf16 runs are not a
formal proof of all implementation behavior.

The optional clip renderer needs Pillow and ffmpeg:

```sh
uv run --with pillow python scripts/render_pixelcore_clip.py recordings/pixelcore-qwen.json PixelCore-demo.mp4
```

The clip is explicitly a 4× replay of recorded full-Qwen pixel outputs. Its input
policy is external paddle assistance. No motion interpolation is added. The
font paths in this helper target macOS; change them on another OS.

## Files and ownership

The new game-specific contribution is in:

- `qgpu/pixelcore/circuit.py`: arithmetic/Boolean game logic, recurrent state,
  pixel packing and next-token output code, compiled into actual layer weights.
- `qgpu/pixelcore/spec.py`: independent reference rules and token format.
- `qgpu/pixelcore/merge.py`: game-specific checkpoint/tokenizer construction.
- `qgpu/pixelcore/runtime.py`: model inference, token checking, recurrent-head checking.
- `qgpu/pixelcore/server.py`, `qgpu/pixelcore/ui/`: local interactive app.
- `tests/test_pixelcore.py`, `scripts/validate_pixelcore.py`: finite-state coverage,
  persistent sequences, runtime failure guards, and actual-model game recordings.

[Qwenputer](https://github.com/divagr18/qwenputer) supplies the MIT-licensed
transformer compiler and Gated DeltaNet memory primitives. The merge approach
adapts [qwenGPU](https://github.com/vixhal-baraiya/qwenGPU), starting from commit
`e2695f6a0a818ebc68e2e14b6b990eddf04c51b3`. The compiler dependency is pinned in
`pyproject.toml` and `uv.lock`. Qwen3.5 supplies the pretrained backbone and its
weight license. Preserve those credits and the original MIT license.

See `UPSTREAM_README.md` for the original qwenGPU documentation and
`WEIGHTWORLD.md` for the earlier resident CUDA particle prototype. PixelCore is
not a general CUDA GPU: it is deliberately a much narrower, game-specific core.

## Further documentation

See [design](docs/design.md), [research](docs/research.md) and [results](docs/results.md).
