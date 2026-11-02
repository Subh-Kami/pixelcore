# WeightWorld — resident simulation and rendering on qwenGPU

WeightWorld extends [qwenGPU](https://github.com/vixhal-baraiya/qwenGPU) with a small
persistent world, a CUDA software rasterizer, and a visual execution/replay tool.
The experiment asks whether a model-embedded GPU can keep simulation state across
kernel launches without the host reading results and re-uploading them as inputs.

This is an engineering extension, not a new pretrained model or a claim to have
invented qwenGPU's embedded GPU. Worldwide novelty has not been established.
In particular, [Qwenputer already uses recurrent state as RAM](https://github.com/divagr18/qwenputer#gated-deltanet-as-ram)
and describes persistent filesystem storage. Persistence or a screen alone is
not our invention; the specific contribution here is a resident CUDA simulation
pipeline, framebuffer workload, and its inspection/verification infrastructure.

## What this adds

- An explicit resident-memory launch protocol: initialize once; upload only
  ROM/LIVE/EXEC records on subsequent launches. Original `Session._launch` uploads
  launch input buffers again and clears memory between launches.
- Three CUDA kernels: clear the framebuffer, advance particles with wall
  reflections, and scatter particle color IDs into an 8×8 pixel buffer.
- An execution backend using a real merged Qwen checkpoint. Execution tokens
  come from its logits; results come from its recurrent state. A mismatch stops
  execution. There is no interpreter fallback in model mode.
- A separately labeled reference backend for development without model weights.
- A loopback-only viewer, live memory/warp/block inspection, deterministic seeds,
  trace export, checked replay, and a compact focus view for recordings.
- Independent physics/pixel tests and a six-launch compiled-circuit residency test.

## Ownership and attribution

The embedded SIMT architecture, transformer circuit, compiler integration,
PTX lowering, merge, base runtime, and token format come from qwenGPU and
[Qwenputer](https://github.com/divagr18/qwenputer). This checkout starts from
qwenGPU commit `e2695f6a0a818ebc68e2e14b6b990eddf04c51b3`.

The WeightWorld contribution is in `qgpu/world.py`, `qgpu/world_server.py`,
`qgpu/world_ui/`, `kernels/world_*`, `tests/test_world.py`, and the reproduction
script. Existing upstream files are retained with their MIT license. Qwen's
weights retain their own license. See the upstream README and LICENSE.

## Memory layout

| Word addresses | Contents |
| --- | --- |
| 0–7 | Four particles' x/y positions, interleaved |
| 8–15 | Their signed x/y velocities |
| 16–79 | 64 framebuffer pixels; 0 background, color IDs 1–4 |

Positions are integers in `[0,127]`; each display pixel covers 16 position units.
The seed selects velocities from `{-7,-5,5,7}`. The kernels preserve speed and
reflect at the walls. There are no particle-to-particle collisions. On overlaps,
the highest particle ID wins in the circuit's deterministic lane order.

Every frame runs clear (64 threads), advance (8 threads), then draw (4 threads).
The model keeps one cache across all launches. Programs are still host-uploaded
on each launch; this is **resident world data**, not resident program ROM or a
fully autonomous model loop. The host schedules launches. No weights are trained.

## Run locally

Requirements: Python 3.11+, uv, and the upstream pinned dependencies.
The checked-in PTX lets the viewer run without recompiling CUDA.

```sh
uv sync --locked
uv run python -m qgpu.world_server serve --backend reference
```

Open `http://127.0.0.1:8765`. This explicitly runs the CPU interpreter, not Qwen.
`./run-world.command` launches the same reference viewer on macOS.

For real model execution, build the original 4-lane/4-stage checkpoint first:

```sh
uv run hf download Qwen/Qwen3.5-2B --revision 15852e8c16360a2fea060d615a32b45270f8a8fc --local-dir models/Qwen3.5-2B
uv run python -m qgpu.merge models/Qwen3.5-2B build/merged-2b
uv run python -m qgpu.world_server serve --backend model --merged build/merged-2b
```

Or run `./run-world.command model` after building. Model and reference modes are
selected at process startup; the viewer cannot silently change the backend.
All GPU tokens and all 80 memory words are checked after each launch. The
reference computes expected values as an observer; it never supplies replacement
state to model mode. Display readback does not become a later host memory upload.

The build and model need substantial RAM and disk space. The upstream estimate
is about 16 GB free memory. Close memory-heavy apps before a model test. A model
run's speed must be measured locally; the reference backend's timings do not
predict model speed. This workload produces 489 execution tokens per frame with
the checked-in PTX; host program prefill adds additional work.

This local build contains 15,261,936,320 bytes of merged weight tensors in eight
shards, plus tokenizer/config files. The base download uses another 4.55 GB.
The validated environment uses PyTorch 2.15.0, Transformers 5.22.1, Python 3.15,
and LLVM 23.1.4. Dependencies are pinned in `uv.lock`; the base revision is pinned
in the download command above.

## Reproduce and verify

```sh
# Recompile real CUDA with a clang that has the NVPTX backend:
uv run python scripts/build_world_kernels.py

# Residency, independent oracle, and upstream bridge/monitor regression checks:
uv run pytest tests/test_world.py tests/test_bridge.py tests/test_monitor.py -q

# Record and replay-check the interpreter:
uv run python -m qgpu.world_server record --backend reference --frames 4 --output recordings/reference.json
uv run python -m qgpu.world_server verify recordings/reference.json

# Actual model run (same workload and verification):
uv run python -m qgpu.world_server record --backend model --merged build/merged-2b --frames 2 --output recordings/model.json
uv run python -m qgpu.world_server verify recordings/model.json

# Original vadd sanity check + two resident frames, with a single model load:
uv run python scripts/validate_world_model.py build/merged-2b --output recordings/model-validation.json
```

`tests/test_world.py` checks 32 frames for each of three seeds against an
independent Python physics/rasterization oracle. A separate test concatenates two
frames (six launches) into a single exact circuit evaluation, checks every
next execution token, and compares all 80 final recurrent memory words. It
uploads initial memory only once. **Circuit evaluation is not full pretrained
Qwen inference**; both checks matter and must be reported separately.

Recordings include generated tokens, host tokens, memory-upload counts, final
snapshots, state hashes, source/PTX hashes, upstream revision, and backend label.
The verifier recomputes execution and checks residency, token streams, and state.
A JSON backend label is not cryptographic proof of inference or of who ran it.
A witnessed model run and its recording provide stronger provenance.

## Recording an honest demonstration

Keep the execution-source badge visible. Show initialization, a complete frame,
then a later frame with zero world-memory upload. Export the trace and show its
verification. Model mode needs a real model recording before describing the
clip as inference inside Qwen. Timelapses and replay should be labeled.

An accurate project description after a successful model run is:

> I extended qwenGPU with a persistent four-particle world and a tiny CUDA
> rasterizer. The world state and pixel buffer live in the model's recurrent
> memory across launches. I added a visual debugger and replayable verification.

Do not claim learned physics, a new foundation model, real-time game performance,
autonomous planning, or worldwide novelty from this prototype.

## Next research steps

1. Establish model residency parity for repeated launches on MPS.
2. Measure tokens, prefill cost, inference time, memory use, and baseline host
   re-upload overhead. Residency alone is not evidence of a speedup.
3. Compare scatter rasterization against gather rendering within the same memory
   budget. Add a task such as navigation with a deterministic success criterion.
4. Investigate more persistent programs or a compiled scheduler only after the
   current device-state invariants are measured and verified.

The first version is intentionally bounded to 32 frames per reset to limit trace
size and growth of the model context. A browser disconnect finishes the current
step; a token or state mismatch requires resetting the world.
