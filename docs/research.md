# PixelCore: research and experiment notes

Research and implementation notes, 2–3 November 2026.

## The technical idea

PixelCore is a narrow game-specific processor, compiled into Qwen-compatible
transformer weights. It computes an 8×8 paddle-and-ball game and rasterizes its
frame. The host supplies buttons; the model emits the next state and all pixels.
An independent CPU specification checks the output, stopping on a mismatch.

The implementation contributes a game-specific circuit, a 14-bit recurrent-state
layout, a packed-pixel token protocol, and a verified merge/runtime. It uses Qwenputer's existing compiler and
memory primitives, and qwenGPU's merge conventions. It does not invent those
foundations or establish worldwide novelty.

## Research that informed the design

1. **Tracr: Compiled Transformers as a Laboratory for Interpretability.** This
   establishes a practical approach to compiling readable algorithms into
   transformer weights with known internal structure. Its RASP compiler is not
   used here; PixelCore uses chipformer. The useful design principle is to have
   an explicit program specification and inspectable, testable computations.
   [Primary paper](https://arxiv.org/html/2301.05062v5).
2. **Tracr-Injection: Distilling Algorithms into Pre-trained Language Models.**
   This studies transferring compiled algorithms into a pretrained model through
   distillation. PixelCore takes a different route: it inserts fixed compiled
   layers, without training. The paper is relevant context, not evidence that
   our specific game or merge is novel.
   [Primary paper](https://arxiv.org/html/2505.10719v2).
3. **Synthesis and Verification of Transformer Programs (2026).** This connects
   C-RASP verification to synchronous dataflow/model-checking techniques. It
   motivated a precise finite-state specification and exhaustive reachable-state
   checks. PixelCore does not implement that paper's SMT/model-checking method
   and does not claim formal verification.
   [Primary paper](https://arxiv.org/html/2602.16473v1).
4. **Current Transformers Qwen3.5 documentation, checked against the installed
   5.22.1 implementation.** It describes the hybrid Gated DeltaNet/full-attention
   stack, the conditional-generation and text-only classes, and slower reference
   implementations when optional kernels are absent. We preserved the stock
   layer types and used bf16 inference on MPS. We did not install NVIDIA/AMD-only
   fast kernels on the Mac.
   [Official documentation](https://huggingface.co/docs/transformers/model_doc/qwen3_5).
5. **Qwenputer.** This is the actual compiler dependency and prior work for
   exact computation and RAM in recurrent state. Its CPU, filesystem, CHIP-8
   implementation and memory primitives predate this project. The dependency
   revision is pinned in the source's `pyproject.toml`/`uv.lock`.
   [Primary repository](https://github.com/divagr18/qwenputer).
6. **qwenGPU.** This provided the original SIMT/CUDA baseline and merge/runtime
   patterns. We first built and tested it, then implemented a narrower core to
   avoid hundreds of general GPU execution records for each small visual frame.
   [Primary repository](https://github.com/vixhal-baraiya/qwenGPU).
7. **Qwenputer Doom-floor-plan raycaster.** This existing project already renders
   and plays a turn-based environment through a CPU compiled into model weights.
   It rules out a broad “first game inside an LLM” claim. PixelCore instead has
   game-specific compiled logic and emits packed color pixels directly.
   [Primary repository](https://github.com/mapestree/qwenputer-doom).

## What was built

The combined checkpoint has Qwen3.5-2B's 24 pretrained layers and eight compiled
game layers. The hidden width increases from 2048 to 2176. Its tensor data totals
5,805,412,992 bytes in three shards. No game weights are learned.

The recurrent state stores ball x/y, two direction bits, paddle position, a
2-bit score and a game-over bit: 14 bits altogether. A model-generated state
token commits that state. Sixteen further tokens each carry four two-bit pixels,
followed by an end token. An ordinary tick therefore produces 18 tokens. The
initial frame emits 17 because the initial state was supplied as an input.

Wall reflections, paddle contact steering, three-save wins, misses, frozen
terminal states and reset are all part of the circuit. Frame display only
unpacks the emitted pixels. It does not calculate collisions or draw sprites
from host-side positions.

A second checkpoint contains the same compiled game as a small standalone
transformer: eight game layers and a zero helper layer for cache position
counting. It is about 54 MiB, without Qwen's pretrained language backbone.
Its faster playback must not be presented as full-Qwen performance.

## Evidence collected

- **Finite-state circuit check:** every action from 266 reachable states, 1,064
  transitions in total, matched the independent specification and pixel output
  in the exact circuit evaluator.
- **Persistent sequence:** a 100-tick sequence matched without host state
  re-upload between ticks.
- **Actual full Qwen run on MPS:** a complete three-save game took 29 ticks and
  emitted 539/539 correct tokens, including the initial frame. Recurrent memory
  matched the emitted state after every frame.
- **Additional live browser run:** manual held movement, pause, a compiled reset,
  and a deliberately missed ball. Its exported 323-token trace replayed exactly
  and ended in the model-computed game-over state.
- **Full-Qwen timing:** mean 1.862 seconds/tick, median 1.842, range 1.643–2.376,
  over the successful recorded 29 ticks.
- **Standalone timing:** mean 0.165 seconds/tick, median 0.152, range 0.121–0.528,
  over the corresponding successful game.
- **Small ordinary-text regression check:** four prompts, 40 positions, 40/40
  top-token agreement with base Qwen, mean KL 5.49e-4, no game-token leakage.
  This was a bf16 CPU check, not a broad language benchmark.
- **Failure guards:** deliberately wrong model tokens and recurrent state both
  stop the runtime rather than substituting the reference answer.

The standalone and full-Qwen timings are separate modes. The earlier CUDA
particle prototype used a different workload, so its 489 tokens/frame and roughly
three-minute frames are not a like-for-like speed benchmark for this game.

## Reproduction and provenance

The source includes the compiler graph, independent game specification, merge,
runtime, viewer, finite-state checks and validation scripts. The base checkpoint
revision is `15852e8c16360a2fea060d615a32b45270f8a8fc`; the qwenGPU starting
commit is `e2695f6a0a818ebc68e2e14b6b990eddf04c51b3`. Tested dependencies include
PyTorch 2.15.0 and Transformers 5.22.1.

Recordings include host buttons, generated tokens, states, pixels, timings and
backend metadata. A JSON backend label can be forged; it is not cryptographic
attestation. The supplied logs and the actual witnessed runs establish the
provenance of these particular recordings. Exhaustive circuit checks and sampled
bf16 runs do not prove every possible hardware/library implementation correct.

## Contribution boundaries

The specific contribution is the dedicated WeightPong transition/rasterization
circuit and its token protocol, integrated into a reproducible Qwen checkpoint
and an inspectable runtime. General transformer compilation, recurrent RAM and
in-model games are prior work. No worldwide-first claim is made.

This is a narrow specialization: unlike qwenGPU's SIMT machine, it cannot load
arbitrary CUDA programs. The rules are compiled into the checkpoint. A useful
next research question is whether such cores can be made composable or
reconfigurable without recompiling each game's rules, evaluated on matched
workloads. That is future work, not an implemented feature.
