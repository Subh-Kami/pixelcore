# Third-party foundations and notices

## qwenGPU

This repository starts from [vixhal-baraiya/qwenGPU](https://github.com/vixhal-baraiya/qwenGPU)
commit `e2695f6a0a818ebc68e2e14b6b990eddf04c51b3`. Its original history, source,
MIT copyright notice and README are retained. The original README is
[`UPSTREAM_README.md`](UPSTREAM_README.md).

The SIMT circuit, PTX parser/lowerer, general CUDA runtime, base merge approach,
monitor and chat application are upstream work. PixelCore's merge/output-code
conventions adapt that code and Qwenputer's conventions.

Copyright (c) 2026 vixhal-baraiya. MIT; see [LICENSE](LICENSE).

## Qwenputer / chipformer

[Qwenputer](https://github.com/divagr18/qwenputer), by Divyansh Agrawal, supplies
the compiler and exact SwiGLU/Gated DeltaNet primitives. The dependency is pinned
to `f07a0b2e38271454c4d455dc2b9bf78d3d399170` in `pyproject.toml` and `uv.lock`.
The dependency is fetched from its own repository rather than vendored here.

Copyright (c) 2026 Divyansh Agrawal. MIT. Its license is included with the
installed distribution and available in the upstream repository. PixelCore does
not redistribute upstream game ROMs or claim their authorship.

## Qwen3.5

The full model is built from [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B),
revision `15852e8c16360a2fea060d615a32b45270f8a8fc`. Preserve its model license and
notices when building or redistributing a modified checkpoint. The model build
copies the base LICENSE into the checkpoint directory.

The standalone PixelCore checkpoint contains emitted circuit weights and no
pretrained Qwen language weights. It is a separate artifact, not the full model.

## PixelCore additions

The dedicated game circuit, state/pixel protocol, game-specific merge/runtime,
viewer, tests and experimental material were added for PixelCore by Subhanshu
Dwivedi. The root MIT license includes both the original and added-code notices.

Other Python dependencies retain their own licenses and are installed through
the committed project configuration. The generated plots identify measured
execution modes separately; demo media credits the compiler and identifies
recorded playback and external button assistance.
