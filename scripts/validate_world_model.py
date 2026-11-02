"""One upstream vadd sanity check, then two resident frames, on one loaded model."""

import argparse
import json
import platform
import time
from pathlib import Path

from qgpu.circuit import kernel
from qgpu.runtime import host_tokens, prepare
from qgpu.world import World, verify_trace

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("merged", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
started = time.perf_counter()
print("Loading the merged checkpoint…", flush=True)
world = World("model", args.merged)
s = world.session
print(f"Loaded on {s.dev} in {time.perf_counter() - started:.1f}s", flush=True)
s.check = True
prepared = prepare("vadd", {"a": [1, -2, 3, 4], "b": [10, 20, -30, 40]}, 4, 4, 4)
sanity = None
tr = kernel.run_blocks(prepared.launch.blocks, prepared.launch.mem, prepared.launch.warps, lanes=4)
expected = tr.tokens[len(host_tokens(prepared.launch)):]
index = 0
for event in s._launch(prepared):
    if event.kind == "token":
        if index >= len(expected) or event.data["t"] != expected[index]:
            raise SystemExit(f"Upstream vadd token mismatch at {index}; execution stopped")
        index += 1
    if event.kind == "result":
        sanity = event.data
if not sanity or sanity["verified"] is not True or sanity["outputs"] != {"c": [11, 18, -27, 44]}:
    raise SystemExit(f"Upstream vadd failed: {sanity}")
print(f"Upstream vadd verified: {sanity}", flush=True)
world.reset(7)
for _ in range(2):
    for event in world.step():
        if event["kind"] == "phase":
            print(f"Frame {world.frame + 1} / {event['data']['name']}", flush=True)
        elif event["kind"] == "phase_done":
            d = event["data"]
            print(f"  verified {d['generated_tokens']} tokens in {d['seconds']:.2f}s; "
                  f"host memory upload {d['host_memory_words']} words", flush=True)
        elif event["kind"] == "frame":
            print(f"Frame {world.frame} verified: {world.last}", flush=True)
trace = world.export()
trace["validation"] = {"upstream_vadd": sanity, "platform": platform.platform(),
                       "torch": s.torch.__version__, "device": s.dev,
                       "total_seconds": time.perf_counter() - started}
print(verify_trace(trace), flush=True)
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(trace, indent=2))
print(f"Saved {args.output}", flush=True)
