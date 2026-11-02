"""Play a verified game through emitted weights; external controller selects buttons only."""
import argparse
import json
import platform
import time
from pathlib import Path

from qgpu.pixelcore.runtime import Game, verify

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("checkpoint", type=Path)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--device")
parser.add_argument("--max-ticks", type=int, default=48)
args = parser.parse_args()
started = time.perf_counter()
print("Loading PixelCore weights…", flush=True)
game = Game(args.checkpoint, args.device)
print(f"Loaded {game.meta['kind']} on {game.device} in {time.perf_counter()-started:.2f}s", flush=True)
list(game.step())
for tick in range(args.max_ticks):
    state = game.state
    # A deliberately simple external input policy, not model-generated strategy.
    action = "left" if state.paddle > state.x else "right" if state.paddle < state.x else "stay"
    for event in game.step(action):
        if event["kind"] == "frame":
            print(f"tick {tick+1}: {action}, score={game.state.score}, over={game.state.over}, "
                  f"{game.records[-1]['seconds']:.3f}s, 18 exact tokens", flush=True)
    if game.state.score == 3 or game.state.over:
        break
trace = game.export()
trace["validation"] = {"platform": platform.platform(), "total_seconds": time.perf_counter()-started,
                       "input_policy": "external controller follows model-emitted ball x"}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(trace, indent=2))
print(verify(trace), flush=True)
if game.state.score != 3:
    raise SystemExit("The controller did not finish the game; inspect the trace")
