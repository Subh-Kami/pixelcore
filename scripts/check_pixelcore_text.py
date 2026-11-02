"""Small text-distribution regression check; separate from gameplay correctness."""
import argparse
import gc
import json
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("base", type=Path)
parser.add_argument("merged", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
prompts = ["The capital of France is", "def fibonacci(n):\n    if n < 2:\n        return n\n",
           "CUDA kernels run on thousands of threads grouped into warps of",
           "Photosynthesis converts light energy into"]
tok = AutoTokenizer.from_pretrained(args.base)
ids = [tok.encode(p) for p in prompts]
v0 = json.loads((args.merged / "pixelcore.json").read_text())["chip_token_offset"]

@torch.no_grad()
def distributions(path):
    model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16).eval()
    result = []
    for values in ids:
        logits = model(input_ids=torch.tensor([values]), use_cache=False).logits[0].float()
        if path == args.merged:
            assert bool((logits.argmax(-1) < v0).all()), "a game token leaked into ordinary text"
        result.append(torch.log_softmax(logits[:, :v0], -1))
    del model
    gc.collect()
    return result

print("Checking base text distributions on CPU…", flush=True)
base = distributions(args.base)
print("Checking PixelCore merged text distributions on CPU…", flush=True)
merged = distributions(args.merged)
kls = torch.cat([(p.exp() * (p-q)).sum(-1) for p,q in zip(base,merged)])
agree = sum(int((p.argmax(-1)==q.argmax(-1)).sum()) for p,q in zip(base,merged))
result = {"prompts": len(prompts), "positions": sum(map(len,ids)), "top_token_agrees": agree,
          "mean_KL": float(kls.mean()), "max_KL": float(kls.max()),
          "game_token_leakage": False, "device": "cpu", "dtype": "bfloat16"}
assert result["mean_KL"] < 0.02, "unexpectedly large text distribution drift"
args.output.write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2), flush=True)
