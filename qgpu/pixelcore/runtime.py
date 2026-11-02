"""Actual next-token inference for PixelCore, with an independent lock checker."""
import json
import time
from pathlib import Path

from qgpu.pixelcore.spec import ACTION0, ACTIONS, END, STATE0, State, action_tokens, frame_tokens


class Game:
    def __init__(self, checkpoint=None, device=None):
        self.model = None
        self.meta = {"kind": "reference", "core_layers": 8, "chip_token_offset": 0}
        self.device = "CPU interpreter"
        if checkpoint:
            import torch
            from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

            path = Path(checkpoint)
            self.meta = json.loads((path / "pixelcore.json").read_text())
            self.device = device or ("mps" if torch.backends.mps.is_available() else
                                     "cuda" if torch.cuda.is_available() else "cpu")
            if self.device == "cuda":
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
            cls = AutoModelForImageTextToText if self.meta["kind"] == "merged" else AutoModelForCausalLM
            self.model = cls.from_pretrained(path, dtype=torch.bfloat16).to(self.device).eval()
            self.torch = torch
        self.offset = self.meta["chip_token_offset"]
        self.reset()

    def reset(self):
        self.state = State()
        self.cache = None
        self.pending = []
        self.records = []
        self.initialized = False
        self.failed = False

    def status(self):
        last = self.records[-1] if self.records else None
        return {"backend": self.meta["kind"], "device": self.device, "meta": self.meta,
                "state": self.state.json(), "pixels": last["pixels"] if last else [0] * 64,
                "ticks": sum(r["action"] is not None for r in self.records),
                "last": {k: last[k] for k in ("seconds", "tokens", "verified")} if last else None,
                "failed": self.failed}

    def export(self):
        return {"format": "pixelcore.trace.v1", "meta": self.meta,
                "device": self.device, "records": self.records, "status": self.status()}

    def _run(self):
        x = self.torch.tensor([[self.offset + t for t in self.pending]], device=self.device)
        with self.torch.no_grad():
            result = self.model(input_ids=x, past_key_values=self.cache,
                                use_cache=True, logits_to_keep=1)
        self.cache, self.pending = result.past_key_values, []
        return int(result.logits[0, -1].argmax()) - self.offset

    def head_state(self):
        raw = self.cache.get_recurrent_state(0, state_idx=0)[0, 0, 0].float().cpu()
        return sum(int(raw[2 + i] > 0) << i for i in range(14))

    def step(self, action=None):
        if self.failed:
            raise ValueError("A previous mismatch stopped this game; reset before continuing")
        if len(self.records) >= 256:
            raise ValueError("Recording limit reached; export and reset")
        if action is not None and action not in ACTIONS:
            raise ValueError("Unknown action")
        if not self.initialized:
            if action is not None:
                raise ValueError("Initialize the frame before sending an action")
            host, expected = [STATE0 + State().pack()], frame_tokens(State())
        else:
            if action is None:
                raise ValueError("Choose an action")
            host, expected = [ACTION0 + ACTIONS.index(action)], action_tokens(self.state, action)
        started = time.perf_counter()
        generated, image, next_state = [], [0] * 64, self.state
        self.pending += host
        yield {"kind": "start", "data": {"backend": self.meta["kind"], "action": action,
                                         "expected_tokens": len(expected)}}
        try:
            for index, want in enumerate(expected):
                token = self._run() if self.model else want
                if token != want:
                    raise ValueError(f"Token {index} mismatch: got {token}, expected {want}. "
                                     "No reference fallback was used.")
                generated.append(token)
                self.pending.append(token)
                if STATE0 <= token < ACTION0:
                    next_state = State.unpack(token - STATE0)
                elif token < 4096:
                    chunk, payload = token >> 8, token & 255
                    for j in range(4):
                        image[chunk * 4 + j] = payload >> (2 * j) & 3
                yield {"kind": "token", "data": {"token": token, "index": index + 1,
                                                 "chunk": token >> 8 if token < 4096 else None}}
            # Recurrent head must hold exactly the model-emitted state, not just the right pixels.
            if self.model and self.head_state() != next_state.pack():
                raise ValueError("The model's recurrent game state did not match its state token")
            record = {"action": action, "host": host, "generated": generated,
                      "before": self.state.pack(), "after": next_state.pack(), "pixels": image,
                      "tokens": len(generated), "seconds": time.perf_counter() - started,
                      "verified": True, "head_verified": bool(self.model)}
            self.state, self.initialized = next_state, True
            self.records.append(record)
            yield {"kind": "frame", "data": self.status()}
        except Exception:
            self.failed = True
            raise


def verify(trace):
    if trace.get("format") != "pixelcore.trace.v1":
        raise ValueError("Unsupported trace")
    game = Game()
    for index, record in enumerate(trace["records"]):
        list(game.step(record["action"]))
        want = game.records[-1]
        for key in ("host", "generated", "before", "after", "pixels", "tokens"):
            if record[key] != want[key]:
                raise ValueError(f"Mismatch at frame {index}, {key}")
    return {"frames": len(game.records), "ticks": max(0, len(game.records) - 1),
            "tokens": sum(r["tokens"] for r in game.records), "consistent": True,
            "won": game.state.score == 3, "backend_claim": trace["meta"]["kind"]}
