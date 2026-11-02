"""WeightWorld: an 80-word resident simulation and framebuffer on qwenGPU.

Only the first launch uploads world memory. Later launches contain ROM/LIVE/EXEC
records only. The model backend uses the actual merged model; the reference backend
is explicitly an interpreter. Verification is an independent observer, never a fallback.
"""

import dataclasses
import hashlib
import json
import random
import time
from pathlib import Path

from qgpu import ptx
from qgpu.bridge import Cancelled, Session
from qgpu.circuit import kernel
from qgpu.lower import lower
from qgpu.merge import STAGES
from qgpu.monitor import Monitor, Program
from qgpu.runtime import GPUError, host_tokens

ROOT = Path(__file__).resolve().parent.parent
UPSTREAM = "e2695f6a0a818ebc68e2e14b6b990eddf04c51b3"
WORDS = 80
MAX_FRAMES = 32
PHASES = (("clear", 64), ("advance", 8), ("draw", 4))


def initial_memory(seed=7):
    rng = random.Random(seed)
    # Include near-wall positions to make reflections visible in a short recording.
    positions = [14, 20, 117, 25, 35, 116, 101, 98]
    velocity = [rng.choice((-7, -5, 5, 7)) for _ in range(8)]
    return {i: v & 0xFFFFFFFF for i, v in enumerate(positions + velocity + [0] * 64)}


def signed(v):
    return v - (1 << 32) if v & (1 << 31) else v


def snapshot(mem):
    return {"positions": [signed(mem.get(i, 0)) for i in range(8)],
            "velocities": [signed(mem.get(i, 0)) for i in range(8, 16)],
            "pixels": [mem.get(i, 0) for i in range(16, 80)]}


def prepare_phase(name, lanes=4, stages=4):
    threads = dict(PHASES)[name]
    block = min(threads, 32)
    mul, logic, atom = STAGES[stages]
    return lower(ptx.load(ROOT / f"kernels/world_{name}.ptx")[f"world_{name}"],
                 (threads // block,), (block,), [0], {0: [0] * WORDS}, lanes=lanes,
                 stages=stages, mul_stages=mul, logic_stages=logic, atom_stages=atom,
                 write_through=True)


def resident_host_tokens(launch, upload=None):
    """Program upload plus launch start; world data is optional and explicit."""
    return host_tokens(dataclasses.replace(launch, mem=upload or {}))


def reference_execution(launch, mem):
    tr = kernel.run_blocks(launch.blocks, mem, launch.warps, lanes=4)
    prefix = host_tokens(dataclasses.replace(launch, mem=mem))
    assert tr.tokens[:len(prefix)] == prefix
    return tr.tokens[len(prefix):], tr.mem


def digest(mem):
    raw = json.dumps([mem.get(i, 0) for i in range(WORDS)], separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


class World:
    def __init__(self, backend="reference", merged=None, seed=7):
        if backend not in ("reference", "model"):
            raise ValueError("backend must be reference or model")
        self.backend = backend
        self.session = None
        if backend == "model":
            if not merged or not (Path(merged) / "qgpu.json").is_file():
                raise GPUError("Build a merged checkpoint first; see WEIGHTWORLD.md.")
            self.session = Session(merged, check=False)
            if (self.session.meta["lanes"], self.session.meta["stages"]) != (4, 4):
                raise GPUError("WeightWorld requires the original 4-lane, 4-stage build.")
        self.programs = {name: prepare_phase(name) for name, _ in PHASES}
        self.reset(seed)

    def reset(self, seed=7):
        self.seed, self.frame, self.launches = seed, 0, 0
        self.mem = initial_memory(seed)
        self.uploaded = False
        self.failed = False
        self.records = []
        self.last = None
        if self.session:
            self.session.reset()
        self.started = time.time()

    def status(self):
        return {"backend": self.backend, "frame": self.frame, "seed": self.seed,
                "launches": self.launches, "words": WORDS, "max_frames": MAX_FRAMES,
                "failed": self.failed, "snapshot": snapshot(self.mem),
                "last": self.last,
                "device": self.session.dev if self.session else "CPU interpreter",
                "checkpoint": self.session.name if self.session else None}

    def export(self):
        hashes = {f"world_{name}.{ext}": hashlib.sha256(
            (ROOT / f"kernels/world_{name}.{ext}").read_bytes()).hexdigest()
            for name, _ in PHASES for ext in ("cu", "ptx")}
        return {"format": "weightworld.trace.v1", "backend": self.backend,
                "upstream": {"url": "https://github.com/vixhal-baraiya/qwenGPU",
                             "commit": UPSTREAM},
                "kernels": hashes, "seed": self.seed,
                "initial_memory": initial_memory(self.seed),
                "model": self.session.meta if self.session else None,
                "status": self.status(), "records": self.records}

    def step(self):
        if self.failed:
            raise GPUError("The last step failed. Reset the world before continuing.")
        if self.frame >= MAX_FRAMES:
            raise GPUError("Recording limit reached. Export the trace, then reset.")
        t0 = time.perf_counter()
        summary = {"generated_tokens": 0, "host_tokens": 0, "host_memory_words": 0,
                   "verified": True, "phases": []}
        try:
            for name, _ in PHASES:
                for event in self._phase(name):
                    if event["kind"] == "phase_done":
                        data = event["data"]
                        for key in ("generated_tokens", "host_tokens", "host_memory_words"):
                            summary[key] += data[key]
                        summary["phases"].append(data)
                    yield event
            self.frame += 1
            summary["seconds"] = time.perf_counter() - t0
            self.last = summary
            yield {"kind": "frame", "data": {**self.status(), "summary": summary}}
        except Exception:
            self.failed = True
            raise

    def _phase(self, name):
        launch = self.programs[name]
        upload = initial_memory(self.seed) if not self.uploaded else None
        host = resident_host_tokens(launch, upload)
        expected, want = reference_execution(launch, self.mem)
        monitor = Monitor()
        program = Program.of(launch, 4)
        program.memory = list(range(WORDS))
        monitor.load(program)
        record = {"phase": name, "frame": self.frame + 1, "backend": self.backend,
                  "program": dataclasses.asdict(program), "host": host, "generated": [],
                  "before_sha256": digest(self.mem), "host_memory_words": len(upload or {}),
                  "verified": False}
        self.records.append(record)
        yield {"kind": "phase", "data": {"name": name, "backend": self.backend,
               "program": record["program"], "host_memory_words": len(upload or {}),
               "host_tokens": len(host)}}
        started = time.perf_counter()
        if self.session:
            s = self.session
            s._push([s.v0 + t for t in host])

            def generated():
                logits = s._run()
                for _ in range(20000):
                    if s.cancel:
                        raise Cancelled()
                    token = int(logits.argmax()) - s.v0
                    s._push([s.v0 + token])
                    yield token
                    if token in {kernel.EXEC(0, launch.warps, ph) for ph in (False, True)}:
                        return
                    logits = s._run()
                raise GPUError("The model launch exceeded the 20,000-token limit.")

            source = generated()
        else:
            source = iter(expected)
        for i, token in enumerate(source):
            record["generated"].append(token)
            if i >= len(expected) or token != expected[i]:
                raise GPUError(f"{name}: token {i} disagreed with the independent reference. "
                               "Execution stopped; no reference fallback was used.")
            monitor.feed(token)
            yield {"kind": "token", "data": {"t": token, "block": monitor.block,
                   "warp": monitor.warp, "last": monitor.last,
                   "reg_writes": monitor.reg_writes, "mem_writes": monitor.mem_writes,
                   "written": monitor.written, "index": i + 1}}
        if len(record["generated"]) != len(expected):
            raise GPUError(f"{name}: launch ended early.")
        if self.session:
            actual = self.session.gpu.read(self.session.ctx.cache)
            if any(actual.get(a, 0) != want.get(a, 0) for a in range(WORDS)):
                raise GPUError(f"{name}: recurrent memory disagreed with the reference.")
            # Actual model state is what the host displays. The reference only checks it.
            self.mem = {a: actual.get(a, 0) for a in range(WORDS)}
        else:
            self.mem = {a: want.get(a, 0) for a in range(WORDS)}
        self.uploaded = True
        self.launches += 1
        data = {"name": name, "generated_tokens": len(record["generated"]),
                "host_tokens": len(host), "host_memory_words": len(upload or {}),
                "seconds": time.perf_counter() - started, "verified": True}
        record.update({"verified": True, "snapshot": snapshot(self.mem),
                       "after_sha256": digest(self.mem), "summary": data})
        yield {"kind": "phase_done", "data": data}


def verify_trace(trace):
    """Replay exported evidence, checking host uploads, execution, state, and kernel hashes.

    This validates consistency with the interpreter. A JSON file is not cryptographic
    proof that model inference occurred; backend provenance still needs a witnessed run.
    """
    if trace.get("format") != "weightworld.trace.v1":
        raise ValueError("Unsupported trace format")
    if trace.get("backend") not in ("reference", "model"):
        raise ValueError("Invalid trace backend")
    if type(trace.get("seed")) is not int or not 0 <= trace["seed"] <= 1000000:
        raise ValueError("Invalid trace seed")
    world = World(seed=trace["seed"])
    if {int(k): v for k, v in trace["initial_memory"].items()} != world.mem:
        raise ValueError("Initial memory does not match the seed")
    if trace["kernels"] != world.export()["kernels"]:
        raise ValueError("Trace kernels do not match this checkout")
    records = trace.get("records", [])
    if not records or len(records) > MAX_FRAMES * len(PHASES):
        raise ValueError("Trace must contain 1 to 96 launches")
    for index, record in enumerate(records):
        name = PHASES[index % len(PHASES)][0]
        launch = world.programs[name]
        upload = initial_memory(world.seed) if index == 0 else None
        expected, want = reference_execution(launch, world.mem)
        program = dataclasses.asdict(Program.of(launch, 4))
        program["memory"] = list(range(WORDS))
        # JSON roundtrips turn the live-out tuples into lists.
        program = json.loads(json.dumps(program))
        checks = (record["phase"] == name,
                  record["frame"] == index // len(PHASES) + 1,
                  record["host"] == resident_host_tokens(launch, upload),
                  record["generated"] == expected,
                  json.loads(json.dumps(record["program"])) == program,
                  record["host_memory_words"] == (WORDS if index == 0 else 0),
                  record["verified"] is True,
                  record["before_sha256"] == digest(world.mem),
                  record["after_sha256"] == digest(want),
                  record["snapshot"] == snapshot(want))
        if not all(checks):
            raise ValueError(f"Trace mismatch at launch {index + 1}")
        world.mem = {a: want.get(a, 0) for a in range(WORDS)}
    return {"launches": len(records), "complete_frames": len(records) // 3,
            "consistent": True, "backend_claim": trace["backend"]}
