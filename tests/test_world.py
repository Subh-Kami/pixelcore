"""Residency, physics, pixels, and exported evidence, independently checked."""

import copy

import pytest

from qgpu.circuit import kernel
from qgpu.world import World, initial_memory, snapshot, verify_trace


def oracle(state):
    positions, velocities = [], []
    for p, v in zip(state["positions"], state["velocities"]):
        q = p + v
        if q < 0:
            q, v = -q, -v
        elif q > 127:
            q, v = 254 - q, -v
        positions.append(q)
        velocities.append(v)
    pixels = [0] * 64
    for i in range(4):
        x, y = positions[2 * i:2 * i + 2]
        pixels[(y // 16) * 8 + x // 16] = i + 1
    return {"positions": positions, "velocities": velocities, "pixels": pixels}


@pytest.mark.parametrize("seed", [0, 7, 42])
def test_32_frames_match_independent_world_and_resident_uploads(seed):
    world = World(seed=seed)
    want = snapshot(initial_memory(seed))
    for _ in range(32):
        want = oracle(want)
        events = list(world.step())
        assert events[-1]["data"]["snapshot"] == want
    assert [r["host_memory_words"] for r in world.records] == [80] + [0] * 95
    for record in world.records[1:]:
        assert not any(kernel.MEM0 <= t < kernel.TEXT for t in record["host"])
    assert verify_trace(world.export())["complete_frames"] == 32


def test_trace_detects_mutated_execution_and_memory_upload():
    world = World()
    list(world.step())
    good = world.export()
    assert verify_trace(good)["consistent"]
    for key in ("generated", "host"):
        bad = copy.deepcopy(good)
        bad["records"][1][key][0] ^= 1
        with pytest.raises(ValueError, match="mismatch"):
            verify_trace(bad)


def test_two_frames_through_actual_compiled_circuit_without_memory_reupload():
    """Exact circuit IR, distinct from interpreter and full pretrained-model inference.

    Every next execution token and all 80 final words must match across six launches
    in one circuit context. No second host MEM upload can conceal a residency bug.
    """
    from chipformer.q35.ir import evaluate

    from qgpu.merge import STAGES

    world = World()
    list(world.step())
    list(world.step())
    sequence, checks = [], []
    for record in world.records:
        start = len(sequence) + len(record["host"]) - 1
        sequence += record["host"] + record["generated"]
        checks += [(start + i, token) for i, token in enumerate(record["generated"])]
    mul, logic, atom = STAGES[4]
    circuit = kernel.Kernel(lanes=4, stages=4, mul_stages=mul,
                            logic_stages=logic, atom_stages=atom)
    carry = {}
    result = evaluate(circuit, sequence, carry=carry)
    bad = [(i, t) for i, t in checks if kernel.predicted(circuit, result, i) != t]
    assert not bad, f"{len(bad)} incorrect circuit predictions: {bad[:3]}"
    actual = kernel.memory(circuit, carry)
    assert {a: actual.get(a, 0) for a in range(80)} == world.mem


def test_reset_reuploads_seed_and_limits_recording():
    world = World()
    list(world.step())
    first = world.status()["snapshot"]
    world.reset(7)
    list(world.step())
    assert world.status()["snapshot"] == first
    assert world.records[0]["host_memory_words"] == 80
    world.frame = 32
    with pytest.raises(Exception, match="limit"):
        list(world.step())


def test_model_mismatch_stops_without_reference_fallback():
    from types import SimpleNamespace

    from qgpu.runtime import GPUError

    class WrongLogits:
        def argmax(self):
            return 100000 + kernel.TEXT  # never a valid next execution token

    world = World()
    before = dict(world.mem)
    world.session = SimpleNamespace(v0=100000, cancel=False,
                                    _push=lambda _: None, _run=lambda: WrongLogits())
    with pytest.raises(GPUError, match="no reference fallback"):
        list(world.step())
    assert world.failed and world.frame == 0 and world.mem == before
    assert not world.records[0]["verified"]
    with pytest.raises(GPUError, match="Reset"):
        list(world.step())


def test_trace_rejects_changed_initial_memory_or_program():
    world = World()
    list(world.step())
    for field in ("initial_memory", "program"):
        bad = copy.deepcopy(world.export())
        if field == "initial_memory":
            bad[field][0] ^= 1
        else:
            bad["records"][0][field]["warps"] += 1
        with pytest.raises(ValueError):
            verify_trace(bad)
