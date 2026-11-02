from collections import deque

from chipformer.q35.ir import evaluate

from qgpu.pixelcore.circuit import PixelCore, predicted
from qgpu.pixelcore.spec import ACTION0, ACTIONS, STATE0, State, action_tokens, advance, frame_tokens


def reachable():
    seen, queue = {State()}, deque([State()])
    while queue:
        state = queue.popleft()
        for action in ACTIONS:
            nxt = advance(state, action)
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return sorted(seen, key=lambda s: s.pack())


def test_all_reachable_game_transitions_and_pixels_exact_in_compiled_circuit():
    circuit = PixelCore()
    seq, checks = [], []
    for state in reachable():
        for action in ACTIONS:
            seq += [STATE0 + state.pack(), ACTION0 + ACTIONS.index(action)]
            expected = action_tokens(state, action)
            start = len(seq) - 1
            seq += expected
            checks += [(start + i, token) for i, token in enumerate(expected)]
    result = evaluate(circuit, seq)
    bad = [(i, want, predicted(circuit, result, i)) for i, want in checks
           if predicted(circuit, result, i) != want]
    assert not bad, f"{len(bad)} incorrect predictions: {bad[:5]}"
    assert circuit.n_layers() == 8


def test_long_game_keeps_state_without_host_state_reupload():
    circuit = PixelCore()
    state = State()
    seq, checks = [STATE0 + state.pack()], []
    first = frame_tokens(state)
    checks += [(len(seq) - 1 + i, t) for i, t in enumerate(first)]
    seq += first
    for index in range(100):
        # Controller chooses buttons; the circuit computes physics and pixels.
        action = "left" if state.paddle > state.x else "right" if state.paddle < state.x else "stay"
        if state.over or state.score == 3:
            action = "reset"
        seq.append(ACTION0 + ACTIONS.index(action))
        expected = action_tokens(state, action)
        checks += [(len(seq) - 1 + i, t) for i, t in enumerate(expected)]
        seq += expected
        state = advance(state, action)
    carry = {}
    result = evaluate(circuit, seq, carry=carry)
    assert all(predicted(circuit, result, i) == want for i, want in checks)


def test_terminal_freeze_reset_and_exact_pixels():
    lost = advance(State(x=6, y=6, vx=1, vy=1, paddle=1), "stay")
    assert lost.over
    assert advance(lost, "right") == lost
    assert advance(lost, "reset") == State()
    won = advance(State(x=3, y=6, vx=1, vy=1, paddle=4, score=2), "stay")
    assert won.score == 3 and not won.over
    assert advance(won, "left") == won


def test_runtime_stops_on_model_token_or_head_mismatch():
    import pytest

    from qgpu.pixelcore.runtime import Game

    game = Game()
    game.model = object()
    game._run = lambda: 9999
    with pytest.raises(ValueError, match="No reference fallback"):
        list(game.step())
    assert game.failed and not game.records
    game = Game()
    expected = iter(frame_tokens(State()))
    game.model = object()
    game._run = lambda: next(expected)
    game.head_state = lambda: State().pack() ^ 1
    with pytest.raises(ValueError, match="recurrent game state"):
        list(game.step())
    assert game.failed and not game.records
