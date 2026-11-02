"""Independent WeightPong specification. Used for verification, never model execution."""
from dataclasses import asdict, dataclass

STATE0 = 1 << 14
ACTION0 = 1 << 15
END = ACTION0 + 4
TEXT = END + 1
VOCAB = TEXT
NCODE = 16
ACTIONS = ("left", "right", "stay", "reset")


@dataclass(frozen=True)
class State:
    x: int = 3
    y: int = 4
    vx: int = 1
    vy: int = 0
    paddle: int = 5
    score: int = 0
    over: int = 0

    def pack(self):
        return (self.x | self.y << 3 | self.vx << 6 | self.vy << 7 |
                self.paddle << 8 | self.score << 11 | self.over << 13)

    @classmethod
    def unpack(cls, value):
        return cls(value & 7, value >> 3 & 7, value >> 6 & 1, value >> 7 & 1,
                   value >> 8 & 7, value >> 11 & 3, value >> 13 & 1)

    def json(self):
        return {**asdict(self), "won": self.score == 3, "packed": self.pack()}


def advance(state, action):
    if action == "reset":
        return State()
    if state.over or state.score == 3:
        return state
    paddle = max(1, min(6, state.paddle + {"left": -1, "right": 1, "stay": 0}[action]))
    vx, vy = state.vx, state.vy
    x = state.x + (1 if vx else -1)
    if x < 1 or x > 6:
        vx = 1 - vx
        x = state.x + (1 if vx else -1)
    y = state.y + (1 if vy else -1)
    score, over = state.score, state.over
    if y < 1:
        y, vy = 2, 1
    elif y > 6:
        if abs(x - paddle) <= 1:
            y, vy, score = 5, 0, score + 1
            if x != paddle:
                vx = int(x > paddle)
        else:
            y, over = 6, 1
    return State(x, y, vx, vy, paddle, score, over)


def pixels(state):
    out = [0] * 64
    for y in range(7):
        for x in range(8):
            if y == 0 or x in (0, 7):
                out[y * 8 + x] = 1
    for x in range(state.paddle - 1, state.paddle + 2):
        out[56 + x] = 3
    out[state.y * 8 + state.x] = 2
    return out


def frame_tokens(state):
    image = pixels(state)
    return [(chunk << 8) | sum(image[4 * chunk + j] << (2 * j) for j in range(4))
            for chunk in range(16)] + [END]


def action_tokens(state, action):
    next_state = advance(state, action)
    return [STATE0 + next_state.pack(), *frame_tokens(next_state)]


def token_name(t):
    if t < 4096:
        return f"<|px {t >> 8} {t & 255:02x}|>"
    if STATE0 <= t < ACTION0:
        return f"<|game_state {t - STATE0:04x}|>"
    if ACTION0 <= t < END:
        return f"<|game_{ACTIONS[t - ACTION0]}|>"
    if t == END:
        return "<|game_frame_end|>"
    return f"<|game_reserved {t}|>"
