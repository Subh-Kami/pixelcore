"""WeightPong game rules and rasterizer, expressed as exact Qwen layer computations.

This is a dedicated state machine, not PTX running on qwenGPU. The compiler is
Qwenputer's MIT-licensed chipformer. Every emitted state and pixel token must be
predicted by the compiled weights; the Python specification is only an observer.
"""
from chipformer.dsl.graph import Expr
from chipformer.q35.ir import Target
from chipformer.q35.prims import Circ, from_bits

from qgpu.pixelcore.spec import ACTION0, END, NCODE, STATE0, TEXT, State

KAPPA = 2.0 ** -20
TAU_TEXT = 2.0 ** -40
CODE_AMP = 128


class PixelCore(Circ):
    def __init__(self, target=None):
        super().__init__(target or Target())
        c, one = self, self.one
        f = lambda name, fn: c.token_feature(name, fn, full=True)
        self.mu = f("game.mu", lambda t: int(t != TEXT))
        self.is_text = f("game.text", lambda t: int(t == TEXT))
        self.tau = f("game.tau", lambda t: TAU_TEXT if t == TEXT else -1)
        is_state = f("game.is_state", lambda t: int(STATE0 <= t < ACTION0))
        actions = [f(f"game.action{i}", lambda t, i=i: int(t == ACTION0 + i)) for i in range(4)]
        groups = [f(f"game.chunk{i}", lambda t, i=i: int(0 <= t < 4096 and t >> 8 == i))
                  for i in range(16)]
        is_end = f("game.is_end", lambda t: int(t == END))
        bits = [f(f"game.input{i}", lambda t, i=i: int(STATE0 <= t < ACTION0 and t >> i & 1))
                for i in range(14)]
        state = c.slot_head("game.state", {0: is_state}, {0: self.mu},
                            {2 + i: b for i, b in enumerate(bits)}, one, gate=self.mu)
        self.state_bits = state
        self.state_head = next(n for n in c.nodes if n.name == "game.state")
        x, y, vx, vy, paddle, score, over = (state[:3], state[3:6], state[6], state[7],
                                           state[8:11], state[11:13], state[13])
        xo, yo, po = (c.onehot_of_bits(value, name) for value, name in
                      ((x, "game.x"), (y, "game.y"), (paddle, "game.paddle")))
        so = c.onehot_of_bits(score, "game.score")
        terminal = c.node(c.t_and([over]) + c.t_and([one - over, so[3]]), "game.terminal")
        is_action = sum(actions, Expr())
        go = c.and_([is_action, one - actions[3], one - terminal], "game.go")
        hold = c.and_([is_action, one - actions[3], terminal], "game.hold")

        pad_cases = [([po[p], actions[a]], max(1, min(6, p + (-1, 1, 0)[a])))
                     for p in range(1, 7) for a in range(3)]
        new_pad = c.select_bits(pad_cases, 3, "game.next_paddle")
        # Frame tokens don't move the paddle. new_pad is only consumed by action logic.
        x_cases = []
        for pos in range(1, 7):
            for direction in (0, 1):
                q = pos + (1 if direction else -1)
                if q < 1 or q > 6:
                    q = pos - (1 if direction else -1)
                x_cases.append(([xo[pos], vx if direction else one - vx], q))
        new_x = c.select_bits(x_cases, 3, "game.next_x")
        new_vx = c.node(c.t_and([vx, one - xo[6]]) + c.t_and([one - vx, xo[1]]),
                        "game.next_vx")
        delta = from_bits(new_x) - from_bits(new_pad)
        aligned = c.node(c.t_in(delta, -1, 2, -6, 6), "game.aligned")
        floor = c.and_([yo[6], vy], "game.floor")
        hit = c.and_([floor, aligned], "game.hit")
        miss = c.and_([floor, one - aligned], "game.miss")
        right_contact = c.node(c.t_ge(delta, 1, -6, 6), "game.right_contact")
        center_contact = c.node(c.t_eq(delta, 0, -6, 6), "game.center_contact")
        steered_vx = c.node(c.t_and([one - hit, new_vx]) + c.t_and([hit, right_contact])
                            + c.t_and([hit, center_contact, new_vx]), "game.steered_vx")
        y_cases = []
        for pos in range(1, 7):
            for direction in (0, 1):
                lits = [yo[pos], vy if direction else one - vy]
                if pos == 6 and direction:
                    y_cases += [(lits + [hit], 5), (lits + [one - hit], 6)]
                else:
                    q = 2 if pos == 1 and not direction else pos + (1 if direction else -1)
                    y_cases.append((lits, q))
        new_y = c.select_bits(y_cases, 3, "game.next_y")
        new_vy = c.node(c.t_and([vy, one - hit]) + c.t_and([one - vy, yo[1]]), "game.next_vy")
        new_score = c.select_bits([([so[s], hit if h else one - hit], (s + h) % 4)
                                   for s in range(4) for h in (0, 1)], 2, "game.next_score")
        new_over = c.node(c.t_and([over]) + c.t_and([one - over, miss]), "game.next_over")
        proposed = new_x + new_y + [steered_vx, new_vy] + new_pad + new_score + [new_over]
        initial = State().pack()
        next_state = []
        for i, value in enumerate(proposed):
            terms = c.t_and([go, value]) + c.t_and([hold, state[i]])
            if initial >> i & 1:
                terms += c.t_and([actions[3]])
            next_state.append(c.node(terms, f"game.commit{i}"))

        # Four 2-bit pixels per token: all pixel colors come from the model.
        chunk = [is_state, *groups[:-1]]
        payload = [[] for _ in range(8)]
        for g in range(16):
            for j in range(4):
                cell = 4 * g + j
                row, col = divmod(cell, 8)
                wall = row < 7 and (row == 0 or col in (0, 7))
                if wall:
                    payload[2 * j] += c.t_and([chunk[g]])
                if row == 7:
                    pad_here = sum((po[p] for p in range(1, 7) if abs(p - col) <= 1), Expr())
                    payload[2 * j] += c.t_and([chunk[g], pad_here])
                    payload[2 * j + 1] += c.t_and([chunk[g], pad_here])
                elif 1 <= row <= 6 and 1 <= col <= 6:
                    payload[2 * j + 1] += c.t_and([chunk[g], yo[row], xo[col]])
        ending = groups[-1] + is_end
        out = []
        for i in range(NCODE):
            terms = list(payload[i]) if i < 8 else []
            if 8 <= i < 12:
                terms += [t for g in range(16) if g >> (i - 8) & 1 for t in c.t_and([chunk[g]])]
            if i < 14:
                terms += c.t_and([next_state[i]])
            if i == 14:
                terms += c.t_and([is_action])
            if END >> i & 1:
                terms += c.t_and([ending])
            out.append(c.node(terms, f"game.output{i}"))
        self.next_bits = out
        self.owner = is_action + is_state + sum(groups, Expr()) + is_end
        self.code = [f(f"game.code{i}", lambda t, i=i: 0 if t == TEXT else
                       CODE_AMP * (2 * (t >> i & 1) - 1)) for i in range(NCODE)]
        for i, e in enumerate(self.code):
            c.ffn([(-1, one, e), (1, one, -e)], f"game.code_clear{i}", into=e)
        top = max(n.level for n in c.nodes)
        for i, e in enumerate(self.code):
            terms = c.t_and([out[i]], 2 * KAPPA) + c.t_and([self.owner], -KAPPA)
            c.ffn(terms, f"exit.code{i}", into=e, min_level=top)
        c.ffn([(1, self.mu, one)], "exit.tau", into=self.tau, min_level=top)
        self.keep = [next(iter(e.terms)) for e in self.code + [self.tau, self.mu]]


def predicted(circuit, values, index):
    return sum(int(values[index, next(iter(bit.terms))]) << i
               for i, bit in enumerate(circuit.next_bits) if bit.terms)
