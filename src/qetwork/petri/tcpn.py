"""Timed coloured Petri net (TCPN) kernel: places hold timestamped coloured tokens,
transitions fire instantaneously on a binding of their input arcs, and output arcs
add delays. No physics lives here -- see petri.entanglement for the repeater net.

Firing rule (Jensen's timed CPNs, the semantics CPN Tools implements):
  * a token is a (colour, timestamp) pair; it is AVAILABLE once the global clock
    reaches its timestamp;
  * a binding of transition t picks one token per input arc -- distinct tokens even
    when two arcs read the same place -- such that every arc filter and the guard
    hold; its ready time is max(clock, latest timestamp it consumes);
  * the clock advances to the earliest ready time over all bindings; among the
    bindings ready then, only the lowest priority value may fire (same-tick tiers,
    as in events.priority);
  * firing removes the consumed tokens and adds each output token with timestamp
    clock + delay. Time never passes inside a transition.

Stochastic choice (generalized stochastic Petri nets): every binding carries a
weight (default 1). Bindings tied on (ready time, priority) race, and the winner is
drawn with probability proportional to weight. Inside one conflict set -- bindings
competing for the same tokens -- this is exactly the GSPN random switch, so two
transitions with IDENTICAL presets and weights p, 1-p resolve as a Bernoulli(p)
outcome. A weight of 0 disables the binding. Ties between unrelated (concurrent)
bindings are broken by the same draw; that only orders them, it never changes
which of them eventually fire.

Arc filters (In.when) see the variables bound by EARLIER arcs, so a guard joining
tokens on a shared field prunes while it binds instead of enumerating the full
Cartesian product. Put the most selective arc first.
"""
# CITE tcpn | timed CPN: tokens carry timestamps, transitions fire instantly, output arcs add @+delay, the global clock jumps to the earliest enabling | K. Jensen & L. M. Kristensen, "Coloured Petri Nets: Modelling and Validation of Concurrent Systems", Springer (2009), Ch. 10 "Timed Coloured Petri Nets"
# CITE gspn | immediate transitions in conflict resolved by a random switch proportional to their weights | M. Ajmone Marsan, G. Conte, G. Balbo, "A class of generalized stochastic Petri nets for the performance evaluation of multiprocessor systems", ACM TOCS 2, 93-122 (1984)

import math
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np

from qetwork.seed import make_rng


@dataclass(slots=True, frozen=True)
class Token:
    color: object
    time: int            # available once the clock reaches it (integer ps, like the Timeline)


@dataclass(slots=True, frozen=True)
class Place:
    name: str
    colorset: type | tuple[type, ...]
    doc: str = ""


@dataclass(slots=True, frozen=True)
class In:
    """Input arc: binds one token of `place` to variable `var`.

    `when(binding, color)` filters candidates against the variables bound by the
    arcs listed before this one; `inscription` is display text for exports."""
    place: str
    var: str
    when: Callable[[Mapping, object], bool] | None = None
    inscription: str = ""


@dataclass(slots=True, frozen=True)
class Out:
    """One token an action produces: colour into `place`, available `delay` ps later."""
    place: str
    color: object
    delay: int = 0


@dataclass(slots=True, frozen=True)
class Context:
    """What an action may read besides its binding: the firing instant and the net's rng."""
    now: int
    rng: np.random.Generator


@dataclass(slots=True, frozen=True)
class Transition:
    name: str
    inputs: tuple[In, ...]
    outputs: Mapping[str, str]            # declared output place -> arc inscription (display text)
    action: Callable[[Mapping, Context], Iterable[Out]]
    guard: Callable[[Mapping], bool] | None = None
    weight: Callable[[Mapping], float] | None = None
    priority: int = 0
    doc: str = ""


@dataclass(slots=True, frozen=True)
class Firing:
    time: int
    transition: str
    binding: Mapping[str, object]
    produced: tuple[tuple[str, Token], ...]


def _colorset_name(cs) -> str:
    if isinstance(cs, tuple):
        return " | ".join(c.__name__ for c in cs)
    return cs.__name__


class TimedColoredPetriNet:
    def __init__(self, name: str, seed: int | None = None, record: bool = False) -> None:
        self.name = name
        self.clock = 0
        self.rng = make_rng(seed)
        self.places: dict[str, Place] = {}
        self.transitions: dict[str, Transition] = {}
        self._marking: dict[str, list[Token]] = {}
        self.fire_count: Counter[str] = Counter()
        self.trace: list[Firing] | None = [] if record else None
        self.observers: list[Callable[[Firing], None]] = []

    # --- structure ---

    def add_place(self, name: str, colorset, doc: str = "") -> Place:
        if name in self.places:
            raise ValueError(f"net {self.name!r} already has a place {name!r}")
        sets = colorset if isinstance(colorset, tuple) else (colorset,)
        if not sets or not all(isinstance(c, type) for c in sets):
            raise TypeError(f"place {name!r}: colorset must be a type or tuple of types, got {colorset!r}")
        place = Place(name, colorset, doc)
        self.places[name] = place
        self._marking[name] = []
        return place

    def add_transition(self, name: str, inputs, outputs: Mapping[str, str], action,
                       guard=None, weight=None, priority: int = 0, doc: str = "") -> Transition:
        if name in self.transitions:
            raise ValueError(f"net {self.name!r} already has a transition {name!r}")
        inputs = tuple(inputs)
        if not inputs:
            raise ValueError(f"transition {name!r} needs at least one input arc "
                             f"(a source transition would fire forever)")
        seen = set()
        for arc in inputs:
            if arc.place not in self.places:
                raise ValueError(f"transition {name!r} reads unknown place {arc.place!r}")
            if arc.var in seen:
                raise ValueError(f"transition {name!r} binds variable {arc.var!r} twice")
            seen.add(arc.var)
        unknown = [p for p in outputs if p not in self.places]
        if unknown:
            raise ValueError(f"transition {name!r} writes unknown places {unknown}")
        if not isinstance(priority, int) or isinstance(priority, bool):
            raise TypeError(f"priority must be an int, got {priority!r}")
        tr = Transition(name, inputs, MappingProxyType(dict(outputs)), action,
                        guard, weight, priority, doc)
        self.transitions[name] = tr
        return tr

    # --- marking ---

    def _check_color(self, place: str, color) -> None:
        cs = self.places[place].colorset
        if not isinstance(color, cs):
            raise TypeError(f"place {place!r} holds {_colorset_name(cs)}, "
                            f"got {type(color).__name__}: {color!r}")

    def put(self, place: str, color, at: int | None = None) -> Token:
        """Add a token to the marking (initial marking, or an external stimulus)."""
        if place not in self.places:
            raise ValueError(f"unknown place {place!r}")
        at = self.clock if at is None else at
        if not isinstance(at, int) or isinstance(at, bool):
            raise TypeError(f"token time must be an int (ps), got {type(at).__name__}: {at!r}")
        if at < self.clock:
            raise ValueError(f"cannot put a token in the past: {at} < clock {self.clock}")
        self._check_color(place, color)
        token = Token(color, at)
        self._marking[place].append(token)
        return token

    def tokens(self, place: str) -> tuple[Token, ...]:
        return tuple(self._marking[place])

    def colors(self, place: str) -> list:
        return [t.color for t in self._marking[place]]

    # --- firing rule ---

    def bindings(self, tr: Transition) -> Iterator[tuple[int, float, dict, tuple]]:
        """Every enabled binding of `tr` as (ready_time, weight, binding, consumed),
        consumed being (place, index) pairs into the current marking."""
        arcs = tr.inputs
        binding: dict = {}
        used: list[tuple[str, int]] = []

        def rec(i: int, ready: int):
            if i == len(arcs):
                if tr.guard is not None and not tr.guard(binding):
                    return
                w = 1.0 if tr.weight is None else float(tr.weight(binding))
                if not (w >= 0 and math.isfinite(w)):
                    raise ValueError(f"transition {tr.name!r}: weight must be finite and >= 0, got {w}")
                if w > 0:
                    yield ready, w, dict(binding), tuple(used)
                return
            arc = arcs[i]
            for idx, tok in enumerate(self._marking[arc.place]):
                if (arc.place, idx) in used:
                    continue
                if arc.when is not None and not arc.when(binding, tok.color):
                    continue
                binding[arc.var] = tok.color
                used.append((arc.place, idx))
                yield from rec(i + 1, max(ready, tok.time))
                used.pop()
                del binding[arc.var]

        yield from rec(0, self.clock)

    def step(self, until: int | None = None) -> Firing | None:
        """Fire one binding; None when nothing can fire (dead marking) or the next
        firing lies beyond `until` (the clock is then left untouched)."""
        best = None
        cands: list = []
        for tr in self.transitions.values():
            for ready, w, binding, consumed in self.bindings(tr):
                key = (ready, tr.priority)
                if best is None or key < best:
                    best, cands = key, [(w, tr, binding, consumed)]
                elif key == best:
                    cands.append((w, tr, binding, consumed))
        if best is None or (until is not None and best[0] > until):
            return None
        if len(cands) == 1:
            _, tr, binding, consumed = cands[0]
        else:
            ws = np.array([c[0] for c in cands])
            _, tr, binding, consumed = cands[int(self.rng.choice(len(cands), p=ws / ws.sum()))]

        self.clock = best[0]
        for place, idx in sorted(consumed, key=lambda pi: pi[1], reverse=True):
            del self._marking[place][idx]       # descending index: earlier deletions never shift later ones

        frozen = MappingProxyType(binding)
        produced = []
        for out in tr.action(frozen, Context(self.clock, self.rng)):
            if out.place not in tr.outputs:
                raise ValueError(f"transition {tr.name!r} produced into undeclared place {out.place!r}")
            if not isinstance(out.delay, int) or isinstance(out.delay, bool) or out.delay < 0:
                raise ValueError(f"transition {tr.name!r}: delay must be an int >= 0, got {out.delay!r}")
            self._check_color(out.place, out.color)
            token = Token(out.color, self.clock + out.delay)
            self._marking[out.place].append(token)
            produced.append((out.place, token))

        self.fire_count[tr.name] += 1
        firing = Firing(self.clock, tr.name, frozen, tuple(produced))
        if self.trace is not None:
            self.trace.append(firing)
        for obs in self.observers:
            obs(firing)
        return firing

    def run(self, until: int | None = None, stop: Callable[["TimedColoredPetriNet"], bool] | None = None,
            max_steps: int = 10_000_000) -> int:
        """Fire until `stop(net)` holds, the marking is dead, or the clock would pass `until`."""
        steps = 0
        while stop is None or not stop(self):
            if steps >= max_steps:
                raise RuntimeError(f"net {self.name!r} exceeded max_steps={max_steps} "
                                   f"(livelock, or raise the bound)")
            if self.step(until) is None:
                break
            steps += 1
        return steps

    # --- export ---

    def to_dot(self) -> str:
        """Graphviz source: places as ellipses (name : colour set), transitions as boxes."""
        def q(s: str) -> str:
            return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'
        lines = [f"digraph {q(self.name)} {{", "  rankdir=LR;", '  node [fontname="Helvetica"];']
        for p in self.places.values():
            lines.append(f"  {q('p:' + p.name)} [shape=ellipse, label={q(p.name + chr(10) + ': ' + _colorset_name(p.colorset))}];")
        for t in self.transitions.values():
            lines.append(f"  {q('t:' + t.name)} [shape=box, style=filled, fillcolor=\"#eeeeee\", "
                         f"label={q(t.name + chr(10) + 'prio ' + str(t.priority))}];")
            for arc in t.inputs:
                lines.append(f"  {q('p:' + arc.place)} -> {q('t:' + t.name)} [label={q(arc.inscription or arc.var)}];")
            for place, ins in t.outputs.items():
                lines.append(f"  {q('t:' + t.name)} -> {q('p:' + place)} [label={q(ins)}];")
        lines.append("}")
        return "\n".join(lines)

    def to_mermaid(self) -> str:
        """Mermaid flowchart (renders on GitHub): circles are places, boxes transitions."""
        def lab(s: str) -> str:
            return s.replace('"', "'")
        ids = {}
        lines = ["flowchart LR"]
        for i, p in enumerate(self.places.values()):
            ids["p:" + p.name] = f"P{i}"
            lines.append(f'  P{i}(("{lab(p.name)}<br/>{lab(_colorset_name(p.colorset))}"))')
        for i, t in enumerate(self.transitions.values()):
            ids["t:" + t.name] = f"T{i}"
            lines.append(f'  T{i}["{lab(t.name)}"]')
        for t in self.transitions.values():
            for arc in t.inputs:
                lines.append(f'  {ids["p:" + arc.place]} -->|"{lab(arc.inscription or arc.var)}"| {ids["t:" + t.name]}')
            for place, ins in t.outputs.items():
                lines.append(f'  {ids["t:" + t.name]} -->|"{lab(ins)}"| {ids["p:" + place]}')
        return "\n".join(lines)
