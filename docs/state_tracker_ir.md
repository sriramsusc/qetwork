# State Tracker — Design / IR

> **Scope:** the **state_tracker** — the one shared object that holds all quantum
> state for a simulation. It is a *pure store*: it keeps quantum states and the
> bookkeeping of which qubits share a state. It knows **nothing** about gates or
> measurement — that physics lives in the **events** layer, which reads and writes
> the store. Density-matrix formalism first; state-vector / stabilizer later.

## Role

The kernel/`Timeline` is physics-agnostic (it schedules; it doesn't know quantum
mechanics). The **StateTracker** is its counterpart on the data side: it is
*semantics*-agnostic — it stores quantum states and answers "which state covers this
qubit," but it does not know what any operation *means*. Operations (gate application,
measurement) are **events** that `get` a state, do the linear algebra, and `set` it back.

There is exactly **one** StateTracker per simulation, held by the Timeline
(`timeline.state_tracker`) and referenced — never owned — by hardware, components, and
events. It must be shared because **entanglement is non-local**: a Bell pair's density
matrix describes qubits held on two different nodes, so it cannot live inside either.

## Decisions locked in

| Question | Decision |
|---|---|
| What it is | A **pure state store** — data + key bookkeeping only |
| Where the physics lives | In the **events** layer (gates, measurement), not here |
| Formalism | **Density matrix** first; state-vector / stabilizer later |
| Qubit identity | An integer **key** from a monotonic counter |
| Entanglement | A single state may cover **multiple keys** (a *joint state*) |
| API surface | `new` · `get` · `set` · `remove` |
| Merging separate states | Done by the **event** (store stays dumb) — *open, see below* |
| Instances | **One** per simulation, held by the Timeline |

## Representation

**Keys.** Every qubit is identified by a unique integer `key`, handed out by a
monotonic counter. Hardware that holds a qubit (a memory) and data carriers (a photon)
store the *key*, not the state itself.

**State object.** Each stored state is a small record:

```
State:
    matrix : ndarray        # density matrix, shape (2**n, 2**n), complex
    keys   : tuple[int]     # the n keys this state covers, in tensor-factor order
```

The `keys` ordering is essential: it says *which subsystem is which* inside `matrix`,
so an operation knows the index of the qubit it's acting on and how to trace out on
measurement.

**Store internals.**

```
StateTracker:
    states   : dict[int, State]   # key -> the State covering it
    _counter : int                # next key to hand out
```

A separable qubit maps to a 1-key state. Two entangled qubits map — *both* — to the
*same* joint State whose `keys == (a, b)`. Entangling is just: replace those keys'
entries so they point at one shared joint State.

## Keys and carriers — who holds what

The density matrix lives **only** in the tracker. Everything else holds a **key** — an
integer handle — never the matrix itself:

```
hardware / photon ──holds──▶ key (int) ──indexes──▶ StateTracker ──holds──▶ matrix
```

Think of the key as a coat-check ticket: the carrier holds the *ticket*; the "coat" (the
quantum state) hangs in the tracker. Two entangled qubits are two tickets pointing at the
*same* coat. This is **forced by non-locality** — an entangled matrix is shared by qubits
that may sit on different nodes, so it cannot live inside any single carrier or node; it
must live in the one shared tracker.

Consequences for how carriers are modelled:

- **Stored qubit — no object, just a key.** A quantum memory holds `key: int | None`
  (`None` = empty). There is **no `Qubit` class**: a stored qubit's *physical* attributes
  (coherence time, fidelity) belong to the **memory hardware**, so the qubit itself is
  nothing but its key.
- **Photon — a small data-carrier class.** A photon in flight has no hardware carrying it,
  so it carries its own physical attributes *plus* the key:
  `Photon(key: int, wavelength, encoding, ...)`. Produced by a source, transported by a
  channel, consumed by a detector. A passive data object — **not** an entity, and defined
  in its own leaf module (imported by both `hardware` and `components`).

Lifecycle of a key:

```
memory init:   memory.key = tracker.new()      # 2×2 |0><0| created
emit:          photon = Photon(key=memory.key); memory.key = None   # key moves, state unchanged
in flight:     operations act on photon.key via the tracker
lost/discard:  tracker.remove(key)
```

## API (the pure store)

```python
new(matrix=|0><0|) -> int
    # register a fresh single-qubit state (default |0><0|), return its key

get(key) -> State
    # return the (possibly joint) State covering `key`, including its full matrix
    # and ordered `keys` — so the caller knows this qubit's tensor index

set(keys, matrix) -> None
    # write a (possibly joint) state for the given ordered `keys`; re-points every
    # one of those keys at this single shared State. This is BOTH "save my change"
    # and "these keys are now entangled into one state."

remove(key) -> None
    # drop a key (qubit lost / discarded)
```

That is the whole surface. No `apply_gate`, no `measure` — by design.

## How events use it (physics stays outside)

**Single-qubit gate on key `k`:**
```
s = tracker.get(k)                 # joint State covering k
i = s.keys.index(k)                # k's tensor index
rho = apply U on s.matrix at i     # U rho U†   (event does the math)
tracker.set(s.keys, rho)           # save
```

**Two-qubit gate on keys `a, b`:**
```
sa, sb = tracker.get(a), tracker.get(b)
if sa is sb:                       # already entangled → one joint state
    joint_keys, joint = sa.keys, sa.matrix
else:                              # separate → merge (event does the kron)
    joint_keys = sa.keys + sb.keys
    joint = kron(sa.matrix, sb.matrix)
rho = apply gate on `joint` at the indices of a, b
tracker.set(joint_keys, rho)       # commit; a and b now share one state
```

**Measurement of key `k`:**
```
s = tracker.get(k)
probs = outcome probabilities from s.matrix at index of k
bit   = sample(probs, rng=timeline.rng)     # shared RNG for reproducibility
rho   = collapse + renormalize s.matrix
tracker.set(s.keys, rho)                     # or split k out into its own State
return bit
```

In every case the tracker only stores and re-points; the tracker never computes a
matrix product. That is what "one universal object that events access and modify" means.

## Open decision

**Where does *merging* live?** When a multi-qubit op spans qubits in *separate*
states, someone must tensor them into one joint state.

- **A — event merges (store fully dumb):** the event does the `kron` and calls
  `set(joint_keys, rho)`. The store never tensors. *(Current lean — matches "dumb
  universal object." The merge bookkeeping is factored into a shared helper in the
  events layer.)*
- **B — store offers `combine(keys)`:** a convenience that tensors separate states and
  re-points keys, so events just `get`/apply/`set`. Slightly less pure; avoids repeating
  merge logic.

## Deferred / out of scope

- **State-vector and stabilizer formalisms** — keep `new`/`get`/`set` stable so a
  `StateVector`-backed state slots in behind the same interface later.
- **Garbage collection of keys** — reclaiming keys of measured/lost qubits.
- **`combine` helper** (option B above) — only if event-side merging gets duplicated.
- Requires **numpy** (add to project deps when implemented).

## Cross-references

- Held by the Timeline as the single shared instance — see [[timeline_ir]].
- Gates and measurement are **events/operations** that use this store, not methods here.
- Hardware (memory) and carriers (photon) hold a qubit's **key**, not its state — see
  **Keys and carriers** above.
