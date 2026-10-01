"""Bell-diagonal colour algebra: a two-qubit pair as a probability vector over its
Pauli frame, so every Pauli channel and every entanglement swap is an XOR-convolution.

A Bell-diagonal state is  rho = sum_k p[k] (P_k (x) I) |Phi+><Phi+| (P_k (x) I)  with
index k = 2x + z labelling P_k = X^x Z^z:

    k = 0 (I) -> Phi+    k = 1 (Z) -> Phi-    k = 2 (X) -> Psi+    k = 3 (Y) -> Psi-

Multiplying Paulis XORs their labels (Y ~ XZ up to a phase that cancels in the
projector), so with (p * q)[k] = sum_i p[i] q[i ^ k]:
  * a Pauli channel q acting on EITHER half:   p -> p * q
    ((I (x) P)|Phi+> = (P^T (x) I)|Phi+> and P^T = +-P, so the side never matters);
  * an ideal swap of p and q, Pauli-frame corrected at the far end:   p * q;
  * Phi+ fidelity:   p[0].
Labels are tracked in the CORRECTED frame: the destination's X^m2 Z^m1 correction
is implicit, only its gate noise is applied.

Exact for Pauli noise (T2 dephasing, depolarizing gates). T1 amplitude damping is
not a Pauli channel; it enters as its Pauli twirl, which keeps the exact diagonal of
the Pauli transfer matrix (coherences decay as e^(-t/2T1), populations as e^(-t/T1))
and drops the non-unital drift toward |0>. A single damping step on one half of a
Bell-diagonal pair still gets the Phi+ fidelity exactly; across swaps the error is
second order in gamma, so it is small only while storage times stay << T1 (at T1 = 2 ms
on a 3-hop, ~2 ms chain it already costs ~0.05 in fidelity). Exact whenever T1 = inf.
Coherent errors (residual rotations, a source phase) are not representable here; for
either, use pair_state.DensityMatrix.
"""
# CITE bell-diag-swap | Bell-diagonal pairs stay Bell-diagonal under swapping; output coefficients are the Klein-group convolution of the inputs | W. Dur, H.-J. Briegel, J. I. Cirac, P. Zoller, "Quantum repeaters based on entanglement purification", PRA 59, 169 (1999)
# CITE pauli-twirl-ad | Pauli twirl of amplitude damping: p_x = p_y = gamma/4, p_z = (1 - sqrt(1-gamma))^2/4 | J. Ghosh, A. G. Fowler, M. R. Geller, "Surface code with decoherence: An analysis of three superconducting architectures", PRA 86, 062318 (2012)

import math

from qetwork.petri.params import t2_star

Bell = tuple[float, float, float, float]

I_, Z_, X_, Y_ = 0, 1, 2, 3                 # label order: Phi+, Phi-, Psi+, Psi-
PHI_PLUS: Bell = (1.0, 0.0, 0.0, 0.0)
IDENTITY: Bell = PHI_PLUS                   # the identity channel has the same vector


def validate(p: Bell, what: str = "Bell vector") -> Bell:
    """Coerce to a 4-tuple of floats and check it is a probability vector."""
    p = tuple(float(x) for x in p)
    if len(p) != 4:
        raise ValueError(f"{what} needs 4 Pauli-frame probabilities, got {len(p)}")
    if any(x < -1e-12 for x in p) or not math.isclose(sum(p), 1.0, abs_tol=1e-9):
        raise ValueError(f"{what} is not a probability vector: {p}")
    return p


def convolve(p: Bell, q: Bell) -> Bell:
    """XOR-convolution on the Klein four-group: channel application AND ideal swap."""
    return tuple(sum(p[i] * q[i ^ k] for i in range(4)) for k in range(4))


def fidelity(p: Bell) -> float:
    """<Phi+| rho |Phi+>."""
    return p[I_]


def werner(visibility: float) -> Bell:
    """V |Phi+><Phi+| + (1-V) I/4 -- the SFWM source's raw pair at phase 0."""
    if not 0 <= visibility <= 1:
        raise ValueError(f"visibility must be in [0,1], got {visibility}")
    r = (1 - visibility) / 4
    return (visibility + r, r, r, r)


def depolarizing_1q(p: float) -> Bell:
    """One-qubit depolarizing: X, Y, Z each with p/3 (err_channels.depolarize, k=1)."""
    if not 0 <= p <= 1:
        raise ValueError(f"p must be in [0,1], got {p}")
    return (1 - p, p / 3, p / 3, p / 3)


def bsm_channel(p_1q: float, p_2q: float) -> Bell:
    """Frame error a gate-based BSM (CNOT, H, Z-measure both -- RepeaterNode.bsm) leaves
    on the swapped pair. A Pauli on the control flips m1 iff it has a Z part (H turns
    it into X), on the target flips m2 iff it has an X part; a wrong m1 leaves a Z on
    the far half, a wrong m2 an X.
      * 2q depolarizing after the CNOT, each of the 15 non-identity Paulis at p/15:
        every frame label is hit by 4 of the 16 (P_c, P_t) pairs, (I, I) included
        -> identity 1 - 4p/5, each of Z, X, Y 4p/15;
      * 1q depolarizing after the H: X or Y on the control flips m1 -> Z w.p. 2p/3."""
    for name, p in (("p_1q", p_1q), ("p_2q", p_2q)):
        if not 0 <= p <= 1:
            raise ValueError(f"{name} must be in [0,1], got {p}")
    two = (1 - 4 * p_2q / 5, 4 * p_2q / 15, 4 * p_2q / 15, 4 * p_2q / 15)
    one = (1 - 2 * p_1q / 3, 2 * p_1q / 3, 0.0, 0.0)
    return convolve(two, one)


def memory_channel(elapsed: float, t1: float, t2: float) -> Bell:
    """Storage noise on one half over `elapsed` ps, split exactly as MemoryQubit.decohere:
    amplitude damping at T1 (Pauli-twirled) then pure dephasing at T2* with
    1/T2* = 1/T2 - 1/(2 T1)."""
    if elapsed < 0:
        raise ValueError(f"elapsed time must be non-negative, got {elapsed}")
    if elapsed == 0:
        return IDENTITY
    out = IDENTITY
    if not math.isinf(t1):
        gamma = 1 - math.exp(-elapsed / t1)
        s = math.sqrt(1 - gamma)
        out = ((1 + s) ** 2 / 4, (1 - s) ** 2 / 4, gamma / 4, gamma / 4)
    ts = t2_star(t1, t2)
    if not math.isinf(ts):
        lam = (1 - math.exp(-elapsed / ts)) / 2
        out = convolve(out, (1 - lam, lam, 0.0, 0.0))
    return out


def apply(p: Bell, *channels: Bell) -> Bell:
    """Apply Pauli channels in sequence (order is irrelevant: the group is abelian)."""
    for ch in channels:
        p = convolve(p, ch)
    return p

