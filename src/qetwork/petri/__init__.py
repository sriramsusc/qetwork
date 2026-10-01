"""Timed coloured Petri net models of the network protocols (design: docs/petri_net.md).

tcpn          -- the generic net kernel (places, timed coloured tokens, transitions)
entanglement  -- link generation + swapping on a linear chain (EntanglementDistribution)
params        -- per-node / per-edge parameters the net's arc expressions read
pair_state    -- what a pair token carries: DensityMatrix (exact) or BellDiagonal
bell          -- Bell-diagonal colour algebra (Pauli channels as XOR-convolutions)
validate      -- cross-check of the net against the event-driven simulator
"""
