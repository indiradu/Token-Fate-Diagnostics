"""Default-off repairs of the block KL word commit (``K-block`` of ``paper_decode``): adjacent-identical veto helper.

The confidence floor and the persistence condition live in ``paper_decode`` (they are one boolean mask each); the veto needs a small
deterministic loop, kept here so it can be unit tested without a model.
"""
from __future__ import annotations

from typing import Collection, Sequence


def veto_adjacent_identical(
    positions: Sequence[int],
    tokens: Sequence[int],
    confidence: Sequence[float],
    exempt: Collection[int] = (),
) -> list[int]:
    """Indices (into the inputs) that survive the adjacent-identical veto, ascending.

    ``positions`` are the slots selected in one call (any order; adjacency means ``p`` and ``p + 1`` both selected), ``tokens`` the token
    each proposes and ``confidence`` the proposal confidence. Procedure (deterministic): repeatedly take the adjacent pair with the
    smallest position ``p`` whose two members propose the same non-exempt token, drop the lower-confidence member (ties: drop the later
    position), and repeat until no such pair remains. Adjacency is only among the selected slots of this call, never against
    already-written neighbours. Tokens in ``exempt`` (padding ids) are never vetoed. Dropping the lower member of a pair cannot empty the
    set, so at least the most confident slot always survives.
    """
    if not (len(positions) == len(tokens) == len(confidence)):
        raise ValueError("positions, tokens and confidence must have the same length")
    if len(set(positions)) != len(positions):
        raise ValueError("positions must be distinct")
    order = sorted(range(len(positions)), key=lambda i: positions[i])
    alive = {positions[i]: i for i in order}
    exempt = set(exempt)
    while True:
        clash = None
        for p in sorted(alive):
            q = p + 1
            if q in alive and tokens[alive[p]] == tokens[alive[q]] and tokens[alive[p]] not in exempt:
                clash = (p, q)
                break
        if clash is None:
            break
        p, q = clash
        drop = p if confidence[alive[p]] < confidence[alive[q]] else q       # tie: drop the later position
        del alive[drop]
    return sorted(alive.values(), key=lambda i: positions[i])
