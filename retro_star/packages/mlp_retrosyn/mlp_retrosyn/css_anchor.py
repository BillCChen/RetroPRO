"""Baseline-anchored conservative micro-replacement CSS sampler (anchor8).

The production CSS channel draws ``topk`` fragments as radius-``guardrail_r``
and radius-``cell_r`` balls around random seed bonds.  On small intermediates
several draws render to the same canonical fragment (deep-radius balls
degenerate into whole-molecule duplicates), so part of the nominal budget is
spent on repeated views of one substructure.  anchor8 keeps the production
draw untouched as the backbone and only reclaims those lost duplicate slots:
each reclaimed slot is filled by the distinct candidate with the largest
DICT-known full-product reaction yield, subject to a score margin and an
explicit reaction-retention gate.

Design rules inherited from the effective-yield campaign post-mortem:

- The production random draw is the default; nothing is replaced unless a
  slot was already wasted as a canonical duplicate.
- Replacement is scored by reactions the DICT can already replay on the
  current product, never by structural novelty alone, and never exceeds
  ``max_replace`` slots per call.
- An empty DICT makes every candidate score zero, so the output reduces to
  the production draw with strict canonical backfill.

The module is model agnostic.  ``known_reactions`` is a callback supplied by
the caller and returns the full-product canonical reactions currently known
for one fragment.
"""

import hashlib
import random

from rdkit import Chem

from .css_effective_yield import (
    enumerate_effective_yield_candidates,
    strict_backfill_fragments,
)
from .tp_free_tools import random_substructure


def _stable_seed(smiles, seed):
    digest = hashlib.sha256(smiles.encode("utf-8")).hexdigest()
    return int(seed) ^ int(digest[:16], 16)


def _canonical(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def production_draw(smiles, topk=8, cell_r=3, guardrail_r=7):
    """Mirror the production random draw: guardrail half plus cell half."""
    topk = int(topk)
    guardrail_num = topk // 2
    cell_num = topk - guardrail_num
    draws = random_substructure(smiles, r=guardrail_r, d=0, num=guardrail_num)
    draws = draws + random_substructure(smiles, r=cell_r, d=0, num=cell_num)
    return draws


def select_anchor_fragments(
    smiles,
    topk,
    known_reactions,
    max_replace=2,
    margin=1,
    cell_r=3,
    guardrail_r=7,
    include_triples=False,
    max_triples=0,
    seed=0,
):
    """Select fragments as production draw plus bounded micro-replacement.

    Steps: production draw; canonical dedup (slots lost to duplicates form
    the reclaim budget); score distinct candidates by DICT-known
    full-product reaction count; reclaim up to ``max_replace`` lost slots
    with candidates scoring at least ``margin``; strict-backfill the rest.
    Returns ``(fragments, metadata)`` where fragments are canonical-unique
    SMILES of length at most ``topk``.
    """
    topk = int(topk)
    if topk < 1:
        raise ValueError("topk must be positive")
    max_replace = max(int(max_replace), 0)
    margin = int(margin)
    rng = random.Random(_stable_seed(smiles, seed))

    draws = production_draw(
        smiles, topk, cell_r=cell_r, guardrail_r=guardrail_r
    )
    kept = []
    seen = set()
    lost_slots = 0
    for draw in draws:
        canonical = _canonical(draw)
        if canonical is None or canonical in seen:
            lost_slots += 1
            continue
        seen.add(canonical)
        kept.append(canonical)

    known_union = set()
    for fragment in kept:
        known_union |= set(known_reactions(fragment) or ())

    budget = min(lost_slots, max_replace, max(topk - len(kept), 0))
    replacements = []
    replacement_rows = []
    rejected_count = 0
    if budget > 0:
        candidates = enumerate_effective_yield_candidates(
            smiles,
            cell_r=cell_r,
            guardrail_r=guardrail_r,
            include_triples=include_triples,
            max_triples=max_triples,
        )
        scored = []
        for record in candidates:
            fragment = record["smiles"]
            if fragment in seen:
                continue
            reactions = set(known_reactions(fragment) or ())
            marginal = reactions - known_union
            scored.append((len(reactions), len(marginal), fragment, record, reactions))
        scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
        for score, marginal_count, fragment, record, reactions in scored:
            if len(replacements) >= budget:
                break
            if score < margin:
                rejected_count += 1
                continue
            # Retention gate: the known-reaction union must not shrink.
            new_union = known_union | reactions
            if len(new_union) < len(known_union):
                rejected_count += 1
                continue
            replacements.append(fragment)
            seen.add(fragment)
            known_union = new_union
            replacement_rows.append({
                "smiles": fragment,
                "families": sorted(record["families"]) + ["anchor_reclaim"],
                "known_reaction_count": score,
                "marginal_new_reaction_count": marginal_count,
                "atom_count": len(record["atoms"]),
                "bond_count": len(record["bonds"]),
            })

    merged = kept + replacements
    output = strict_backfill_fragments(
        smiles,
        merged,
        topk,
        ("r3", "r7"),
        cell_r=cell_r,
        guardrail_r=guardrail_r,
        max_triples=0,
    )
    output_canonicals = []
    output_seen = set()
    for fragment in output:
        canonical = _canonical(fragment)
        if canonical is None or canonical in output_seen:
            continue
        output_seen.add(canonical)
        output_canonicals.append(canonical)

    replacement_set = set(replacements)
    selected_rows = []
    for fragment in output_canonicals:
        if fragment in replacement_set:
            row = next(
                item for item in replacement_rows if item["smiles"] == fragment
            )
            selected_rows.append(row)
            continue
        mol = Chem.MolFromSmiles(fragment)
        families = ["anchor_base"] if fragment in kept else ["anchor_backfill"]
        selected_rows.append({
            "smiles": fragment,
            "families": families,
            "known_reaction_count": 0,
            "atom_count": mol.GetNumAtoms() if mol is not None else 0,
            "bond_count": mol.GetNumBonds() if mol is not None else 0,
        })

    metadata = {
        "sampler": "anchor8",
        "requested_count": topk,
        "selected_count": len(output_canonicals),
        "capacity_limited": len(output_canonicals) < topk,
        "production_draw_count": len(draws),
        "base_unique_count": len(kept),
        "lost_slot_count": lost_slots,
        "replace_budget": budget,
        "margin": margin,
        "replaced_count": len(replacements),
        "rejected_candidate_count": rejected_count,
        "known_reaction_union_count": len(known_union),
        "selected": selected_rows,
    }
    return output_canonicals, metadata
