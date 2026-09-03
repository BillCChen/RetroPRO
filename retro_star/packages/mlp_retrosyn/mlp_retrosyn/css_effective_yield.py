"""Effective-reaction-yield sampler for a strict CSS cardinality budget.

The selector combines radius-3 cells, radius-7 cells, adjacent radius-3
pairs, and connected radius-3 triples.  Cached rules provide an immediately
available estimate of useful yield: each rule is applied to the full product
and the selector greedily maximizes the union of canonical reaction outputs.
Reserved exploration slots are filled by structurally novel candidates so
that a growing DICT cannot suppress unseen chemistry.

The module is model agnostic.  ``known_reactions`` is a callback supplied by
the caller and returns the full-product canonical reactions currently known
for one fragment.
"""

import hashlib
import math
import random

from rdkit import Chem, rdBase

from .css_hierarchical import (
    _build_cells,
    _cell_adjacency,
    _complete_aromatic,
    _connected_triples,
    _internal_bonds,
    _render,
)


FAMILY_ORDER = ("r3", "r7", "pair", "triple")
BALANCED_SIZE_TARGETS = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.55)


def _stable_seed(smiles, seed):
    digest = hashlib.sha256(smiles.encode("utf-8")).hexdigest()
    return int(seed) ^ int(digest[:16], 16)


def _canonical_fragment(mol, atoms):
    fragment = _render(mol, atoms)
    # Some fused aromatic cuts remain chemically incomplete even after ring
    # completion. They are expected candidate rejections, not planner errors.
    with rdBase.BlockLogs():
        parsed = Chem.MolFromSmiles(fragment)
    if parsed is None or "." in fragment:
        return None
    return Chem.MolToSmiles(parsed, canonical=True, isomericSmiles=True)


def enumerate_effective_yield_candidates(
    smiles,
    cell_r=3,
    guardrail_r=7,
    include_triples=True,
    max_triples=256,
):
    """Enumerate unique parseable candidates across four CSS families.

    Candidates that render to the same canonical fragment are merged.  The
    first deterministic atom embedding is retained for structural novelty;
    all contributing family labels are preserved.
    """
    if cell_r < 0 or guardrail_r < 0:
        raise ValueError("fragment radii must be non-negative")
    if max_triples < 0:
        raise ValueError("max_triples must be non-negative")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("cannot parse SMILES: %s" % smiles)
    if mol.GetNumBonds() == 0:
        canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        return [{
            "smiles": canonical,
            "atoms": frozenset(range(mol.GetNumAtoms())),
            "bonds": frozenset(),
            "families": frozenset(("whole",)),
        }]

    seed_bonds = list(mol.GetBonds())
    rings, r3_cells = _build_cells(mol, seed_bonds, cell_r)
    _, r7_cells = _build_cells(mol, seed_bonds, guardrail_r)
    adjacency = _cell_adjacency(mol, r3_cells)
    records = {}

    def add_candidate(atoms, family):
        atoms = set(atoms) | _complete_aromatic(rings, set(atoms))
        fragment = _canonical_fragment(mol, atoms)
        if fragment is None:
            return
        existing = records.get(fragment)
        if existing is None:
            records[fragment] = {
                "smiles": fragment,
                "atoms": frozenset(atoms),
                "bonds": frozenset(_internal_bonds(mol, atoms)),
                "families": {family},
            }
        else:
            existing["families"].add(family)

    for atoms in r3_cells:
        add_candidate(atoms, "r3")
    for atoms in r7_cells:
        add_candidate(atoms, "r7")

    for i in range(len(r3_cells)):
        for j in adjacency[i]:
            if i < j:
                add_candidate(r3_cells[i] | r3_cells[j], "pair")

    if include_triples:
        triples = sorted(_connected_triples(r3_cells, adjacency))
        if max_triples > 0 and len(triples) > max_triples:
            step = float(len(triples)) / float(max_triples)
            triples = [triples[int(i * step)] for i in range(max_triples)]
        for triple in triples:
            atoms = set()
            for cell_index in triple:
                atoms |= r3_cells[cell_index]
            add_candidate(atoms, "triple")

    output = []
    for fragment in sorted(records):
        record = records[fragment]
        record["families"] = frozenset(record["families"])
        output.append(record)
    if not output:
        canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        output.append({
            "smiles": canonical,
            "atoms": frozenset(range(mol.GetNumAtoms())),
            "bonds": frozenset(_internal_bonds(mol, set(range(mol.GetNumAtoms())))),
            "families": frozenset(("whole",)),
        })
    return output


def strict_backfill_fragments(
    smiles,
    existing,
    topk,
    families,
    cell_r=3,
    guardrail_r=7,
    max_triples=256,
):
    """Backfill a sampler draw to a canonical-unique cardinality budget."""
    selected = []
    seen = set()
    for fragment in existing:
        mol = Chem.MolFromSmiles(fragment)
        if mol is None:
            continue
        canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        if canonical not in seen:
            seen.add(canonical)
            selected.append(canonical)
    if len(selected) >= topk:
        return selected[:topk]

    allowed = set(families)
    candidates = enumerate_effective_yield_candidates(
        smiles,
        cell_r=cell_r,
        guardrail_r=guardrail_r,
        include_triples="triple" in allowed,
        max_triples=max_triples,
    )
    remaining = [
        record["smiles"]
        for record in candidates
        if record["families"] & allowed and record["smiles"] not in seen
    ]
    random.shuffle(remaining)
    for fragment in remaining:
        seen.add(fragment)
        selected.append(fragment)
        if len(selected) == topk:
            break
    return selected



def cached_strict_fragments(
    smiles,
    topk,
    is_cached,
    cache_score,
    exploration_slots=2,
    cell_r=3,
    guardrail_r=7,
    exploration_oversample=4,
):
    """Cache-first fragment draw over the enumerated candidate pool.

    Cached fragments fill ``topk - exploration_slots`` slots, ranked by
    cache_score descending then SMILES; the remaining slots go to fresh
    random draws that are neither cached nor already selected.  With an
    empty cache the draw degrades to a plain random pool.
    """
    if topk < 1:
        raise ValueError("topk must be positive")
    if exploration_slots < 0:
        raise ValueError("exploration_slots must be non-negative")
    exploration_slots = min(exploration_slots, topk)
    candidates = enumerate_effective_yield_candidates(
        smiles,
        cell_r=cell_r,
        guardrail_r=guardrail_r,
        include_triples=False,
        max_triples=0,
    )
    cached = [record["smiles"] for record in candidates
              if is_cached(record["smiles"])]
    cached.sort(key=lambda fragment: (-cache_score(fragment), fragment))
    cached_budget = topk - exploration_slots
    selected = list(dict.fromkeys(cached[:cached_budget]))
    seen = set(selected)
    need = topk - len(selected)
    fresh = []
    if need > 0:
        from .tp_free_tools import random_substructure
        draws = random_substructure(
            smiles, r=cell_r, d=0,
            num=max(need, exploration_slots) * exploration_oversample,
        )
        for fragment in draws:
            mol = Chem.MolFromSmiles(fragment)
            if mol is None:
                continue
            canonical = Chem.MolToSmiles(mol, canonical=True,
                                         isomericSmiles=True)
            if canonical in seen or is_cached(canonical):
                continue
            seen.add(canonical)
            fresh.append(canonical)
            if len(fresh) == need:
                break
    return selected + fresh

def _family_label(record):
    for family in FAMILY_ORDER:
        if family in record["families"]:
            return family
    return sorted(record["families"])[0]


def _choose_best(frontier, rng):
    best_key = max(item[0] for item in frontier)
    tied = [item[1] for item in frontier if item[0] == best_key]
    tied.sort(key=lambda record: record["smiles"])
    return rng.choice(tied)


def _structural_choice(
    remaining,
    covered_atoms,
    covered_bonds,
    used_families,
    rng,
    target_atom_count,
    desired_fraction,
    profile,
):
    frontier = []
    for record in remaining:
        new_bonds = len(record["bonds"] - covered_bonds)
        new_atoms = len(record["atoms"] - covered_atoms)
        size = max(len(record["bonds"]), 1)
        normalized_bond_novelty = new_bonds / math.sqrt(float(size))
        family_bonus = int(not (record["families"] & used_families))
        unknown_bonus = int(not record["known_reactions"])
        if profile == "legacy":
            key = (
                family_bonus,
                unknown_bonus,
                round(normalized_bond_novelty, 12),
                new_bonds,
                new_atoms,
                -len(record["atoms"]),
            )
        elif profile == "balanced":
            atom_fraction = len(record["atoms"]) / float(target_atom_count)
            size_distance = abs(atom_fraction - desired_fraction)
            key = (
                -round(size_distance, 12),
                new_bonds,
                new_atoms,
                family_bonus,
                unknown_bonus,
            )
        else:
            raise ValueError("unknown exploration profile: %s" % profile)
        frontier.append((key, record))
    return _choose_best(frontier, rng)


def select_effective_yield_fragments(
    smiles,
    topk,
    known_reactions,
    exploration_slots=2,
    cell_r=3,
    guardrail_r=7,
    include_triples=True,
    max_triples=256,
    seed=0,
    exploration_profile="balanced",
):
    """Select a strict unique fragment set under an effective-yield objective.

    The known portion is selected by marginal full-product reaction yield.
    At least ``exploration_slots`` are reserved for structurally novel unseen
    candidates when capacity permits.  If fewer than ``topk`` unique
    candidates exist, all candidates are returned and ``capacity_limited`` is
    set in the metadata.
    """
    topk = int(topk)
    if topk < 1:
        raise ValueError("topk must be positive")
    exploration_slots = min(max(int(exploration_slots), 0), topk)
    rng = random.Random(_stable_seed(smiles, seed))
    target_mol = Chem.MolFromSmiles(smiles)
    target_atom_count = max(target_mol.GetNumAtoms(), 1)
    candidates = enumerate_effective_yield_candidates(
        smiles,
        cell_r=cell_r,
        guardrail_r=guardrail_r,
        include_triples=include_triples,
        max_triples=max_triples,
    )
    for record in candidates:
        reactions = known_reactions(record["smiles"])
        record["known_reactions"] = frozenset(reactions or ())

    limit = min(topk, len(candidates))
    known_budget = max(limit - exploration_slots, 0)
    selected = []
    selected_smiles = set()
    covered_reactions = set()
    covered_atoms = set()
    covered_bonds = set()
    used_families = set()

    while len(selected) < known_budget:
        frontier = []
        for record in candidates:
            if record["smiles"] in selected_smiles:
                continue
            marginal = record["known_reactions"] - covered_reactions
            if not marginal:
                continue
            key = (
                len(marginal),
                len(record["known_reactions"]),
                int("r7" in record["families"]),
                -len(record["atoms"]),
            )
            frontier.append((key, record))
        if not frontier:
            break
        chosen = _choose_best(frontier, rng)
        selected.append(chosen)
        selected_smiles.add(chosen["smiles"])
        covered_reactions |= chosen["known_reactions"]
        covered_atoms |= chosen["atoms"]
        covered_bonds |= chosen["bonds"]
        used_families |= chosen["families"]

    while len(selected) < limit:
        remaining = [
            record for record in candidates
            if record["smiles"] not in selected_smiles
        ]
        if not remaining:
            break
        size_index = len(selected) % len(BALANCED_SIZE_TARGETS)
        chosen = _structural_choice(
            remaining,
            covered_atoms,
            covered_bonds,
            used_families,
            rng,
            target_atom_count,
            BALANCED_SIZE_TARGETS[size_index],
            exploration_profile,
        )
        selected.append(chosen)
        selected_smiles.add(chosen["smiles"])
        covered_reactions |= chosen["known_reactions"]
        covered_atoms |= chosen["atoms"]
        covered_bonds |= chosen["bonds"]
        used_families |= chosen["families"]

    metadata = {
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "requested_count": topk,
        "capacity_limited": len(candidates) < topk,
        "known_reaction_union_count": len(covered_reactions),
        "exploration_profile": exploration_profile,
        "selected": [
            {
                "smiles": record["smiles"],
                "families": sorted(record["families"]),
                "known_reaction_count": len(record["known_reactions"]),
                "atom_count": len(record["atoms"]),
                "bond_count": len(record["bonds"]),
            }
            for record in selected
        ],
    }
    return [record["smiles"] for record in selected], metadata
