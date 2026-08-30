"""Coverage-greedy hierarchical substructure sampler for CSS (paircov).

The production CSS channel draws large fragments as isotropic radius-R
balls around a random seed bond; on small intermediates those balls
degenerate into near-whole-molecule duplicates. paircov replaces the
large half of the sampling budget with connected unions of two adjacent
radius-`cell_r` "cells", selected by a greedy maximum-marginal-coverage
rule so the emitted large fragments cover as many distinct atoms as
possible instead of piling onto the same region.

Cell adjacency (atom-set intersection or direct bonding) guarantees that
every union is connected. Aromatic-ring completion follows the same
convention as `random_substructure`: any all-aromatic ring touched by
the distance-driven atom set is included whole, non-recursively.

Only the large half lives here. The small half stays wired through the
original `random_substructure`, so the ablation against production
isolates a single variable: what replaces the large-radius channel.
"""
import random
from collections import deque
from rdkit import Chem


def _bond_distances(mol, bond):
    """Shortest graph distance from the seed bond (both ends = 0)."""
    dist = {bond.GetBeginAtomIdx(): 0, bond.GetEndAtomIdx(): 0}
    queue = deque(dist)
    while queue:
        idx = queue.popleft()
        for nb in mol.GetAtomWithIdx(idx).GetNeighbors():
            n = nb.GetIdx()
            if n not in dist:
                dist[n] = dist[idx] + 1
                queue.append(n)
    return dist


def _aromatic_rings(mol):
    rings = []
    for ring in mol.GetRingInfo().AtomRings():
        rs = set(ring)
        if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in rs):
            rings.append(rs)
    return rings


def _complete_aromatic(rings, atoms):
    completion = set()
    for ring in rings:
        if ring & atoms:
            completion.update(ring)
    return completion


def _build_cells(mol, seed_bonds, cell_r):
    """One cell per seed bond: sphere(cell_r) plus aromatic completion."""
    rings = _aromatic_rings(mol)
    cells = []
    for bond in seed_bonds:
        dist = _bond_distances(mol, bond)
        atoms = {i for i, di in dist.items() if di <= cell_r}
        atoms |= _complete_aromatic(rings, atoms)
        cells.append(atoms)
    return rings, cells


def _cell_adjacency(mol, cells):
    """Cells are adjacent when they share an atom or are directly bonded."""
    adj = [set() for _ in cells]
    atom_to_cells = {}
    for ci, atoms in enumerate(cells):
        for a in atoms:
            atom_to_cells.setdefault(a, []).append(ci)
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        for ca in atom_to_cells.get(a, ()):  # pragma: no branch
            for cb in atom_to_cells.get(b, ()):
                if ca != cb:
                    adj[ca].add(cb)
                    adj[cb].add(ca)
    return adj


def _render(mol, atoms):
    return Chem.MolFragmentToSmiles(
        mol, atomsToUse=sorted(atoms), canonical=True, isomericSmiles=True)


def _dedup_keep_order(items):
    seen, out = set(), []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def paircov_large_fragments(smiles, cell_r=3, num=4, seed=None):
    """Sample up to `num` large fragments as coverage-greedy cell unions.

    Cells are radius-`cell_r` balls (plus aromatic completion) around
    every bond of the molecule; adjacent cell pairs form the candidate
    pool. Pairs whose rendered fragment cannot be parsed back by RDKit
    (partially cut fused rings, ~7% under the production convention)
    are removed from the pool up front: downstream would drop them
    silently, so filtering here recovers the sampling budget without
    changing what reaches the model. Selection is greedy: each round
    picks a pool entry whose atom union adds the most previously
    uncovered atoms, breaking ties uniformly at random. Once no entry
    adds any new atom, the remaining budget is filled by uniform random
    draws from the pool. Returns unique canonical fragment SMILES
    (length <= num); falls back to the whole molecule when the pool is
    empty.
    """
    if cell_r < 0:
        raise ValueError("cell_r must be non-negative")
    rng = random.Random(seed) if seed is not None else random
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("cannot parse SMILES: %s" % smiles)
    if mol.GetNumBonds() == 0:
        return [Chem.MolToSmiles(mol)]

    rings, cells = _build_cells(mol, list(mol.GetBonds()), cell_r)
    adj = _cell_adjacency(mol, cells)

    pool = []  # (atoms, frag_smiles); atoms include aromatic completion
    for i in range(len(cells)):
        for j in adj[i]:
            if i < j:
                atoms = cells[i] | cells[j]
                atoms |= _complete_aromatic(rings, atoms)
                frag = _render(mol, atoms)
                if Chem.MolFromSmiles(frag) is not None:
                    pool.append((atoms, frag))
    if not pool:
        return [Chem.MolToSmiles(mol)]

    covered = set()
    frags = []
    for _ in range(num):
        best_gain = 0
        best = []
        for atoms, frag in pool:
            gain = len(atoms - covered)
            if gain > best_gain:
                best_gain = gain
                best = [(atoms, frag)]
            elif gain == best_gain:
                best.append((atoms, frag))
        if best_gain == 0:
            break
        atoms, frag = rng.choice(best)
        covered |= atoms
        frags.append(frag)

    if len(frags) < num:  # molecule fully covered: top up at random
        for atoms, frag in rng.choices(pool, k=num - len(frags)):
            frags.append(frag)

    out = _dedup_keep_order(frags)
    if not out:
        out = [Chem.MolToSmiles(mol)]
    return out[:num]


def _greedy_select(candidates, covered, num, rng, topk=1):
    """Greedy max-marginal-coverage selection.

    candidates: list of (atoms, payload). Picks up to `num` entries;
    each round ranks candidates by marginal new-atom gain and chooses
    uniformly at random among the top-`topk` (topk=1 = argmax with
    uniform tie-break). Stops early when no candidate adds a new atom.
    Returns (chosen payloads, updated covered set).
    """
    chosen = []
    covered = set(covered)
    for _ in range(num):
        ranked = []
        for atoms, payload in candidates:
            gain = len(atoms - covered)
            if gain > 0:
                ranked.append((gain, atoms, payload))
        if not ranked:
            break
        ranked.sort(key=lambda x: -x[0])
        cutoff = ranked[min(topk, len(ranked)) - 1][0]
        frontier = [(a, p) for g, a, p in ranked if g >= cutoff]
        atoms, payload = rng.choice(frontier)
        covered |= atoms
        chosen.append(payload)
    return chosen, covered


def fullcov_fragments(smiles, cell_r=3, num=8, seed=None, topk=1):
    """Coverage-greedy fragments for both halves of the budget.

    Small half: single radius-`cell_r` cells selected greedily by
    marginal atom coverage (instead of the random seed bonds used by
    production and paircov). Large half: adjacent-cell pairs whose
    greedy selection continues from the small half's covered set, so
    large fragments are steered away from regions the small fragments
    already cover. Both pools are deduplicated by rendered fragment and
    filtered for RDKit re-parseability up front. `topk` trades coverage
    for diversity: picks are uniform among the top-`topk` candidates by
    marginal gain. Returns unique canonical fragment SMILES
    (length <= num); falls back to the whole molecule when nothing can
    be composed.
    """
    if cell_r < 0:
        raise ValueError("cell_r must be non-negative")
    rng = random.Random(seed) if seed is not None else random
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("cannot parse SMILES: %s" % smiles)
    if mol.GetNumBonds() == 0:
        return [Chem.MolToSmiles(mol)]

    rings, cells = _build_cells(mol, list(mol.GetBonds()), cell_r)

    def make_pool(atom_sets):
        pool, seen = [], set()
        for atoms in atom_sets:
            frag = _render(mol, atoms)
            if frag in seen:
                continue
            seen.add(frag)
            if Chem.MolFromSmiles(frag) is not None:
                pool.append((atoms, frag))
        return pool

    cell_pool = make_pool(cells)
    n_small = num // 2
    small_chosen, covered = _greedy_select(cell_pool, set(), n_small, rng, topk)

    adj = _cell_adjacency(mol, cells)
    pair_sets = []
    for i in range(len(cells)):
        for j in adj[i]:
            if i < j:
                atoms = cells[i] | cells[j]
                pair_sets.append(atoms | _complete_aromatic(rings, atoms))
    pair_pool = make_pool(pair_sets)
    large_chosen, covered = _greedy_select(
        pair_pool, covered, num - n_small, rng, topk)

    frags = small_chosen + large_chosen
    if len(frags) < num:  # saturated: top up uniformly from both pools
        rest = cell_pool + pair_pool
        if rest:
            for atoms, frag in rng.choices(rest, k=num - len(frags)):
                frags.append(frag)

    out = _dedup_keep_order(frags)
    if not out:
        out = [Chem.MolToSmiles(mol)]
    return out[:num]


def _internal_bonds(mol, atoms):
    """Indices of bonds with both endpoints inside the atom set."""
    out = set()
    for b in mol.GetBonds():
        if b.GetBeginAtomIdx() in atoms and b.GetEndAtomIdx() in atoms:
            out.add(b.GetIdx())
    return out


def bondcov_large_fragments(smiles, cell_r=3, num=4, seed=None):
    """Like paircov, but the greedy gain counts new internal bonds.

    The candidate pool is identical to paircov (adjacent radius-`cell_r`
    cell pairs, deduplicated by rendered fragment, filtered for RDKit
    re-parseability). The greedy objective is the number of newly
    covered bonds (both endpoints inside the fragment) rather than
    newly covered atoms: reaction centers are bonds, so this aligns the
    selection pressure with the probability that some emitted fragment
    contains the true first disconnection. Returns unique canonical
    fragment SMILES (length <= num); falls back to the whole molecule
    when the pool is empty.
    """
    if cell_r < 0:
        raise ValueError("cell_r must be non-negative")
    rng = random.Random(seed) if seed is not None else random
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("cannot parse SMILES: %s" % smiles)
    if mol.GetNumBonds() == 0:
        return [Chem.MolToSmiles(mol)]

    rings, cells = _build_cells(mol, list(mol.GetBonds()), cell_r)
    adj = _cell_adjacency(mol, cells)

    pool, seen = [], set()  # (atoms, bond_set, frag)
    for i in range(len(cells)):
        for j in adj[i]:
            if i < j:
                atoms = cells[i] | cells[j]
                atoms |= _complete_aromatic(rings, atoms)
                frag = _render(mol, atoms)
                if frag in seen:
                    continue
                seen.add(frag)
                if Chem.MolFromSmiles(frag) is not None:
                    pool.append((atoms, _internal_bonds(mol, atoms), frag))
    if not pool:
        return [Chem.MolToSmiles(mol)]

    covered_bonds = set()
    frags = []
    for _ in range(num):
        best_gain, best = 0, []
        for atoms, bset, frag in pool:
            gain = len(bset - covered_bonds)
            if gain > best_gain:
                best_gain, best = gain, [(atoms, bset, frag)]
            elif gain == best_gain:
                best.append((atoms, bset, frag))
        if best_gain == 0:
            break
        atoms, bset, frag = rng.choice(best)
        covered_bonds |= bset
        frags.append(frag)

    if len(frags) < num:
        for atoms, bset, frag in rng.choices(pool, k=num - len(frags)):
            frags.append(frag)

    out = _dedup_keep_order(frags)
    if not out:
        out = [Chem.MolToSmiles(mol)]
    return out[:num]


def _connected_triples(cells, adj):
    """All connected 3-cell index sets, as sorted unique tuples."""
    triples = set()
    for i in range(len(cells)):
        for j in adj[i]:
            if i < j:
                for k in adj[i] | adj[j]:
                    if k != i and k != j:
                        triples.add(tuple(sorted((i, j, k))))
    return triples


def triplecov_large_fragments(smiles, cell_r=3, num=4, seed=None):
    """Large fragments as coverage-greedy unions of three cells.

    Same greedy maximum-marginal atom coverage as paircov, but the
    candidate pool is every connected triple of radius-`cell_r` cells
    (roughly 1.5x the atom span of a pair), aimed at targets too large
    for two-cell unions to reach a useful size. Triple pools are an
    order of magnitude larger than pair pools, so fragments are only
    rendered when selected: an unparseable pick is banned and the next
    best candidate is tried, which keeps per-call rendering near `num`.
    Falls back to pair unions when no connected triple exists, and to
    the whole molecule when nothing can be composed.
    """
    if cell_r < 0:
        raise ValueError("cell_r must be non-negative")
    rng = random.Random(seed) if seed is not None else random
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("cannot parse SMILES: %s" % smiles)
    if mol.GetNumBonds() == 0:
        return [Chem.MolToSmiles(mol)]

    rings, cells = _build_cells(mol, list(mol.GetBonds()), cell_r)
    adj = _cell_adjacency(mol, cells)

    pool, seen = [], set()  # atom sets of connected triples
    for tri in _connected_triples(cells, adj):
        atoms = set()
        for c in tri:
            atoms |= cells[c]
        atoms |= _complete_aromatic(rings, atoms)
        key = tuple(sorted(atoms))
        if key not in seen:
            seen.add(key)
            pool.append(atoms)

    if not pool:  # molecule too small for triples: use pair unions
        for i in range(len(cells)):
            for j in adj[i]:
                if i < j:
                    atoms = cells[i] | cells[j]
                    atoms |= _complete_aromatic(rings, atoms)
                    key = tuple(sorted(atoms))
                    if key not in seen:
                        seen.add(key)
                        pool.append(atoms)
    if not pool:
        return [Chem.MolToSmiles(mol)]

    covered = set()
    frags = []
    banned = set()
    while len(frags) < num:
        best_gain, best = 0, []
        for idx, atoms in enumerate(pool):
            if idx in banned:
                continue
            gain = len(atoms - covered)
            if gain > best_gain:
                best_gain, best = gain, [idx]
            elif gain == best_gain:
                best.append(idx)
        if best_gain == 0 or not best:
            break
        progress = False
        for idx in rng.sample(best, len(best)):  # shuffled frontier
            frag = _render(mol, pool[idx])
            banned.add(idx)
            if Chem.MolFromSmiles(frag) is None:
                continue
            covered |= pool[idx]
            frags.append(frag)
            progress = True
            break
        if not progress:
            continue  # frontier exhausted: recompute at the next gain tier

    if len(frags) < num:  # saturated: bounded random top-up
        attempts = 0
        while len(frags) < num and attempts < 4 * num:
            attempts += 1
            atoms = pool[rng.randrange(len(pool))]
            frag = _render(mol, atoms)
            if Chem.MolFromSmiles(frag) is not None:
                frags.append(frag)

    out = _dedup_keep_order(frags)
    if not out:
        out = [Chem.MolToSmiles(mol)]
    return out[:num]
