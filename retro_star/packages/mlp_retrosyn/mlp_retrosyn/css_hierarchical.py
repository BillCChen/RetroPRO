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
