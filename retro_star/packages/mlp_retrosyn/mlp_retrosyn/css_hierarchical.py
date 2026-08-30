"""Hierarchical (cell-graph) substructure samplers for CSS.

Instead of drawing large fragments as isotropic radius-R balls around a
single seed bond, these samplers grow small radius-`cell_r` "cells" first
and compose large fragments as connected unions of adjacent cells:

  - "pair":      uniformly sampled adjacent cell pairs (2-cell unions)
  - "walk":      random walk over the cell adjacency graph, merging cells
                 until a size budget is reached
  - "partition": deterministic transitive merge of all touching cells
                 (connected components of the cell graph)

Cell adjacency (atom-set intersection or direct bonding) guarantees that
every emitted fragment is connected. Aromatic-ring completion follows the
same convention as `random_substructure`: any all-aromatic ring touched by
the distance-driven atom set is included whole, non-recursively.

The production entry point is `hierarchical_substructure`, mirroring the
signature style of `random_substructure` in tp_free_tools.py.
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


def _sample_seed_bonds(rng, bonds, num):
    if num > len(bonds):
        return bonds + rng.choices(bonds, k=num - len(bonds))
    return rng.sample(bonds, num)


def _pair_fragments(rng, mol, cells, adj, rings, num):
    pairs = [(i, j) for i in range(len(cells)) for j in adj[i] if i < j]
    frags = []
    if pairs:
        for i, j in (rng.sample(pairs, min(num, len(pairs)))):
            atoms = cells[i] | cells[j]
            atoms |= _complete_aromatic(rings, atoms)
            frags.append(_render(mol, atoms))
    return frags


def _walk_fragments(rng, mol, cells, adj, rings, num, size_frac):
    n_atoms = mol.GetNumAtoms()
    budget = max(1, int(round(size_frac * n_atoms)))
    frags = []
    for start in range(len(cells)):
        current = set(cells[start])
        visited = {start}
        frontier = [c for c in adj[start] if c not in visited]
        while len(current) < budget and frontier:
            nxt = rng.choice(frontier)
            visited.add(nxt)
            current |= cells[nxt]
            frontier = [c for c in adj[nxt] if c not in visited] + [c for c in frontier if c != nxt and c not in visited]
        current |= _complete_aromatic(rings, current)
        frags.append(_render(mol, current))
    return frags


def _partition_fragments(rng, mol, cells, adj, rings, num):
    seen = [False] * len(cells)
    components = []
    for i in range(len(cells)):
        if seen[i]:
            continue
        comp, stack, seen[i] = set(), [i], True
        while stack:
            u = stack.pop()
            comp |= cells[u]
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        components.append(comp)
    components.sort(key=len, reverse=True)
    frags = []
    for atoms in components:
        atoms = set(atoms) | _complete_aromatic(rings, set(atoms))
        frags.append(_render(mol, atoms))
    if len(frags) < num:  # top up with deterministic adjacent-pair unions
        pairs = sorted(
            ((len(cells[i] | cells[j]), i, j) for i in range(len(cells)) for j in adj[i] if i < j))
        for _, i, j in pairs:
            atoms = cells[i] | cells[j]
            atoms |= _complete_aromatic(rings, atoms)
            frags.append(_render(mol, atoms))
            if len(frags) >= num:
                break
    return frags


def hierarchical_substructure(smiles, mode="walk", cell_r=3, num=1, size_frac=0.6, seed=None):
    """Sample up to `num` connected fragments via cell-graph composition.

    mode: "pair" | "walk" | "partition". size_frac only applies to "walk"
    (stop merging once the fragment covers this fraction of the molecule).
    Returns a list of canonical fragment SMILES (length <= num, unique).
    Falls back to the whole molecule when no composition is possible.
    """
    if cell_r < 0:
        raise ValueError("cell_r must be non-negative")
    if mode not in ("pair", "walk", "partition"):
        raise ValueError("unknown mode: %s" % mode)
    rng = random.Random(seed) if seed is not None else random
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("cannot parse SMILES: %s" % smiles)
    if mol.GetNumBonds() == 0:
        return [Chem.MolToSmiles(mol)] * num

    seed_bonds = _sample_seed_bonds(rng, list(mol.GetBonds()), num)
    rings, cells = _build_cells(mol, seed_bonds, cell_r)
    adj = _cell_adjacency(mol, cells)

    if mode == "pair":
        frags = _pair_fragments(rng, mol, cells, adj, rings, num)
    elif mode == "walk":
        frags = _walk_fragments(rng, mol, cells, adj, rings, num, size_frac)
    else:
        frags = _partition_fragments(rng, mol, cells, adj, rings, num)

    out = _dedup_keep_order(frags)
    if not out:
        out = [Chem.MolToSmiles(mol)]
    return out[:num]
