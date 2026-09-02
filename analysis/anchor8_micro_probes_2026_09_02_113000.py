"""Targeted micro-probes on top of the anchor8 loss diagnosis.

S1: target 22 root reaction in anchor — generation vs validation bisect using
    per-candidate retro telemetry (root expanded only once, so all rows for
    that molecule belong to iteration 1).
S2: target 9 anchor route d1/d7 — were the winning reactions sourced from
    reclaimed fragments?
S3: target 11 anchor — did the paper-route d5 product ever arrive in the
    anchor tree as an added reactant (scheduling vs generation)?
S4: target 5 anchor route root reaction — first-seen and reclaim status.
"""

import json
import os
import pickle
import sys
from collections import defaultdict

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

BASE = "/home/chenqixuan/retropro_effective_yield_dev_2026_09_01_095400/analysis_outputs/anchor8_overnight_2026_09_02_003500"
ANCHOR = os.path.join(BASE, "s2_main_anchor8_i1000")
CONTROL = os.path.join(BASE, "s3_control_random_i1000")
PAPER = "/home/chenqixuan/retro_star/retro_star/results/data_collection/template_free_pth_hard_iter1000_topk8_rd7_0_3_0_gpu1_0521_112605/plan.pkl"


def canon(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, isomericSmiles=False) if mol is not None else None


def canon_parts(smiles):
    parts = smiles.split(".")
    out = [canon(p) for p in parts]
    if any(p is None for p in out):
        return None
    return ".".join(sorted(out))


def reaction_key(product, reactants):
    left = canon(product)
    right = canon_parts(reactants) if isinstance(reactants, str) else ".".join(
        sorted(canon(r) for r in reactants))
    if left is None or right is None:
        return None
    return left + ">>" + right


def route_steps(plan_path, tid):
    with open(plan_path, "rb") as handle:
        plan = pickle.load(handle)
    route = plan["routes"][tid]
    mols, parents, children = route.mols, route.parents, route.children
    depth = {0: 0}
    for idx in range(1, len(mols)):
        p = parents[idx]
        depth[idx] = depth.get(p, 0) + 1 if p >= 0 else 0
    steps = {}
    for idx in range(len(mols)):
        if route.templates[idx] is None:
            continue
        steps[depth[idx]] = {
            "product": mols[idx],
            "reactants": [mols[c] for c in (children[idx] or [])],
            "template": route.templates[idx],
        }
    return steps


def probe_s1():
    steps = route_steps(os.path.join(CONTROL, "plan.pkl"), 22)
    root = steps[0]
    root_canon = canon(root["product"])
    ref_reactants = canon_parts(".".join(sorted(root["reactants"])))
    hits = []
    total = 0
    path = os.path.join(ANCHOR, "retro_candidates.jsonl")
    with open(path) as handle:
        for line in handle:
            row = json.loads(line)
            if row["task_id"] != 22 or row["grain"] != "retro_candidate":
                continue
            if canon(row["expanded_mol"]) != root_canon:
                continue
            total += 1
            if canon_parts(row["retro_raw"]) == ref_reactants:
                hits.append({
                    "fragment": row["fragment"],
                    "retro_valid": row["retro_valid"],
                    "forward_consistent": row["forward_consistent"],
                })
    return {"root_retro_candidates": total, "exact_reactant_hits": hits,
            "ref_reactants": ref_reactants}


def load_frag_families(result_dir, tid):
    fam = {}
    path = os.path.join(result_dir, "fragment_yield.jsonl")
    with open(path) as handle:
        for line in handle:
            row = json.loads(line)
            if row["task_id"] != tid:
                continue
            fam[row["fragment"]] = row["families"]
    return fam


def reaction_sources(result_dir, tid, keys):
    found = {}
    path = os.path.join(result_dir, "expansion_data", "reaction_candidates.jsonl")
    with open(path) as handle:
        for line in handle:
            row = json.loads(line)
            if row["target_id"] != tid:
                continue
            key = reaction_key(row["expanded_mol"], row["reactants"])
            if key in keys and key not in found:
                found[key] = {
                    "iteration": row["iteration"],
                    "rank": row["candidate_rank"],
                    "cost": row["cost"],
                    "source_fragments": row.get("source_fragments", []),
                }
    return found


def probe_s2():
    steps = route_steps(os.path.join(ANCHOR, "plan.pkl"), 9)
    keys = {}
    for d in (1, 7):
        if d in steps:
            keys[reaction_key(steps[d]["product"], steps[d]["reactants"])] = d
    fam = load_frag_families(ANCHOR, 9)
    found = reaction_sources(ANCHOR, 9, set(keys))
    out = {}
    for key, d in keys.items():
        rec = found.get(key)
        if rec is None:
            out[d] = {"status": "missing"}
            continue
        out[d] = dict(rec)
        out[d]["source_families"] = [
            {"fragment": frag, "families": fam.get(frag, fam.get(canon(frag), []))}
            for frag in rec["source_fragments"]
        ]
    return out


def probe_s3():
    steps = route_steps(PAPER, 11)
    d5_canon = canon(steps[5]["product"])
    arrived = None
    path = os.path.join(ANCHOR, "expansion_data", "reaction_candidates.jsonl")
    with open(path) as handle:
        for line in handle:
            row = json.loads(line)
            if row["target_id"] != 11:
                continue
            parts = [canon(r) for r in row["reactants"]]
            if d5_canon in parts:
                rec = {
                    "iteration": row["iteration"],
                    "added": row["expected_added_to_tree"],
                    "valid": row["valid"],
                }
                if arrived is None or rec["iteration"] < arrived["iteration"]:
                    arrived = rec
    return {"d5_product_arrived_as_reactant": arrived}


def probe_s4():
    steps = route_steps(os.path.join(ANCHOR, "plan.pkl"), 5)
    root = steps[0]
    key = reaction_key(root["product"], root["reactants"])
    fam = load_frag_families(ANCHOR, 5)
    found = reaction_sources(ANCHOR, 5, {key})
    rec = found.get(key)
    if rec is None:
        return {"status": "missing"}
    rec["source_families"] = [
        {"fragment": frag, "families": fam.get(frag, fam.get(canon(frag), []))}
        for frag in rec["source_fragments"]
    ]
    return rec


def main():
    report = {
        "s1_target22_root_bisect": probe_s1(),
        "s2_target9_route_sources": probe_s2(),
        "s3_target11_d5_arrival": probe_s3(),
        "s4_target5_anchor_root": probe_s4(),
    }
    out = os.path.join(BASE, "anchor8_micro_probes_2026_09_02_113000.json")
    with open(out, "w") as handle:
        json.dump(report, handle, indent=1)
    print(json.dumps(report, indent=1)[:3000])
    print("wrote", out)


if __name__ == "__main__":
    main()
