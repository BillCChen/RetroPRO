"""Per-stage breakpoint diagnosis for anchor8 overnight Pistachio-hard targets.

For each audited target we take a reference route (from the arm that solved
it) and replay the other arm's expansion/candidate/fragment/telemetry logs to
locate the first breakpoint layer:
  L1 parent product expanded
  L2 compatible fragments selected (product-side SMARTS containment proxy)
  L3 exact route reaction observed in full-product candidates
  L4 route template seen anywhere in the arm (candidate logs / final DICT)
  L5 fragment-level retro chain alive (retro_candidates stage flags)
  L6 route-child molecule expanded afterwards (tree scheduling)

Usage: run on a55002 with the unirxn2 interpreter; see main().
"""

import argparse
import glob
import json
import os
import pickle
from collections import defaultdict

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


def canon(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=False)


def canon_multi(smiles):
    if "." not in smiles:
        return canon(smiles)
    parts = [canon(part) for part in smiles.split(".")]
    if any(part is None for part in parts):
        return None
    return ".".join(sorted(parts))


def reaction_key(product, reactants):
    left = canon(product)
    if left is None:
        return None
    if isinstance(reactants, str):
        right = canon_multi(reactants)
    else:
        parts = [canon(r) for r in reactants]
        if any(p is None for p in parts):
            return None
        right = ".".join(sorted(parts))
    if right is None:
        return None
    return left + ">>" + right


def product_side_smarts(template):
    if not template or ">>" not in template:
        return None
    return Chem.MolFromSmarts(template.split(">>", 1)[0])


def extract_route_steps(route):
    """Ordered product->reactants steps from a SynRoute object."""
    mols = route.mols
    parents = route.parents
    children = route.children
    depth = {0: 0}
    for idx in range(1, len(mols)):
        parent = parents[idx]
        depth[idx] = depth.get(parent, 0) + 1 if parent >= 0 else 0
    steps = []
    for idx in range(len(mols)):
        if route.templates[idx] is None:
            continue
        reactant_ids = children[idx] or []
        steps.append({
            "node": idx,
            "depth": depth[idx],
            "product": mols[idx],
            "reactants": [mols[c] for c in reactant_ids],
            "template": route.templates[idx],
            "cost": route.costs.get(idx),
            "key": reaction_key(mols[idx], [mols[c] for c in reactant_ids]),
        })
    steps.sort(key=lambda s: s["depth"])
    return steps


def load_plan_steps(plan_path, target_ids):
    with open(plan_path, "rb") as handle:
        plan = pickle.load(handle)
    out = {}
    routes = plan["routes"]
    for tid in target_ids:
        route = routes[tid]
        if route is None:
            out[tid] = None
            continue
        out[tid] = extract_route_steps(route)
    return out


class ArmLogs:
    """Stream an arm's JSONL logs once, keeping only the audited targets."""

    def __init__(self, result_dir, target_ids):
        self.dir = result_dir
        self.target_ids = set(target_ids)
        self.expansions = defaultdict(list)       # tid -> [node record]
        self.reaction_first = defaultdict(dict)   # tid -> key -> first record
        self.reaction_count = defaultdict(int)    # tid -> candidate rows
        self.added_count = defaultdict(int)
        self.per_expansion_candidates = defaultdict(lambda: defaultdict(int))
        self.templates_seen = defaultdict(set)    # tid -> template strings
        self.frag_rows = defaultdict(list)        # (tid, expanded canon) -> [fragment row]
        self.retro_stage = defaultdict(lambda: defaultdict(int))  # tid -> stage counts
        self.reaction_stage = defaultdict(lambda: defaultdict(int))
        self.dict_templates = set()
        self._scan()

    def _scan(self):
        node_path = os.path.join(self.dir, "expansion_data", "node_expansions.jsonl")
        cand_path = os.path.join(self.dir, "expansion_data", "reaction_candidates.jsonl")
        frag_path = os.path.join(self.dir, "fragment_yield.jsonl")
        retro_path = os.path.join(self.dir, "retro_candidates.jsonl")

        with open(node_path) as handle:
            for line in handle:
                row = json.loads(line)
                tid = row["target_id"]
                if tid in self.target_ids:
                    self.expansions[tid].append(row)

        with open(cand_path) as handle:
            for line in handle:
                row = json.loads(line)
                tid = row["target_id"]
                if tid not in self.target_ids:
                    continue
                self.reaction_count[tid] += 1
                if row.get("expected_added_to_tree"):
                    self.added_count[tid] += 1
                self.per_expansion_candidates[tid][row["iteration"]] += 1
                if row.get("template"):
                    self.templates_seen[tid].add(row["template"])
                key = reaction_key(row["expanded_mol"], row["reactants"])
                if key is None:
                    continue
                first = self.reaction_first[tid].get(key)
                if first is None or row["iteration"] < first["iteration"]:
                    self.reaction_first[tid][key] = {
                        "iteration": row["iteration"],
                        "rank": row["candidate_rank"],
                        "cost": row["cost"],
                        "score": row["score"],
                        "valid": row["valid"],
                        "added": row["expected_added_to_tree"],
                        "template": row["template"],
                        "source_fragments": row.get("source_fragments", []),
                        "expanded_mol_id": row["expanded_mol_id"],
                    }

        with open(frag_path) as handle:
            for line in handle:
                row = json.loads(line)
                tid = row["task_id"]
                if tid not in self.target_ids:
                    continue
                exp_canon = canon(row["target"])
                if exp_canon is None:
                    continue
                self.frag_rows[(tid, exp_canon)].append({
                    "fragment": row["fragment"],
                    "sel_count": row["selected_fragment_count"],
                    "families": row["families"],
                    "retro_raw": row["retro_raw_count"],
                    "retro_valid": row["retro_valid_count"],
                    "forward_ok": row["forward_consistent_count"],
                    "mapped": row["mapped_count"],
                    "template": row["template_extracted_count"],
                    "full_product": row["full_product_unique_reaction_count"],
                })

        with open(retro_path) as handle:
            for line in handle:
                row = json.loads(line)
                tid = row["task_id"]
                if tid not in self.target_ids:
                    continue
                if row["grain"] == "retro_candidate":
                    stage = self.retro_stage[tid]
                    stage["n"] += 1
                    stage["valid"] += int(bool(row["retro_valid"]))
                    fc = row["forward_consistent"]
                    stage["forward_true"] += int(fc is True)
                    stage["forward_false"] += int(fc is False)
                elif row["grain"] == "reaction":
                    stage = self.reaction_stage[tid]
                    stage["n"] += 1
                    stage["mapped"] += int(bool(row["mapped"]))
                    stage["template"] += int(bool(row["template_extracted"]))
                    stage["in_output"] += int(bool(row["in_full_product_output"]))

        dict_files = glob.glob(os.path.join(self.dir, "tp_free_DICT_final_*.pkl"))
        if dict_files:
            with open(dict_files[0], "rb") as handle:
                payload = pickle.load(handle)
            rules = payload.get("rules", payload) if isinstance(payload, dict) else {}
            for rule_list in rules.values():
                for rule in rule_list:
                    self.dict_templates.add(rule)

    def expansions_of(self, tid, product_smiles):
        pc = canon(product_smiles)
        rows = [r for r in self.expansions.get(tid, []) if canon(r["expanded_mol"]) == pc]
        rows.sort(key=lambda r: r["iteration"])
        return rows

    def expansion_fragsets(self, tid, product_smiles, n_expected):
        """Reconstruct per-expansion fragment sets for one product.

        fragment_yield rows carry no iteration; rows for one (target, product)
        arrive in expansion order, each expansion writing
        ``selected_fragment_count`` rows.  Returns a list of fragment-row
        lists aligned with ``expansions_of`` order, or None when the chunking
        is inconsistent.
        """
        pc = canon(product_smiles)
        rows = self.frag_rows.get((tid, pc), [])
        if n_expected <= 0:
            return []
        if not rows:
            return [[] for _ in range(n_expected)]
        sets = []
        pos = 0
        while pos < len(rows) and len(sets) < n_expected:
            size = rows[pos]["sel_count"]
            if size <= 0 or pos + size > len(rows):
                return None
            chunk = rows[pos:pos + size]
            if any(row["sel_count"] != size for row in chunk):
                return None
            sets.append(chunk)
            pos += size
        if pos != len(rows) or len(sets) != n_expected:
            return None
        return sets


def analyze_target(tid, ref_steps, ref_name, arms):
    """Layer-by-layer breakpoint analysis for one target in each non-ref arm."""
    report = {"target_id": tid, "reference_arm": ref_name,
              "reference_steps": len(ref_steps), "steps": []}
    prod_smarts_cache = {}
    for step in ref_steps:
        row = {
            "depth": step["depth"],
            "cost_ref": step["cost"],
            "key_present": {},
        }
        template = step["template"]
        if template not in prod_smarts_cache:
            prod_smarts_cache[template] = product_side_smarts(template)
        prod_pattern = prod_smarts_cache[template]
        for arm_name, logs in arms.items():
            cell = {}
            expansions = logs.expansions_of(tid, step["product"])
            cell["product_expansions"] = [r["iteration"] for r in expansions]
            fragsets = logs.expansion_fragsets(tid, step["product"], len(expansions))
            compat = []
            for idx, r in enumerate(expansions):
                rows = fragsets[idx] if fragsets is not None else []
                n_compat = 0
                n_reclaim = 0
                for frag in rows:
                    if "anchor_reclaim" in frag["families"]:
                        n_reclaim += 1
                    mol = Chem.MolFromSmiles(frag["fragment"])
                    if mol is not None and prod_pattern is not None \
                            and mol.HasSubstructMatch(prod_pattern):
                        n_compat += 1
                compat.append({"iteration": r["iteration"], "n_fragments": len(rows),
                               "n_compatible": n_compat, "n_reclaim": n_reclaim})
            cell["fragment_compatibility"] = compat
            cell["fragsets_consistent"] = fragsets is not None
            first = logs.reaction_first[tid].get(step["key"])
            cell["reaction_first"] = first
            cell["template_in_candidates"] = template in logs.templates_seen.get(tid, set())
            cell["template_in_final_dict"] = template in logs.dict_templates
            row[arm_name] = cell
        report["steps"].append(row)
    return report


def summarize_arm_activity(tid, logs):
    exps = logs.expansions.get(tid, [])
    counts = list(logs.per_expansion_candidates.get(tid, {}).values())
    return {
        "expansions": len(exps),
        "candidate_rows": logs.reaction_count.get(tid, 0),
        "added_rows": logs.added_count.get(tid, 0),
        "mean_candidates_per_expansion": (sum(counts) / len(counts)) if counts else 0,
        "retro_stage": dict(logs.retro_stage.get(tid, {})),
        "reaction_stage": dict(logs.reaction_stage.get(tid, {})),
    }


def reclaim_share_on_route(tid, ref_steps, logs):
    """Share of route-product expansions whose fragment set includes a reclaim."""
    flagged = 0
    total = 0
    for step in ref_steps:
        expansions = logs.expansions_of(tid, step["product"])
        fragsets = logs.expansion_fragsets(tid, step["product"], len(expansions))
        for idx, _r in enumerate(expansions):
            total += 1
            rows = fragsets[idx] if fragsets is not None else []
            if any("anchor_reclaim" in frag["families"] for frag in rows):
                flagged += 1
    return {"route_expansions": total, "with_reclaim": flagged}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-dir", required=True)
    parser.add_argument("--control-dir", required=True)
    parser.add_argument("--paper-plan", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    target_ids = [22, 5, 9, 11, 16, 19]
    arms = {
        "anchor": ArmLogs(args.anchor_dir, target_ids),
        "control": ArmLogs(args.control_dir, target_ids),
    }
    anchor_steps = load_plan_steps(os.path.join(args.anchor_dir, "plan.pkl"), target_ids)
    control_steps = load_plan_steps(os.path.join(args.control_dir, "plan.pkl"), target_ids)
    paper_steps = load_plan_steps(args.paper_plan, target_ids)

    jobs = [
        (22, control_steps[22], "control", {"anchor": arms["anchor"]}),
        (5, control_steps[5], "control", {"anchor": arms["anchor"]}),
        (5, anchor_steps[5], "anchor", {"control": arms["control"]}),
        (9, anchor_steps[9], "anchor", {"control": arms["control"]}),
        (11, paper_steps[11], "paper", arms),
        (16, paper_steps[16], "paper", arms),
        (19, paper_steps[19], "paper", arms),
    ]

    results = []
    for tid, ref_steps, ref_name, other_arms in jobs:
        if not ref_steps:
            results.append({"target_id": tid, "reference_arm": ref_name,
                            "error": "no reference route"})
            continue
        rep = analyze_target(tid, ref_steps, ref_name, other_arms)
        rep["activity"] = {
            arm: summarize_arm_activity(tid, logs) for arm, logs in
            dict(other_arms, **({ref_name: arms[ref_name]} if ref_name in arms else {})).items()
        }
        rep["reclaim_on_route"] = {
            arm: reclaim_share_on_route(tid, ref_steps, logs)
            for arm, logs in other_arms.items()
        }
        results.append(rep)
        print("analyzed target %d (ref=%s, steps=%d)" % (tid, ref_name, len(ref_steps)))

    with open(args.out, "w") as handle:
        json.dump({"schema_version": "1.0.0", "results": results}, handle, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
