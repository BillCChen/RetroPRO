"""Offline truth bond-breaking evaluation for the CSS substructure samplers.

Question: for a given target, does at least one of the substructure fragments
a sampler produces (K per draw, unioned over S seeds) fully contain the true
first-step reaction centre?  Reports per-arm hit rate with Wilson 95% CIs,
stratified by atom-map confidence and reaction-centre size, for two centre
definitions (strict changed-atom set, and changed-atom set + 1-bond shell).

Part 1 of the CSS sampler formal experiment -- the 190-target track only.
The search-tree-intermediate track is deferred.

Pure RDKit + rxnmapper + rdchiral.  No planner, no GPU.

Reaction centre = rdchiral get_changed_atoms on the product side of the
atom-mapped step-1 reaction (product >> precursors, i.e. forward precursors
-> product).  Atom maps come from rxnmapper; confidence is recorded and the
report is stratified by it (>=0.9 tier is the trustworthy headline).
"""
import argparse
import json
import math
import os
import pickle
import sys
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

ARMS = ("random", "paircov", "fullcov", "bondcov", "triplecov")
CONF_TIERS = (("ge_0.9", 0.9, 2.0), ("0.7_0.9", 0.7, 0.9),
              ("0.5_0.7", 0.5, 0.7), ("lt_0.5", -1.0, 0.5))
RC_BUCKETS = (("1-4", 1, 5), ("5-8", 5, 9), ("9+", 9, 10 ** 6))


# --- sampler harness ----------------------------------------------------

def make_sampler():
    from mlp_retrosyn.tp_free_inference import TP_free_Model

    model = TP_free_Model.__new__(TP_free_Model)
    model.use_CCS = True
    return model


def sample_fragments(model, smiles, arm, rd, topk, seeds):
    """Union of fragment SMILES an arm yields over `seeds` independent draws."""
    import random

    frags = set()
    prev = os.environ.get("TP_FREE_CSS_SAMPLER")
    os.environ["TP_FREE_CSS_SAMPLER"] = arm
    try:
        for s in range(seeds):
            random.seed(1000 + s)
            frags.update(model.random_sampling(smiles, rd, topk))
    finally:
        if prev is None:
            os.environ.pop("TP_FREE_CSS_SAMPLER", None)
        else:
            os.environ["TP_FREE_CSS_SAMPLER"] = prev
    return frags


# --- truth: first-step reaction centre --------------------------------

def first_reaction(entry):
    """Step-1 reaction string of one routes_possible_test_hard entry.

    Each entry is a list of 'product>>precursors' strings (retro order);
    entry[0] is the true first step.
    """
    return entry[0] if isinstance(entry, (list, tuple)) else entry


def reaction_centre_atoms(rxn_smi, mapper):
    """(target_mol, rc_strict, rc_shell, confidence) for one step-1 reaction.

    target_mol has map numbers stripped; rc_strict / rc_shell are atom-index
    sets in target_mol's space.  Returns None if the reaction cannot be
    mapped or no changed atoms are found on the product side.
    """
    from rdchiral import template_extractor as te

    try:
        res = mapper.get_attention_guided_atom_maps([rxn_smi])[0]
    except Exception:
        return None
    mapped, conf = res["mapped_rxn"], float(res["confidence"])
    prod_s, prec_s = mapped.split(">>")[0], mapped.split(">>")[-1]
    try:
        precursors = te.mols_from_smiles_list(
            te.replace_deuterated(prec_s).split("."))
        products = te.mols_from_smiles_list(
            te.replace_deuterated(prod_s).split("."))
        _, changed_tags, err = te.get_changed_atoms(precursors, products)
    except Exception:
        return None
    if err:
        return None
    pmol = Chem.MolFromSmiles(prod_s)
    if pmol is None:
        return None
    tags = set(str(t) for t in changed_tags)
    rc_strict = {a.GetIdx() for a in pmol.GetAtoms()
                 if str(a.GetAtomMapNum()) in tags}
    if not rc_strict:
        return None
    rc_shell = set(rc_strict)
    for i in list(rc_strict):
        rc_shell.update(nb.GetIdx() for nb in pmol.GetAtomWithIdx(i).GetNeighbors())
    for a in pmol.GetAtoms():
        a.SetAtomMapNum(0)
    return pmol, rc_strict, rc_shell, conf


def load_target_truth(pkl_path, mapper):
    with open(pkl_path, "rb") as fh:
        data = pickle.load(fh)
    records, skipped = [], 0
    for entry in data:
        out = reaction_centre_atoms(first_reaction(entry), mapper)
        if out is None:
            skipped += 1
            continue
        mol, rc_strict, rc_shell, conf = out
        records.append({
            "target": Chem.MolToSmiles(mol),
            "mol": mol,
            "rc_strict": rc_strict,
            "rc_shell": rc_shell,
            "conf": conf,
            "n_atoms": mol.GetNumAtoms(),
        })
    return records, skipped


# --- hit test --------------------------------------------------------

def fragment_hit(target_mol, frags, rc_atoms):
    """(hit, n_unparseable) over a set of fragment SMILES for one RC set."""
    hit = False
    unparseable = 0
    for smi in frags:
        fmol = Chem.MolFromSmiles(smi)
        if fmol is None:
            unparseable += 1
            continue
        if hit:
            continue
        for match in target_mol.GetSubstructMatches(fmol, uniquify=True,
                                                    maxMatches=500):
            if rc_atoms.issubset(match):
                hit = True
                break
    return hit, unparseable


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = p + z * z / (2.0 * n)
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return (p, (centre - half) / denom, (centre + half) / denom)


def _tier(conf):
    for name, lo, hi in CONF_TIERS:
        if lo <= conf < hi:
            return name
    return "lt_0.5"


def _bucket(size):
    for name, lo, hi in RC_BUCKETS:
        if lo <= size < hi:
            return name
    return "9+"


def summarise(counter):
    """counter[key][arm] -> (hits_strict, hits_shell, n, unparseable)."""
    out = {}
    for key, per_arm in counter.items():
        out[key] = {}
        for arm, (hs, hsh, n, unp) in per_arm.items():
            ps, los, his = wilson(hs, n)
            psh, losh, hish = wilson(hsh, n)
            out[key][arm] = {
                "n": n,
                "strict": {"hits": hs, "rate": round(ps, 4),
                           "ci95": [round(los, 4), round(his, 4)]},
                "shell": {"hits": hsh, "rate": round(psh, 4),
                          "ci95": [round(losh, 4), round(hish, 4)]},
                "unparseable_frags": unp,
            }
    return out


def run(records, arms, rd, topk, seeds, out_path):
    model = make_sampler()
    overall = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0]))
    by_conf = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0]))
    by_rc = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0]))
    per_target = []
    for i, rec in enumerate(records):
        tmol = rec["mol"]
        tier = _tier(rec["conf"])
        bucket = _bucket(len(rec["rc_strict"]))
        row = {"target": rec["target"], "conf": round(rec["conf"], 3),
               "rc_strict_size": len(rec["rc_strict"]),
               "rc_shell_size": len(rec["rc_shell"])}
        for arm in arms:
            frags = sample_fragments(model, rec["target"], arm, rd, topk, seeds)
            hs, unp = fragment_hit(tmol, frags, rec["rc_strict"])
            hsh, _ = fragment_hit(tmol, frags, rec["rc_shell"])
            for tbl, key in ((overall, "all"), (by_conf, tier), (by_rc, bucket)):
                c = tbl[key][arm]
                c[0] += int(hs)
                c[1] += int(hsh)
                c[2] += 1
                c[3] += unp
            row["%s_strict" % arm] = hs
            row["%s_shell" % arm] = hsh
        per_target.append(row)
        if (i + 1) % 20 == 0:
            print("  %d/%d" % (i + 1, len(records)), flush=True)

    report = {
        "config": {"rd": rd, "topk": topk, "seeds": seeds, "arms": list(arms),
                   "n_targets": len(records)},
        "overall": summarise(overall),
        "by_conf_tier": summarise(by_conf),
        "by_rc_size": summarise(by_rc),
        "per_target": per_target,
    }
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2)

    for title, tbl in (("OVERALL", report["overall"]),
                       ("BY CONF TIER", report["by_conf_tier"]),
                       ("BY RC SIZE", report["by_rc_size"])):
        print("\n== %s ==" % title)
        for key in sorted(tbl):
            print(" [%s]" % key)
            for arm in arms:
                s = tbl[key].get(arm)
                if not s:
                    continue
                print("   %-9s n=%-3d strict %.3f [%.3f,%.3f]  shell %.3f "
                      "[%.3f,%.3f]  unparse=%d" % (
                          arm, s["n"],
                          s["strict"]["rate"], s["strict"]["ci95"][0],
                          s["strict"]["ci95"][1],
                          s["shell"]["rate"], s["shell"]["ci95"][0],
                          s["shell"]["ci95"][1], s["unparseable_frags"]))
    print("\nwrote", out_path)


def probe_pkl(pkl_path, n=3):
    with open(pkl_path, "rb") as fh:
        data = pickle.load(fh)
    print("type:", type(data).__name__, "len:", len(data))
    for i in range(min(n, len(data))):
        entry = data[i]
        print("[%d] type=%s len=%s" % (
            i, type(entry).__name__,
            len(entry) if hasattr(entry, "__len__") else "NA"))
        print("    ", repr(entry)[:600])


def parse_args(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--rd", default="3,0", help="single (r,d) channel, comma-sep")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="first N targets only")
    ap.add_argument("--out", default="css_offline_eval_targets.json")
    ap.add_argument("--probe", action="store_true")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])
    if args.probe:
        probe_pkl(args.pkl)
        return
    r, d = (int(x) for x in args.rd.split(","))
    rd = [(r, d)]
    arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    from rxnmapper import RXNMapper

    print("mapping step-1 reactions ...", flush=True)
    records, skipped = load_target_truth(args.pkl, RXNMapper())
    if args.limit:
        records = records[:args.limit]
    print("targets with a usable reaction centre: %d (skipped %d)"
          % (len(records), skipped), flush=True)
    run(records, arms, rd, args.topk, args.seeds, args.out)


if __name__ == "__main__":
    main()
