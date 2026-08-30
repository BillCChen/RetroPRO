"""Smoke battery for the CSS hierarchical substructure samplers.

Exercises TP_free_Model.random_sampling across all five TP_FREE_CSS_SAMPLER
modes without loading any model (TP_free_Model.__new__ bypass). Pure RDKit.

Checks, over a set of real targets plus small intermediates:
  - fragment count in [1, topk], all unique, all connected, all parseable
  - default (env unset) == explicit "random" after canonicalisation
  - unknown mode -> ValueError
  - single-bond / no-bond molecules -> whole-molecule fallback
  - seed-bearing module functions are deterministic

Usage:
  python test_css_samplers.py [uspto190_targets.txt]
"""
import os
import sys
import random
from pathlib import Path

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

MODES = ("random", "paircov", "fullcov", "bondcov", "triplecov")
RD_SINGLE = [(3, 0)]
TOPK = 8


def load_model():
    from mlp_retrosyn.tp_free_inference import TP_free_Model

    model = TP_free_Model.__new__(TP_free_Model)
    model.use_CCS = True
    return model


def _connected_and_valid(smi):
    return "." not in smi and Chem.MolFromSmiles(smi) is not None


def check_basic(model, targets):
    for mode in MODES:
        os.environ["TP_FREE_CSS_SAMPLER"] = mode
        for smi in targets:
            out = model.random_sampling(smi, RD_SINGLE, TOPK)
            assert 1 <= len(out) <= TOPK, (mode, smi, len(out))
            assert len(set(out)) == len(out), ("dup", mode, smi)
            assert all(_connected_and_valid(s) for s in out), (mode, smi, out)
        print("  [%-9s] %d molecules OK" % (mode, len(targets)))


def check_default_equiv(model, smi):
    os.environ.pop("TP_FREE_CSS_SAMPLER", None)
    random.seed(123)
    a = sorted(Chem.MolToSmiles(Chem.MolFromSmiles(s))
               for s in model.random_sampling(smi, RD_SINGLE, TOPK))
    random.seed(123)
    os.environ["TP_FREE_CSS_SAMPLER"] = "random"
    b = sorted(Chem.MolToSmiles(Chem.MolFromSmiles(s))
               for s in model.random_sampling(smi, RD_SINGLE, TOPK))
    assert a == b, "default and explicit random diverge after canonicalisation"
    print("  default == explicit random OK")


def check_unknown_mode(model, smi):
    os.environ["TP_FREE_CSS_SAMPLER"] = "does-not-exist"
    try:
        model.random_sampling(smi, RD_SINGLE, TOPK)
    except ValueError:
        print("  unknown mode -> ValueError OK")
    else:
        raise AssertionError("unknown mode did not raise ValueError")


def check_fallbacks(model):
    for mode in MODES:
        os.environ["TP_FREE_CSS_SAMPLER"] = mode
        assert model.random_sampling("CC", RD_SINGLE, TOPK), ("single-bond", mode)
        assert model.random_sampling("C", RD_SINGLE, TOPK), ("no-bond", mode)
    print("  single-bond / no-bond fallbacks OK")


def check_determinism(smi):
    from mlp_retrosyn import css_hierarchical as ch

    for fn in (ch.paircov_large_fragments, ch.bondcov_large_fragments,
               ch.triplecov_large_fragments):
        a = fn(smi, cell_r=3, num=4, seed=7)
        b = fn(smi, cell_r=3, num=4, seed=7)
        assert a == b, fn.__name__
    a = ch.fullcov_fragments(smi, cell_r=3, num=8, seed=7)
    b = ch.fullcov_fragments(smi, cell_r=3, num=8, seed=7)
    assert a == b, "fullcov_fragments"
    print("  seeded determinism OK")


def collect_targets(argv):
    targets = []
    if len(argv) > 1 and Path(argv[1]).exists():
        targets = [ln.strip() for ln in open(argv[1]) if ln.strip()][:20]
    targets += [
        "CC(=O)Oc1ccccc1C(=O)O",
        "C[C@H](c1ccccc1)N1C[C@]2(C(=O)OC(C)(C)C)C=CC[C@@H]2C1=S",
    ]
    return targets


def main(argv):
    targets = collect_targets(argv)
    model = load_model()
    print("basic battery:")
    check_basic(model, targets)
    check_default_equiv(model, targets[-1])
    check_unknown_mode(model, targets[-1])
    check_fallbacks(model)
    check_determinism(model and targets[0])
    os.environ.pop("TP_FREE_CSS_SAMPLER", None)
    print("ALL CSS SAMPLER SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main(sys.argv)
