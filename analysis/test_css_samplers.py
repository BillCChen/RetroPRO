"""Smoke battery for the CSS hierarchical substructure samplers.

Exercises TP_free_Model.random_sampling across all six TP_FREE_CSS_SAMPLER
modes without loading any model (TP_free_Model.__new__ bypass). Pure RDKit.

Checks, over a set of real targets plus small intermediates:
  - fragment count in [1, topk], all unique, all connected, all parseable
  - default (env unset) == explicit "random" after canonicalisation
  - unknown mode -> ValueError
  - single-bond / no-bond molecules -> whole-molecule fallback
  - seed-bearing module functions are deterministic
  - yield8 returns eight canonical-unique fragments when capacity permits
  - yield8 prioritises the union of known full-product reactions

Usage:
  python test_css_samplers.py [uspto190_targets.txt]
"""
import os
import sys
import random
import json
import tempfile
import threading
import types
from collections import defaultdict
from pathlib import Path

from rdkit import Chem
from rdkit import RDLogger

REPO_ROOT = Path(__file__).resolve().parents[1]
MLP_PACKAGE_ROOT = REPO_ROOT / "retro_star" / "packages" / "mlp_retrosyn"
if str(MLP_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(MLP_PACKAGE_ROOT))

RDLogger.DisableLog("rdApp.*")

MODES = (
    "random", "paircov", "fullcov", "bondcov", "triplecov", "yield8",
    "yield8_hybrid", "anchor8",
)
RD_SINGLE = [(3, 0)]
TOPK = 8


def load_model():
    from mlp_retrosyn.tp_free_inference import TP_free_Model

    model = TP_free_Model.__new__(TP_free_Model)
    model.use_CCS = True
    model._dict_ref = {}
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


def check_yield8_contract(model, smi):
    from mlp_retrosyn.css_effective_yield import (
        enumerate_effective_yield_candidates,
        select_effective_yield_fragments,
    )

    candidates = enumerate_effective_yield_candidates(smi)
    assert len(candidates) >= TOPK, len(candidates)
    bounded = enumerate_effective_yield_candidates(smi, max_triples=8)
    assert sum("triple" in row["families"] for row in bounded) <= 8

    known = {
        candidates[0]["smiles"]: {"reaction-a", "reaction-b"},
        candidates[1]["smiles"]: {"reaction-a", "reaction-b"},
        candidates[2]["smiles"]: {"reaction-c"},
    }

    def lookup(fragment):
        return known.get(fragment, set())

    selected, metadata = select_effective_yield_fragments(
        smi, TOPK, lookup, exploration_slots=2, seed=17)
    selected_again, metadata_again = select_effective_yield_fragments(
        smi, TOPK, lookup, exploration_slots=2, seed=17)
    assert selected == selected_again
    assert metadata == metadata_again
    assert len(selected) == TOPK
    assert len(set(selected)) == TOPK
    assert metadata["known_reaction_union_count"] == 3
    assert candidates[2]["smiles"] in selected

    os.environ["TP_FREE_CSS_SAMPLER"] = "yield8"
    random.seed(17)
    runtime = model.random_sampling(smi, RD_SINGLE, TOPK)
    assert len(runtime) == TOPK
    assert len({Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in runtime}) == TOPK

    small, small_meta = select_effective_yield_fragments(
        "C", TOPK, lambda _fragment: set(), seed=17)
    assert small == ["C"]
    assert small_meta["capacity_limited"] is True
    print("  yield8 strict-cardinality and marginal-yield contract OK")


def check_strict_baseline_contract(model, smi):
    os.environ["TP_FREE_CSS_STRICT_TOPK"] = "1"
    configs = (
        ("random", [(7, 0), (3, 0)]),
        ("random", [(3, 0)]),
        ("paircov", [(3, 0)]),
        ("triplecov", [(3, 0)]),
    )
    for mode, rd_list in configs:
        os.environ["TP_FREE_CSS_SAMPLER"] = mode
        random.seed(29)
        output = model.random_sampling(smi, rd_list, TOPK)
        canonical = {
            Chem.MolToSmiles(Chem.MolFromSmiles(fragment))
            for fragment in output
        }
        assert len(output) == TOPK, (mode, rd_list, len(output))
        assert len(canonical) == TOPK, (mode, rd_list, canonical)
    os.environ.pop("TP_FREE_CSS_STRICT_TOPK", None)
    print("  strict baseline top-k backfill contract OK")


def check_yield8_hybrid_contract(model, smi):
    os.environ["TP_FREE_CSS_SAMPLER"] = "yield8_hybrid"
    os.environ.pop("TP_FREE_YIELD_GUARDRAIL_SLOTS", None)
    random.seed(31)
    output = model.random_sampling(smi, RD_SINGLE, TOPK)
    canonical = [Chem.MolToSmiles(Chem.MolFromSmiles(item)) for item in output]
    assert len(output) == TOPK
    assert len(set(canonical)) == TOPK
    sampled, metadata = model._yield8_hybrid_sampling(smi, TOPK)
    assert len(sampled) == TOPK
    assert metadata["production_guardrail_budget"] == 4
    assert metadata["exploration_profile"] == "hybrid"
    os.environ["TP_FREE_YIELD_GUARDRAIL_SLOTS"] = "6"
    sampled, metadata = model._yield8_hybrid_sampling(smi, TOPK)
    assert len(sampled) == TOPK
    assert metadata["production_guardrail_budget"] == 6
    os.environ.pop("TP_FREE_YIELD_GUARDRAIL_SLOTS", None)
    print("  yield8 hybrid guardrail contract OK")


def check_yield_attribution(model):
    rule = "[C:1]-[O:2]>>[C:1].[O:2]"
    result = model._rules_to_result_with_sources(
        "CCO", [("fragment-a", rule), ("fragment-b", rule)]
    )
    assert result is not None
    assert result["reactants"] == ["CC.O"]
    assert result["fragment_sources"] == [["fragment-a", "fragment-b"]]

    filtered = model._filter_with_attribution(
        ["CC"], ["C.C"], ["CC"], ["fragment-a"]
    )
    assert len(filtered) == 1
    assert filtered[0]["fragment"] == "fragment-a"

    model._stats_lock = threading.Lock()
    item = {
        "target": "CCO",
        "task_id": 7,
        "selection_metadata": {"capacity_limited": False},
        "fragment_stats": {
            "fragment-a": {
                "selected_rank": 1,
                "fragment": "fragment-a",
                "families": ["r3"],
                "dict_rule_count": 1,
                "dict_applicable_rule_count": 1,
                "augmentation_count": 1,
                "retro_raw_count": 3,
                "retro_valid_count": 3,
                "forward_consistent_count": 1,
                "mapped_count": 1,
                "template_extracted_count": 1,
                "full_product_reactions": {"CC.O"},
            }
        },
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "fragment-yield-{pid}.jsonl")
        os.environ["TP_FREE_FRAGMENT_YIELD_LOG"] = path
        model._write_fragment_yield_telemetry(item, TOPK)
        rendered = path.format(pid=os.getpid())
        with open(rendered) as handle:
            row = json.loads(handle.readline())
        assert row["full_product_unique_reaction_count"] == 1
        assert row["retro_raw_count"] == 3
    os.environ.pop("TP_FREE_FRAGMENT_YIELD_LOG", None)
    print("  fragment attribution and telemetry contract OK")


def check_run_batch_attribution(model):
    class EchoModel(object):
        def inference(self, rows):
            return list(rows)

    class FakeMapper(object):
        def map_reactions(self, reactions):
            return ["CCO>>CC.O" for _reaction in reactions]

    model.RD_list = RD_SINGLE
    model.use_DICT = False
    model.retro_topk = 1
    model.forward_topk = 1
    model.retro_model = EchoModel()
    model.forward_model = EchoModel()
    model.mapper = FakeMapper()
    model._stats_lock = threading.Lock()
    model._dict_ref = defaultdict(list)
    model._per_target_stats = defaultdict(
        lambda: {
            "substructure_lookups": 0,
            "substructure_hits": 0,
            "new_keys": 0,
            "new_template_values": 0,
        }
    )
    model._global_total_substructure_lookups = 0
    model._global_total_substructure_hits = 0
    model._global_substructure_hit_counts = defaultdict(int)
    model._global_template_hit_counts = defaultdict(int)
    model.random_sampling = types.MethodType(
        lambda _self, _target, _rd_list, _topk: ["CCO", "CC"], model
    )
    rule = "[C:1]-[O:2]>>[C:1].[O:2]"
    model._extract_templates_from_mapped = types.MethodType(
        lambda _self, fragments, _mapped: [(fragment, rule) for fragment in fragments],
        model,
    )

    previous_sampler = os.environ.get("TP_FREE_CSS_SAMPLER")
    os.environ["TP_FREE_CSS_SAMPLER"] = "random"
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "run-batch-{pid}.jsonl")
        os.environ["TP_FREE_FRAGMENT_YIELD_LOG"] = path
        output = model.run_batch(["CCO"], topk=2, task_ids=[7])[0]
        assert output is not None
        assert output["reactants"] == ["CC.O"]
        assert set(output["fragment_sources"][0]) == {"CC", "CCO"}, output
        rendered = path.format(pid=os.getpid())
        with open(rendered) as handle:
            rows = [json.loads(line) for line in handle]
        assert len(rows) == 2
        by_fragment = {row["fragment"]: row for row in rows}
        assert set(by_fragment) == {"CC", "CCO"}
        for row in rows:
            assert row["augmentation_count"] >= 1
            assert row["retro_raw_count"] == row["augmentation_count"]
            assert row["retro_valid_count"] == row["retro_raw_count"]
            assert row["forward_consistent_count"] == row["retro_valid_count"]
        assert all(row["full_product_unique_reaction_count"] == 1 for row in rows)
    os.environ.pop("TP_FREE_FRAGMENT_YIELD_LOG", None)
    if previous_sampler is None:
        os.environ.pop("TP_FREE_CSS_SAMPLER", None)
    else:
        os.environ["TP_FREE_CSS_SAMPLER"] = previous_sampler
    print("  run_batch fragment attribution integration OK")


def check_effective_cache_contract(model):
    model.use_DICT = True
    model.random_sampling = types.MethodType(
        lambda _self, _target, _rd_list, _topk: ["CCO"], model
    )
    os.environ["TP_FREE_CSS_SAMPLER"] = "random"
    os.environ["TP_FREE_EFFECTIVE_CACHE"] = "1"

    model._dict_ref = {
        "CCO": ["[N:1]-[C:2]>>[N:1].[C:2]"],
    }
    miss = model._prepare_single_target("CCO", 1, task_id=8)
    miss_stats = miss["fragment_stats"]["CCO"]
    assert miss_stats["dict_rule_count"] == 1
    assert miss_stats["dict_applicable_rule_count"] == 0
    assert miss_stats["augmentation_count"] > 1
    assert miss["dict_rule_entries"] == []

    model._dict_ref = {
        "CCO": ["[C:1]-[O:2]>>[C:1].[O:2]"],
    }
    hit = model._prepare_single_target("CCO", 1, task_id=9)
    hit_stats = hit["fragment_stats"]["CCO"]
    assert hit_stats["dict_rule_count"] == 1
    assert hit_stats["dict_applicable_rule_count"] == 1
    assert hit_stats["augmentation_count"] == 1
    assert len(hit["dict_rule_entries"]) == 1

    os.environ.pop("TP_FREE_EFFECTIVE_CACHE", None)
    model.use_DICT = False
    model._dict_ref = defaultdict(list)
    print("  effective-cache applicability contract OK")


def check_anchor8_contract(model, smi):
    from mlp_retrosyn.css_anchor import select_anchor_fragments, _canonical
    from mlp_retrosyn.css_effective_yield import (
        enumerate_effective_yield_candidates,
    )

    os.environ["TP_FREE_CSS_STRICT_TOPK"] = "1"
    model._dict_ref = {}
    random.seed(29)
    os.environ["TP_FREE_CSS_SAMPLER"] = "random"
    baseline = model.random_sampling(smi, [(7, 0), (3, 0)], TOPK)
    baseline_canonical = {_canonical(fragment) for fragment in baseline}
    random.seed(29)
    os.environ["TP_FREE_CSS_SAMPLER"] = "anchor8"
    anchored = model.random_sampling(smi, [(7, 0), (3, 0)], TOPK)
    anchored_canonical = {_canonical(fragment) for fragment in anchored}
    assert baseline_canonical == anchored_canonical, (
        baseline_canonical,
        anchored_canonical,
    )
    os.environ.pop("TP_FREE_CSS_STRICT_TOPK", None)

    chain = "CCCCCCCCCCCCCCCC"
    whole = _canonical(chain)
    random.seed(7)
    _probe, probe_meta = select_anchor_fragments(
        chain, TOPK, lambda fragment: set())
    kept = {
        row["smiles"]
        for row in probe_meta["selected"]
        if row["families"] == ["anchor_base"]
    }
    all_candidates = {
        record["smiles"]
        for record in enumerate_effective_yield_candidates(chain)
    }
    outside = sorted(all_candidates - kept - {whole})
    assert outside, (kept, all_candidates)
    known = {fragment: {"reaction-%s" % fragment} for fragment in outside[:3]}

    random.seed(7)
    output, metadata = select_anchor_fragments(
        chain, TOPK, lambda fragment: known.get(fragment, set()))
    assert metadata["lost_slot_count"] > 0
    assert 1 <= metadata["replaced_count"] <= 2
    reclaimed = [
        row["smiles"]
        for row in metadata["selected"]
        if "anchor_reclaim" in row["families"]
    ]
    assert reclaimed and all(fragment in output for fragment in reclaimed)

    random.seed(7)
    _output_margin, margin_meta = select_anchor_fragments(
        chain, TOPK, lambda fragment: known.get(fragment, set()), margin=99)
    assert margin_meta["replaced_count"] == 0
    assert margin_meta["rejected_candidate_count"] > 0

    random.seed(7)
    _output_cap, cap_meta = select_anchor_fragments(
        chain, TOPK, lambda fragment: known.get(fragment, set()), max_replace=1)
    assert cap_meta["replaced_count"] <= 1

    random.seed(11)
    first, first_meta = select_anchor_fragments(
        chain, TOPK, lambda fragment: known.get(fragment, set()))
    random.seed(11)
    second, second_meta = select_anchor_fragments(
        chain, TOPK, lambda fragment: known.get(fragment, set()))
    assert first == second and first_meta == second_meta

    small, small_meta = select_anchor_fragments("C", TOPK, lambda fragment: set())
    assert small == ["C"] and small_meta["capacity_limited"] is True
    print("  anchor8 baseline-anchor and micro-replacement contract OK")


def check_anchor8_candidate_telemetry(model):
    class EchoModel(object):
        def inference(self, rows):
            return list(rows)

    class FakeMapper(object):
        def map_reactions(self, reactions):
            return ["CCO>>CC.O" for _reaction in reactions]

    model.RD_list = RD_SINGLE
    model.use_DICT = False
    model.retro_topk = 1
    model.forward_topk = 1
    model.retro_model = EchoModel()
    model.forward_model = EchoModel()
    model.mapper = FakeMapper()
    model._stats_lock = threading.Lock()
    model._dict_ref = defaultdict(list)
    model._per_target_stats = defaultdict(
        lambda: {
            "substructure_lookups": 0,
            "substructure_hits": 0,
            "new_keys": 0,
            "new_template_values": 0,
        }
    )
    model._global_total_substructure_lookups = 0
    model._global_total_substructure_hits = 0
    model._global_substructure_hit_counts = defaultdict(int)
    model._global_template_hit_counts = defaultdict(int)
    model._anchor8_sampling = types.MethodType(
        lambda _self, _target, _topk: (
            ["CCO", "CC"], {"selected": [], "capacity_limited": False}
        ),
        model,
    )
    rule = "[C:1]-[O:2]>>[C:1].[C:2]"
    model._extract_templates_from_mapped_indexed = types.MethodType(
        lambda _self, fragments, _mapped: [
            (index, fragment, rule) for index, fragment in enumerate(fragments)
        ],
        model,
    )

    previous_sampler = os.environ.get("TP_FREE_CSS_SAMPLER")
    os.environ["TP_FREE_CSS_SAMPLER"] = "anchor8"
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "candidates-{pid}.jsonl")
        os.environ["TP_FREE_RETRO_CANDIDATE_LOG"] = path
        output = model.run_batch(["CCO"], topk=2, task_ids=[7])[0]
        assert output is not None
        rendered = path.format(pid=os.getpid())
        with open(rendered) as handle:
            rows = [json.loads(line) for line in handle]
        retro_rows = [row for row in rows if row["grain"] == "retro_candidate"]
        reaction_rows = [row for row in rows if row["grain"] == "reaction"]
        assert len(retro_rows) >= 2, rows
        assert all(row["retro_valid"] for row in retro_rows)
        assert all(row["forward_consistent"] for row in retro_rows)
        assert all(row["sampler"] == "anchor8" for row in retro_rows)
        assert {row["fragment"] for row in retro_rows} == {"CCO", "CC"}
        assert len(reaction_rows) >= 2, reaction_rows
        assert all(row["mapped"] for row in reaction_rows)
        assert all(row["template_extracted"] for row in reaction_rows)
        assert all(row["in_full_product_output"] is False for row in reaction_rows)
    os.environ.pop("TP_FREE_RETRO_CANDIDATE_LOG", None)
    if previous_sampler is None:
        os.environ.pop("TP_FREE_CSS_SAMPLER", None)
    else:
        os.environ["TP_FREE_CSS_SAMPLER"] = previous_sampler
    print("  anchor8 per-candidate telemetry contract OK")


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
    check_yield8_contract(model, targets[-1])
    check_strict_baseline_contract(model, targets[-1])
    check_yield8_hybrid_contract(model, targets[-1])
    check_anchor8_contract(model, targets[-1])
    check_anchor8_candidate_telemetry(model)
    check_yield_attribution(model)
    check_run_batch_attribution(model)
    check_effective_cache_contract(model)
    os.environ.pop("TP_FREE_CSS_SAMPLER", None)
    print("ALL CSS SAMPLER SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main(sys.argv)
