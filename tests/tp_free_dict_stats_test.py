"""Lightweight tests for DICT cache statistics (no GPU / model weights)."""
import sys
import threading
from collections import defaultdict
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / 'retro_star' / 'packages' / 'mlp_retrosyn'))
from mlp_retrosyn.tp_free_inference import TP_free_Model


def _empty_per_target():
    return {
        'substructure_lookups': 0,
        'substructure_hits': 0,
        'new_keys': 0,
        'new_template_values': 0,
    }


def test_get_dict_cache_report_shape():
    m = object.__new__(TP_free_Model)
    m._dict_ref = defaultdict(list, {'k1': ['a', 'b']})
    m._dict_is_shared = False
    m._stats_lock = threading.Lock()
    m._per_target_stats = defaultdict(_empty_per_target)
    m._per_target_stats[0]['substructure_lookups'] = 10
    m._per_target_stats[0]['substructure_hits'] = 4
    m._global_substructure_hit_counts = defaultdict(int, {'sk': 3})
    m._global_template_hit_counts = defaultdict(int, {'tpl': 2})
    m._global_total_substructure_lookups = 10
    m._global_total_substructure_hits = 4
    m._key_first_infer = {}
    m._pair_first_infer = {}
    m._pair_key_sep = '|||'
    m._dict_stats_topk = 10

    r = TP_free_Model.get_dict_cache_report(m)
    assert r['aggregation_mode'] == 'full'
    assert r['global']['dict_num_keys'] == 1
    assert r['global']['dict_num_values'] == 2
    assert r['global']['substructure_hit_rate'] == 0.4
    assert len(r['per_target']) == 1
    assert r['per_target'][0]['substructure_hit_rate'] == 0.4


def test_renew_dict_first_infer():
    m = object.__new__(TP_free_Model)
    m._dict_ref = defaultdict(list)
    m._dict_is_shared = False
    m._dict_lock = None
    m._stats_lock = threading.Lock()
    m._per_target_stats = defaultdict(_empty_per_target)
    m._key_first_infer = {}
    m._pair_first_infer = {}
    m._pair_key_sep = '|||'
    m._dict_update_count = 0
    m.dict_dump_every = 0
    m.dict_dump_dir = ''
    m.dict_dump_with_meta = False

    TP_free_Model.renew_DICT(m, [('ck', 'r1')], task_id=0, target_smiles='TGT')
    assert 'ck' in m._key_first_infer
    assert m._key_first_infer['ck']['task_id'] == 0
    pk = TP_free_Model._pair_key_str(m, 'ck', 'r1')
    assert pk in m._pair_first_infer
    assert m._per_target_stats[0]['new_keys'] == 1
    assert m._per_target_stats[0]['new_template_values'] == 1

    TP_free_Model.renew_DICT(m, [('ck', 'r2')], task_id=1, target_smiles='OTHER')
    assert m._per_target_stats[1]['new_keys'] == 0
    assert m._per_target_stats[1]['new_template_values'] == 1


def main():
    test_get_dict_cache_report_shape()
    test_renew_dict_first_infer()
    print('tp_free_dict_stats_test: PASS')


if __name__ == '__main__':
    main()
