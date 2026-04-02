import sys
import types
import tempfile
from collections import defaultdict
from pathlib import Path


def _install_dependency_stubs():
    if 'onmt.translate.translator' not in sys.modules:
        translator_mod = types.ModuleType('onmt.translate.translator')

        class _DummyTranslator:
            def translate(self, src=None, tgt=None, batch_size=1, attn_debug=False, batch_type='sents'):
                count = len(src or [])
                return [0.0] * count, [['C'] for _ in range(count)]

        def build_translator(opt, report_score=False, out_file=None):
            return _DummyTranslator()

        translator_mod.build_translator = build_translator
        translate_mod = types.ModuleType('onmt.translate')
        onmt_mod = types.ModuleType('onmt')
        sys.modules['onmt'] = onmt_mod
        sys.modules['onmt.translate'] = translate_mod
        sys.modules['onmt.translate.translator'] = translator_mod

    if 'rxnmapper' not in sys.modules:
        rxnmapper_mod = types.ModuleType('rxnmapper')

        class BatchedMapper:
            def __init__(self, batch_size=10):
                self.batch_size = batch_size

            def map_reactions(self, reactions):
                return reactions

        class RXNMapper:
            pass

        rxnmapper_mod.BatchedMapper = BatchedMapper
        rxnmapper_mod.RXNMapper = RXNMapper
        sys.modules['rxnmapper'] = rxnmapper_mod


_install_dependency_stubs()
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / 'retro_star' / 'packages' / 'mlp_retrosyn'))
from mlp_retrosyn.tp_free_inference import TP_free_Model


class _FakeRetroModel:
    def __init__(self):
        self.calls = []

    def inference(self, smiles_list):
        self.calls.append(list(smiles_list))
        return [f"retro:{s}" for s in smiles_list]


class _FakeForwardModel:
    def __init__(self):
        self.calls = []

    def inference(self, retro_list):
        self.calls.append(list(retro_list))
        return [f"fwd:{r}" for r in retro_list]


def test_run_batch_batches_model_calls():
    model = object.__new__(TP_free_Model)
    model.use_DICT = False
    model.retro_topk = 1
    model.forward_topk = 1
    model.retro_model = _FakeRetroModel()
    model.forward_model = _FakeForwardModel()
    model.DICT = {}
    model._prepare_single_target = lambda x, topk: {
        'target': x,
        'aug_smiles': [f'{x}_1', f'{x}_2'],
        'dict_rules': [],
    }
    model._is_valid_retro = lambda r: True
    model.filter = lambda target, smiles, retro, forward: [
        (target, f'{smiles[i]}>>{retro[i]}') for i in range(len(smiles))
    ]
    model.extract_templates = lambda reactions: [('k', f'rule_{len(reactions)}')] if reactions else []
    model.renew_DICT = lambda templates: None
    model._rules_to_result = lambda target, rules: {
        'reactants': [target],
        'scores': [1.0],
        'template': rules,
    } if rules else None

    outputs = TP_free_Model.run_batch(model, ['A', 'B'], topk=8)

    assert len(model.retro_model.calls) == 1
    assert model.retro_model.calls[0] == ['A_1', 'A_2', 'B_1', 'B_2']
    assert len(model.forward_model.calls) == 1
    assert len(model.forward_model.calls[0]) == 4
    assert len(outputs) == 2
    assert outputs[0]['template'] == ['rule_2']
    assert outputs[1]['template'] == ['rule_2']


def test_run_delegates_to_run_batch():
    model = object.__new__(TP_free_Model)
    model.run_batch = lambda x_list, topk=20: [{'target': x_list[0], 'topk': topk}]

    out = TP_free_Model.run(model, 'CCC', topk=7)

    assert out == {'target': 'CCC', 'topk': 7}


def test_dict_dump_behaviors():
    model = object.__new__(TP_free_Model)
    model.use_DICT = True
    model.DICT = defaultdict(list)
    model.dict_dump_every = 1
    model.dict_dump_dir = tempfile.mkdtemp(prefix="tp_free_dump_")
    model.dict_dump_on_exit = True
    model._dict_update_count = 0
    model._dict_dump_serial = 0

    TP_free_Model.renew_DICT(model, [('A', 'rule1')])
    first = sorted(Path(model.dict_dump_dir).glob("*.pkl"))
    assert len(first) == 1

    TP_free_Model.renew_DICT(model, [('A', 'rule2')])
    second = sorted(Path(model.dict_dump_dir).glob("*.pkl"))
    assert len(second) == 2

    # force dump should always produce a snapshot when DICT has content
    forced = TP_free_Model.finalize_dict_cache(model)
    assert forced is not None
    third = sorted(Path(model.dict_dump_dir).glob("*.pkl"))
    assert len(third) == 3


def main():
    test_run_batch_batches_model_calls()
    test_run_delegates_to_run_batch()
    test_dict_dump_behaviors()
    print('tp_free_batch_smoke_test: PASS')


if __name__ == '__main__':
    main()
