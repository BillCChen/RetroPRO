import sys
import types
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


class _FakeMapper:
    def __init__(self):
        self.calls = []

    def map_reactions(self, reactions):
        self.calls.append(list(reactions))
        return list(reactions)  # pass-through; real format doesn't matter for stubs


def test_run_batch_batches_model_calls():
    model = object.__new__(TP_free_Model)
    model.use_DICT = False
    model.retro_topk = 1
    model.forward_topk = 1
    model.retro_model = _FakeRetroModel()
    model.forward_model = _FakeForwardModel()
    model.mapper = _FakeMapper()
    # New internal attributes added after refactor
    model._dict_ref = {}
    model._dict_is_shared = False
    model._prepare_single_target = lambda x, topk: {
        'target': x,
        'aug_smiles': [f'{x}_1', f'{x}_2'],
        'dict_rules': [],
    }
    model._is_valid_retro = lambda r: True
    model.filter = lambda target, smiles, retro, forward: [
        (target, f'{smiles[i]}>>{retro[i]}') for i in range(len(smiles))
    ]
    # Mirrors old extract_templates mock: returns one rule per reaction slice
    model._extract_templates_from_mapped = lambda ccs, mapped: [('k', f'rule_{len(ccs)}')] if ccs else []
    model.renew_DICT = lambda templates: None
    model._rules_to_result = lambda target, rules: {
        'reactants': [target],
        'scores': [1.0],
        'template': rules,
    } if rules else None

    outputs = TP_free_Model.run_batch(model, ['A', 'B'], topk=8)

    # Retro and forward models must each be called exactly once (batched across both molecules)
    assert len(model.retro_model.calls) == 1
    assert model.retro_model.calls[0] == ['A_1', 'A_2', 'B_1', 'B_2']
    assert len(model.forward_model.calls) == 1
    assert len(model.forward_model.calls[0]) == 4
    # Mapper must be called exactly once across all molecules (Bug 2 fix)
    assert len(model.mapper.calls) == 1
    assert len(model.mapper.calls[0]) == 4  # 2 reactions per molecule × 2 molecules
    assert len(outputs) == 2
    assert outputs[0]['template'] == ['rule_2']
    assert outputs[1]['template'] == ['rule_2']


def test_run_delegates_to_run_batch():
    model = object.__new__(TP_free_Model)
    model.run_batch = lambda x_list, topk=20: [{'target': x_list[0], 'topk': topk}]

    out = TP_free_Model.run(model, 'CCC', topk=7)

    assert out == {'target': 'CCC', 'topk': 7}


def main():
    test_run_batch_batches_model_calls()
    test_run_delegates_to_run_batch()
    print('tp_free_batch_smoke_test: PASS')


if __name__ == '__main__':
    main()
