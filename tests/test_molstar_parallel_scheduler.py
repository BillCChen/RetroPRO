import importlib.util
from pathlib import Path


_MODULE_PATH = Path(__file__).resolve().parents[1] / 'retro_star' / 'alg' / 'molstar_parallel.py'
_spec = importlib.util.spec_from_file_location('molstar_parallel_module', _MODULE_PATH)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
molstar_parallel = _module.molstar_parallel


class FakeTask:
    PLAN = {}

    def __init__(
        self,
        target_mol,
        target_mol_id,
        starting_mols,
        value_fn,
        iterations,
        viz=False,
        viz_dir=None,
    ):
        cfg = self.PLAN[target_mol]
        self.target_mol = target_mol
        self.target_mol_id = target_mol_id
        self.steps_needed = int(cfg['steps'])
        self.succ = bool(cfg['succ'])
        self.steps_taken = 0
        self.done = False

    def next_frontier_smiles(self):
        if self.done:
            return None
        if self.steps_taken >= self.steps_needed:
            self.done = True
            return None
        self.steps_taken += 1
        return f"{self.target_mol}|{self.steps_taken}"

    def apply_expansion(self, result):
        if self.steps_taken >= self.steps_needed:
            self.done = True

    def is_done(self):
        return self.done

    def finalize_message(self):
        msg = ({'target': self.target_mol}, self.steps_taken, self.steps_taken * 10)
        return self.succ, msg


def test_pool_refill_and_completion_order():
    FakeTask.PLAN = {
        'A': {'steps': 2, 'succ': True},
        'B': {'steps': 4, 'succ': False},
        'C': {'steps': 1, 'succ': True},
    }
    completion_order = []
    batch_sizes = []

    def expand_batch(smiles_batch):
        batch_sizes.append(len(smiles_batch))
        return [None] * len(smiles_batch)

    def on_task_done(task_idx, succ, msg):
        completion_order.append((task_idx, succ, msg[1]))

    targets = ['A', 'B', 'C']
    outputs = molstar_parallel(
        target_mols=targets,
        starting_mols=set(),
        expand_batch_fn=expand_batch,
        value_fn=lambda _: 0.0,
        iterations=10,
        pool_size=2,
        task_factory=FakeTask,
        on_task_done=on_task_done,
    )

    assert [out[0] for out in outputs] == [True, False, True]
    assert [out[1][1] for out in outputs] == [2, 4, 1]
    assert [x[0] for x in completion_order] == [0, 2, 1]
    assert batch_sizes == [2, 2, 2, 1]


def test_batch_output_mismatch_is_handled():
    FakeTask.PLAN = {
        'X': {'steps': 1, 'succ': True},
        'Y': {'steps': 2, 'succ': True},
    }

    def expand_batch(smiles_batch):
        return [None]

    outputs = molstar_parallel(
        target_mols=['X', 'Y'],
        starting_mols=set(),
        expand_batch_fn=expand_batch,
        value_fn=lambda _: 0.0,
        iterations=10,
        pool_size=2,
        task_factory=FakeTask,
    )

    assert outputs[0][1][1] == 1
    assert outputs[1][1][1] == 2
