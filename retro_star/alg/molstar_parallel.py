import logging

import sys
sys.path.append("/home/chenqixuan/retro_star/retro_star/alg")


def _safe_batch_expand(expand_batch_fn, smiles_batch, task_ids):
    if len(smiles_batch) == 0:
        return []
    outputs = None
    try:
        outputs = expand_batch_fn(smiles_batch, task_ids)
    except TypeError:
        try:
            outputs = expand_batch_fn(smiles_batch)
        except Exception as exc:
            logging.info("Batch expansion failed: %s", exc)
            return [None] * len(smiles_batch)
    except Exception as exc:
        logging.info("Batch expansion failed: %s", exc)
        return [None] * len(smiles_batch)

    if outputs is None:
        return [None] * len(smiles_batch)

    if len(outputs) != len(smiles_batch):
        logging.info(
            "Batch output size mismatch: expected=%d got=%d",
            len(smiles_batch),
            len(outputs),
        )
        if len(outputs) < len(smiles_batch):
            outputs = list(outputs) + [None] * (len(smiles_batch) - len(outputs))
        else:
            outputs = list(outputs[:len(smiles_batch)])

    return outputs


def molstar_parallel(
    target_mols,
    starting_mols,
    expand_batch_fn,
    value_fn,
    iterations,
    pool_size=8,
    viz=False,
    viz_dir=None,
    on_task_done=None,
    task_factory=None,
):
    """Run fixed-width multi-molecule planning with batched one-step expansion.

    Returns:
        List[(succ, msg)] aligned with target_mols order.
    """

    num_targets = len(target_mols)
    if num_targets == 0:
        return []

    pool_size = max(1, int(pool_size))
    if task_factory is None:
        from alg.molstar_task import MolStarTask
        task_factory = MolStarTask

    next_target_idx = 0
    active = {}
    finished = [None] * num_targets

    def fill_pool():
        nonlocal next_target_idx
        while len(active) < pool_size and next_target_idx < num_targets:
            task_idx = next_target_idx
            target_mol = target_mols[task_idx]
            active[task_idx] = task_factory(
                target_mol=target_mol,
                target_mol_id=task_idx,
                starting_mols=starting_mols,
                value_fn=value_fn,
                iterations=iterations,
                viz=viz,
                viz_dir=viz_dir,
            )
            next_target_idx += 1

    fill_pool()
    schedule_round = 0

    while len(active) > 0:
        schedule_round += 1
        if schedule_round % 20 == 1:
            logging.info(
                "Scheduler round=%d active=%d finished=%d/%d",
                schedule_round,
                len(active),
                sum(x is not None for x in finished),
                num_targets,
            )

        batch_smiles = []
        batch_task_ids = []

        for task_idx in list(active.keys()):
            task = active[task_idx]
            smiles = task.next_frontier_smiles()
            if smiles is not None:
                batch_smiles.append(smiles)
                batch_task_ids.append(task_idx)

        if len(batch_smiles) > 0:
            batch_outputs = _safe_batch_expand(expand_batch_fn, batch_smiles, batch_task_ids)
            for task_idx, output in zip(batch_task_ids, batch_outputs):
                task = active.get(task_idx)
                if task is not None:
                    task.apply_expansion(output)

        done_ids = []
        for task_idx, task in active.items():
            if task.is_done():
                done_ids.append(task_idx)

        for task_idx in done_ids:
            task = active.pop(task_idx)
            task_result = task.finalize_message()
            finished[task_idx] = task_result
            if on_task_done is not None:
                try:
                    on_task_done(task_idx, task_result[0], task_result[1])
                except Exception as exc:
                    logging.info("on_task_done callback failed for task %d: %s", task_idx, exc)

        fill_pool()

    return finished
