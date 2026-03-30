import logging
import os
import pickle
import random
import time

import numpy as np
import torch

from common.parse_args import args
from common.prepare_utils import prepare_mlp, prepare_molstar_planner, prepare_r_smiles, prepare_starting_molecules
from common.smiles_to_fp import smiles_to_fp
from model import ValueMLP

from alg.molstar_parallel import molstar_parallel


def _resolve_runtime_config():
    mode = os.environ.get('RETROPRO_DEVICE', 'auto').strip().lower()
    cuda_ok = torch.cuda.is_available()

    one_step_device = -1
    value_device = torch.device('cpu')

    if mode == 'cpu':
        one_step_device = -1
        value_device = torch.device('cpu')
    elif mode == 'cuda':
        if args.gpu >= 0 and cuda_ok:
            one_step_device = args.gpu
            value_device = torch.device('cuda')
        else:
            logging.info('RETROPRO_DEVICE=cuda but CUDA unavailable or gpu<0; fallback to CPU.')
    else:
        if args.gpu >= 0 and cuda_ok:
            one_step_device = args.gpu
            value_device = torch.device('cuda')
        else:
            one_step_device = -1
            value_device = torch.device('cpu')

    return one_step_device, value_device


def _wrap_targets_as_routes(targets):
    return [[target + '>'] for target in targets if target]


def _load_routes_from_file(route_file):
    if route_file.endswith('.pkl'):
        routes = pickle.load(open(route_file, 'rb'))
        if len(routes) == 0:
            return []
        first = routes[0]
        if isinstance(first, (list, tuple)):
            logging.info('%d routes extracted from %s loaded', len(routes), route_file)
            return routes
        if isinstance(first, str):
            targets = [x.split('>')[0] for x in routes]
            wrapped = _wrap_targets_as_routes(targets)
            logging.info('%d targets extracted from %s loaded', len(wrapped), route_file)
            return wrapped
        raise ValueError('Unsupported route pickle format: %s' % type(first))

    targets = []
    with open(route_file, 'r') as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if '>' in line:
                targets.append(line.split('>')[0])
                continue
            if "'" in line:
                pieces = line.split("'")
                if len(pieces) >= 2 and pieces[1]:
                    targets.append(pieces[1])
                    continue
            targets.append(line)

    wrapped = _wrap_targets_as_routes(targets)
    logging.info('%d targets extracted from %s loaded', len(wrapped), route_file)
    return wrapped


def _load_routes(test_routes):
    if os.path.exists(test_routes):
        return _load_routes_from_file(test_routes)

    if test_routes == "uspto190":
        route_file = "dataset/routes_possible_test_hard.pkl"
        routes = pickle.load(open(route_file, 'rb'))
        logging.info('%d routes extracted from %s loaded', len(routes), route_file)
        return routes
    if test_routes == "pth_hard":
        file = "dataset/pistachio_hard_targets.txt"
        with open(file, 'r') as f:
            lines = [line.strip().split("'")[1] for line in f.readlines()]
        return [[line + '>'] for line in lines]
    if test_routes == "pth_reach":
        file = "dataset/pistachio_reachable_targets.txt"
        with open(file, 'r') as f:
            lines = [line.strip().split("'")[1] for line in f.readlines()]
        return [[line + '>'] for line in lines]
    if test_routes == "8XIK_olorofim":
        file = "dataset/8XIK_olorofim.txt"
        with open(file, 'r') as f:
            lines = [line.strip() for line in f.readlines()]
        return [[line + '>'] for line in lines]
    if test_routes == "8XIK_NCI":
        file = "/home/chenqixuan/retro_star/retro_star/dataset/8XIK_NVI_PAI.txt"
        return _load_routes_from_file(file)
    raise ValueError("Unknown test routes dataset: %s" % test_routes)


def _build_one_step_model(runtime_gpu):
    if args.one_step_type == "template_based":
        return prepare_mlp(args.mlp_templates, args.mlp_model_dump, device=runtime_gpu)
    if args.one_step_type == "template_free":
        import ast

        args.RD_list = ast.literal_eval(args.RD_list)
        logging.info("Parsed RD_list: %s", args.RD_list)
        return prepare_r_smiles(
            args.retro_model_path,
            args.retro_topk,
            args.forward_model_path,
            args.forward_topk,
            args.CSS,
            args.RD_list,
            args.DICT,
            device=runtime_gpu,
        )
    raise ValueError("Unknown one step model type: %s" % args.one_step_type)


def _build_value_fn(device):
    if not args.use_value_fn:
        return lambda x: 0.0

    model = ValueMLP(
        n_layers=args.n_layers,
        fp_dim=args.fp_dim,
        latent_dim=args.latent_dim,
        dropout_rate=0.1,
        device=device,
    ).to(device)
    model_f = '%s/%s' % (args.save_folder, args.value_model)
    logging.info('Loading value nn from %s', model_f)
    model.load_state_dict(torch.load(model_f, map_location=device))
    model.eval()

    def value_fn(mol):
        fp = smiles_to_fp(mol, fp_dim=args.fp_dim).reshape(1, -1)
        fp = torch.FloatTensor(fp).to(device)
        return model(fp).item()

    return value_fn


def _build_expand_batch_fn(one_step):
    topk = args.expansion_topk
    disable_run_batch = os.environ.get('RETROPRO_DISABLE_RUN_BATCH', '0') == '1'

    if (not disable_run_batch) and hasattr(one_step, 'run_batch') and callable(one_step.run_batch):
        logging.info('One-step model supports run_batch; enabling batch expansion path.')

        def expand_batch(smiles_batch):
            return one_step.run_batch(smiles_batch, topk=topk)

        return expand_batch

    if disable_run_batch:
        logging.info('RETROPRO_DISABLE_RUN_BATCH=1; forcing per-item expansion fallback.')
    logging.info('One-step model does not provide run_batch; fallback to per-item expansion.')

    def expand_batch(smiles_batch):
        outputs = []
        for smiles in smiles_batch:
            try:
                outputs.append(one_step.run(smiles, topk=topk))
            except Exception as exc:
                logging.info('Expansion error for %s: %s', smiles, exc)
                outputs.append(None)
        return outputs

    return expand_batch


def _init_result(num_targets):
    return {
        'succ': [None] * num_targets,
        'cumulated_time': [None] * num_targets,
        'iter': [None] * num_targets,
        'routes': [None] * num_targets,
        'route_costs': [None] * num_targets,
        'route_lens': [None] * num_targets,
        'final_node': [None] * num_targets,
    }


def _record_result(result, idx, succ, msg, elapsed):
    route = None
    iter_used = args.iterations
    final_node = None
    if msg is not None:
        if len(msg) > 0:
            route = msg[0]
        if len(msg) > 1 and msg[1] is not None:
            iter_used = msg[1]
        if len(msg) > 2:
            final_node = msg[2]

    result['succ'][idx] = bool(succ)
    result['cumulated_time'][idx] = elapsed
    result['iter'][idx] = iter_used
    result['routes'][idx] = route
    result['final_node'][idx] = final_node

    if succ and route is not None:
        result['route_costs'][idx] = route.total_cost
        result['route_lens'][idx] = route.length
    else:
        result['route_costs'][idx] = None
        result['route_lens'][idx] = None


def _save_plan(result):
    with open(args.result_folder + '/plan.pkl', 'wb') as f:
        pickle.dump(result, f)


def _log_progress(result, done_count, num_targets, t0):
    succ_flags = [bool(x) for x in result['succ'] if x is not None]
    tot_succ = int(np.array(succ_flags).sum()) if len(succ_flags) > 0 else 0
    avg_time = (time.time() - t0) / max(done_count, 1)
    done_iters = [x for x in result['iter'] if x is not None]
    avg_iter = float(np.array(done_iters, dtype=float).mean()) if len(done_iters) > 0 else float('nan')
    logging.info(
        'Succ: %d/%d/%d | avg time: %.2f s | avg iter: %.2f',
        tot_succ,
        done_count,
        num_targets,
        avg_time,
        avg_iter,
    )


def _run_serial(target_mols, starting_mols, one_step, value_fn):
    plan_handle = prepare_molstar_planner(
        one_step=one_step,
        value_fn=value_fn,
        starting_mols=starting_mols,
        expansion_topk=args.expansion_topk,
        iterations=args.iterations,
        viz=args.viz,
        viz_dir=args.viz_dir,
    )

    num_targets = len(target_mols)
    result = _init_result(num_targets)
    t0 = time.time()

    for idx, target_mol in enumerate(target_mols):
        try:
            succ, msg = plan_handle(target_mol, idx)
        except Exception as exc:
            logging.info('Error planning for target %d: %s', idx, exc)
            succ, msg = False, (None, args.iterations, None)

        _record_result(result, idx, succ, msg, time.time() - t0)
        _log_progress(result, idx + 1, num_targets, t0)
        _save_plan(result)

    return result


def _run_parallel(target_mols, starting_mols, one_step, value_fn):
    num_targets = len(target_mols)
    result = _init_result(num_targets)
    t0 = time.time()
    done_counter = {'count': 0}

    def on_task_done(task_idx, succ, msg):
        done_counter['count'] += 1
        _record_result(result, task_idx, succ, msg, time.time() - t0)
        _log_progress(result, done_counter['count'], num_targets, t0)
        _save_plan(result)

    batch_expand_fn = _build_expand_batch_fn(one_step)
    finished = molstar_parallel(
        target_mols=target_mols,
        starting_mols=starting_mols,
        expand_batch_fn=batch_expand_fn,
        value_fn=value_fn,
        iterations=args.iterations,
        pool_size=args.parallel_num,
        viz=args.viz,
        viz_dir=args.viz_dir,
        on_task_done=on_task_done,
    )

    for idx, output in enumerate(finished):
        if result['iter'][idx] is not None:
            continue
        if output is None:
            succ, msg = False, (None, args.iterations, None)
        else:
            succ, msg = output
        _record_result(result, idx, succ, msg, time.time() - t0)

    _save_plan(result)
    return result


def retro_plan():
    runtime_gpu, device = _resolve_runtime_config()

    starting_mols = prepare_starting_molecules(args.starting_molecules)
    routes = _load_routes(args.test_routes)
    target_mols = [route[0].split('>')[0] for route in routes]

    one_step = _build_one_step_model(runtime_gpu)
    value_fn = _build_value_fn(device)

    if not os.path.exists(args.result_folder):
        os.mkdir(args.result_folder)

    if args.multi_pool:
        logging.info(
            'Running multi-pool planner: pool=%d, iterations=%d, topk=%d',
            args.parallel_num,
            args.iterations,
            args.expansion_topk,
        )
        return _run_parallel(target_mols, starting_mols, one_step, value_fn)

    logging.info('Running legacy serial planner.')
    return _run_serial(target_mols, starting_mols, one_step, value_fn)


if __name__ == '__main__':
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    retro_plan()
