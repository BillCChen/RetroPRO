import logging
import multiprocessing as mp
import os
import pickle
import random
import sys
import time
import inspect

import numpy as np
import torch

from common.parse_args import args
from common.inference_run_params import build_inference_run_params
from common.prepare_utils import prepare_mlp, prepare_molstar_planner, prepare_r_smiles, prepare_starting_molecules
from common.smiles_to_fp import smiles_to_fp
from model import ValueMLP

from alg.molstar_parallel import molstar_parallel


def _resolve_runtime_config():
    """Return (logical_cuda_index_for_models, value_torch_device).

    parse_args sets CUDA_VISIBLE_DEVICES=str(args.gpu) when args.gpu >= 0, so only one
    physical GPU is visible and it is always ``cuda:0`` inside this process. OpenNMT /
    TP_free_Model expect an integer 0 in that case, not args.gpu (would raise invalid
    device ordinal for --gpu 1,2,...).
    """
    gpu_id = args.gpu
    if gpu_id >= 0 and not torch.cuda.is_available():
        logging.info('Requested gpu=%d but CUDA is unavailable; fallback to CPU/MPS where possible.', gpu_id)
        gpu_id = -1

    if gpu_id >= 0:
        value_device = torch.device('cuda')
        logical_cuda_index = 0
        if args.gpu != 0:
            logging.info(
                'Physical GPU %d selected (CUDA_VISIBLE_DEVICES=%s); using logical cuda:%d for models.',
                args.gpu,
                os.environ.get('CUDA_VISIBLE_DEVICES', ''),
                logical_cuda_index,
            )
        return logical_cuda_index, value_device
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        value_device = torch.device('mps')
        logging.info('Using MPS device for value model inference.')
    else:
        value_device = torch.device('cpu')
    return -1, value_device


def _load_routes(test_routes):
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
        with open(file, 'r') as f:
            lines = [line.strip() for line in f.readlines()]
        return [[line + '>'] for line in lines]

    if os.path.exists(test_routes):
        route_file = test_routes
        if route_file.endswith('.pkl'):
            routes = pickle.load(open(route_file, 'rb'))
            if not routes:
                logging.info('0 routes extracted from %s loaded', route_file)
                return []
            first = routes[0]
            if isinstance(first, str):
                normalized = [[item if '>' in item else item + '>'] for item in routes]
                logging.info('%d routes extracted from %s loaded', len(normalized), route_file)
                return normalized
            if isinstance(first, (list, tuple)):
                logging.info('%d routes extracted from %s loaded', len(routes), route_file)
                return routes
            raise ValueError("Unsupported route pickle format in %s: %s" % (route_file, type(first)))

        lines = []
        with open(route_file, 'r', encoding='utf-8', errors='ignore') as f:
            for raw in f.readlines():
                line = raw.strip()
                if not line:
                    continue
                if '>' in line:
                    target = line.split('>')[0]
                elif "'" in line:
                    parts = line.split("'")
                    target = parts[1] if len(parts) > 1 and parts[1] else line
                else:
                    target = line
                lines.append([target + '>'])
        logging.info('%d routes extracted from %s loaded', len(lines), route_file)
        return lines

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
    disable_batch = os.getenv('RETROPRO_DISABLE_RUN_BATCH', '0') == '1'

    if not disable_batch and hasattr(one_step, 'run_batch') and callable(one_step.run_batch):
        logging.info('One-step model supports run_batch; enabling batch expansion path.')

        batch_accepts_task_ids = 'task_ids' in inspect.signature(one_step.run_batch).parameters

        def expand_batch(smiles_batch, task_ids):
            if batch_accepts_task_ids:
                return one_step.run_batch(smiles_batch, topk=topk, task_ids=task_ids)
            return one_step.run_batch(smiles_batch, topk=topk)

        return expand_batch

    logging.info('One-step model does not provide run_batch (or batch disabled); fallback to per-item expansion.')

    run_accepts_task_id = 'task_id' in inspect.signature(one_step.run).parameters

    def expand_batch(smiles_batch, task_ids):
        outputs = []
        for smiles, tid in zip(smiles_batch, task_ids):
            try:
                if run_accepts_task_id:
                    outputs.append(one_step.run(smiles, topk=topk, task_id=tid))
                else:
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


def _attach_inference_run_params(result):
    """Attach reproducibility / correlation metadata (RD_list, topk, dataset, etc.)."""
    d = build_inference_run_params(args)
    if isinstance(result.get('succ'), list):
        d['num_targets'] = len(result['succ'])
    result['inference_run_params'] = d


def _save_plan(result):
    _attach_inference_run_params(result)
    with open(args.result_folder + '/plan.pkl', 'wb') as f:
        pickle.dump(result, f)


def _attach_dict_cache_report(result, one_step):
    if args.one_step_type != 'template_free' or not args.DICT:
        return
    if hasattr(one_step, 'get_dict_cache_report'):
        _attach_inference_run_params(result)
        report = one_step.get_dict_cache_report()
        rp = result.get('inference_run_params')
        if rp is not None:
            report = dict(report)
            report['run_params'] = rp
        result['dict_cache_report'] = report


def _log_dict_cache_summary(report):
    if not report:
        return
    g = report.get('global') or {}
    logging.info(
        'DICT cache global: keys=%s values=%s lookups=%s hits=%s rate=%s (aggregation=%s)',
        g.get('dict_num_keys'),
        g.get('dict_num_values'),
        g.get('substructure_lookups_total'),
        g.get('substructure_hits_total'),
        g.get('substructure_hit_rate'),
        report.get('aggregation_mode'),
    )
    per = report.get('per_target') or []
    if not per:
        return
    rates = [r.get('substructure_hit_rate') for r in per if r.get('substructure_hit_rate') is not None]
    if rates:
        logging.info(
            'DICT cache per-target mean: substructure_hit_rate=%.4f new_keys=%.4f new_template_values=%.4f (n=%d)',
            float(np.mean(rates)),
            float(np.mean([r.get('new_keys', 0) for r in per])),
            float(np.mean([r.get('new_template_values', 0) for r in per])),
            len(per),
        )


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

    _attach_dict_cache_report(result, one_step)
    _save_plan(result)
    _log_dict_cache_summary(result.get('dict_cache_report'))
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

    _attach_dict_cache_report(result, one_step)
    _save_plan(result)
    _log_dict_cache_summary(result.get('dict_cache_report'))
    return result


# ---------------------------------------------------------------------------
# Multi-GPU worker (must be a module-level function for multiprocessing pickle)
# ---------------------------------------------------------------------------

def _gpu_worker(
    worker_idx,
    gpu_id,
    mol_slice,
    idx_map,
    starting_mols,
    one_step_cfg,
    shared_dict,
    dict_lock,
    result_pkl_path,
    args_dict,
):
    """Per-GPU planning worker. Runs inside a separate process."""
    # Set GPU visibility BEFORE any CUDA/torch import in this process.
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    import logging as _log
    import time as _time
    import sys as _sys
    import pickle as _pickle
    import numpy as _np
    import torch as _torch

    retro_dir = os.path.dirname(os.path.abspath(__file__))
    for _p in [retro_dir, os.path.join(retro_dir, 'packages', 'mlp_retrosyn')]:
        if _p not in _sys.path:
            _sys.path.insert(0, _p)

    _log.basicConfig(
        level=_log.INFO,
        format=f'%(asctime)s [GPU{gpu_id}] %(message)s',
        datefmt='%m-%d %H:%M',
    )

    try:
        # --- Value function ---
        if args_dict.get('use_value_fn'):
            from model import ValueMLP
            from common.smiles_to_fp import smiles_to_fp as _smiles_to_fp
            _device = _torch.device('cuda') if _torch.cuda.is_available() else _torch.device('cpu')
            _vmodel = ValueMLP(
                n_layers=args_dict['n_layers'],
                fp_dim=args_dict['fp_dim'],
                latent_dim=args_dict['latent_dim'],
                dropout_rate=0.1,
                device=_device,
            ).to(_device)
            _model_f = os.path.join(args_dict['save_folder'], args_dict['value_model'])
            _vmodel.load_state_dict(_torch.load(_model_f, map_location=_device))
            _vmodel.eval()

            def value_fn(mol):
                fp = _smiles_to_fp(mol, fp_dim=args_dict['fp_dim']).reshape(1, -1)
                fp = _torch.FloatTensor(fp).to(_device)
                return _vmodel(fp).item()
        else:
            value_fn = lambda x: 0.0

        # --- One-step model with shared DICT ---
        from mlp_retrosyn.tp_free_inference import TP_free_Model
        _gpu_device = 0 if _torch.cuda.is_available() else -1
        one_step = TP_free_Model(
            retro_model_path=one_step_cfg['retro_model_path'],
            retro_topk=one_step_cfg['retro_topk'],
            forward_model_path=one_step_cfg['forward_model_path'],
            forwad_topk=one_step_cfg['forward_topk'],
            CCS=one_step_cfg['CSS'],
            RD_list=one_step_cfg['RD_list'],
            DICT=one_step_cfg['DICT'],
            device=_gpu_device,
            shared_dict=shared_dict,
            dict_lock=dict_lock,
        )

        # --- Planning ---
        from alg.molstar_parallel import molstar_parallel as _molstar_parallel

        topk = args_dict.get('expansion_topk', 8)

        def expand_batch(smiles_batch, task_ids):
            global_ids = [idx_map[t] for t in task_ids]
            return one_step.run_batch(smiles_batch, topk=topk, task_ids=global_ids)

        _t0 = _time.time()
        finished = _molstar_parallel(
            target_mols=mol_slice,
            starting_mols=starting_mols,
            expand_batch_fn=expand_batch,
            value_fn=value_fn,
            iterations=args_dict.get('iterations', 100),
            pool_size=args_dict.get('parallel_num', 64),
            viz=False,
            viz_dir=None,
        )
        _elapsed = _time.time() - _t0

        # --- Collect results keyed by original index ---
        worker_result = []
        for local_i, (orig_idx, output) in enumerate(zip(idx_map, finished)):
            mol_elapsed = _elapsed * (local_i + 1) / max(len(mol_slice), 1)
            if output is None:
                worker_result.append((orig_idx, False, (None, args_dict.get('iterations', 100), None), mol_elapsed))
            else:
                succ, msg = output
                worker_result.append((orig_idx, succ, msg, mol_elapsed))

        os.makedirs(os.path.dirname(result_pkl_path), exist_ok=True)
        with open(result_pkl_path, 'wb') as _f:
            _pickle.dump(worker_result, _f)
        _log.info('Worker %d done: %d molecules, elapsed=%.1fs', worker_idx, len(mol_slice), _elapsed)

    except Exception as exc:
        _log.error('Worker %d (GPU %d) failed: %s', worker_idx, gpu_id, exc, exc_info=True)
        # Write an empty result so the orchestrator doesn't hang.
        os.makedirs(os.path.dirname(result_pkl_path), exist_ok=True)
        with open(result_pkl_path, 'wb') as _f:
            _pickle.dump([], _f)


def _run_multi_gpu(target_mols, starting_mols, one_step_cfg):
    gpu_ids = [int(g.strip()) for g in args.gpu_list.split(',') if g.strip()]
    n = len(gpu_ids)
    num_targets = len(target_mols)

    # Round-robin partition: worker i owns molecules i, i+n, i+2n, ...
    mol_slices = [target_mols[i::n] for i in range(n)]
    idx_maps = [list(range(i, num_targets, n)) for i in range(n)]

    # Shared DICT across all workers via Manager
    manager = mp.Manager()
    shared_dict = manager.dict()
    dict_lock = manager.Lock()

    result_pkl_paths = [
        os.path.join(args.result_folder, f'worker_{i}_result.pkl') for i in range(n)
    ]

    args_dict = vars(args)

    logging.info('Launching %d GPU workers: gpu_ids=%s, molecules=%d', n, gpu_ids, num_targets)

    processes = []
    for i in range(n):
        p = mp.Process(
            target=_gpu_worker,
            args=(
                i,
                gpu_ids[i],
                mol_slices[i],
                idx_maps[i],
                starting_mols,
                one_step_cfg,
                shared_dict,
                dict_lock,
                result_pkl_paths[i],
                args_dict,
            ),
            name=f'gpu-worker-{i}',
        )
        p.start()
        processes.append(p)
        logging.info('Started worker %d on GPU %d (%d molecules)', i, gpu_ids[i], len(mol_slices[i]))

    for p in processes:
        p.join()
        if p.exitcode != 0:
            logging.warning('Worker %s exited with code %d', p.name, p.exitcode)

    # Merge results
    t0 = time.time()
    result = _init_result(num_targets)
    for pkl_path in result_pkl_paths:
        if not os.path.exists(pkl_path):
            logging.warning('Worker result file missing: %s', pkl_path)
            continue
        with open(pkl_path, 'rb') as f:
            worker_result = pickle.load(f)
        for orig_idx, succ, msg, elapsed in worker_result:
            _record_result(result, orig_idx, succ, msg, elapsed)

    if args.one_step_type == 'template_free' and args.DICT:
        nk = 0
        nv = 0
        for _k, rules in shared_dict.items():
            nk += 1
            nv += len(rules)
        _rp = build_inference_run_params(args)
        _rp['num_targets'] = num_targets
        result['dict_cache_report'] = {
            'aggregation_mode': 'multi_gpu_shared_dict_only',
            'per_target': [],
            'global': {
                'dict_num_keys': nk,
                'dict_num_values': nv,
                'substructure_lookups_total': None,
                'substructure_hits_total': None,
                'substructure_hit_rate': None,
                'substructure_hit_counts_topk': {},
                'template_hit_counts_topk': {},
            },
            'first_infer': {'key': {}, 'key_rule': {}},
            'run_params': _rp,
        }
        logging.info(
            'DICT cache (multi-GPU merged dict only): keys=%d values=%d',
            nk,
            nv,
        )

    _save_plan(result)
    _dump_shared_tp_free_dict_on_exit(shared_dict)
    succ_count = sum(1 for x in result['succ'] if x)
    logging.info('Multi-GPU planning done: %d/%d succeeded', succ_count, num_targets)
    return result


def _build_one_step_cfg():
    """Return a plain dict describing the one-step model config (picklable for workers)."""
    import ast
    rd_list = args.RD_list
    if isinstance(rd_list, str):
        rd_list = ast.literal_eval(rd_list)
    return {
        'retro_model_path': args.retro_model_path,
        'retro_topk': args.retro_topk,
        'forward_model_path': args.forward_model_path,
        'forward_topk': args.forward_topk,
        'CSS': args.CSS,
        'RD_list': rd_list,
        'DICT': args.DICT,
    }


def _ensure_tp_free_dict_dump_dir():
    """Periodic DICT dumps go next to plan.pkl when TP_FREE_DICT_DUMP_DIR is unset."""
    if args.DICT and args.one_step_type == 'template_free':
        if not os.environ.get('TP_FREE_DICT_DUMP_DIR'):
            os.environ['TP_FREE_DICT_DUMP_DIR'] = os.path.abspath(args.result_folder)
            logging.info(
                'TP_FREE_DICT_DUMP_DIR unset; defaulting to result folder: %s',
                os.environ['TP_FREE_DICT_DUMP_DIR'],
            )


def _dump_tp_free_dict_on_exit(one_step):
    """Write tp_free_DICT_final_*.pkl next to plan.pkl. Default on; set TP_FREE_DICT_DUMP_ON_EXIT=0 to skip."""
    if os.getenv('TP_FREE_DICT_DUMP_ON_EXIT', '1') != '1':
        return
    if one_step is None or not hasattr(one_step, 'save_dict_snapshot'):
        return
    if getattr(one_step, '_dict_is_shared', False):
        logging.info('TP_FREE_DICT_DUMP_ON_EXIT: skip local snapshot (shared dict / multi-GPU worker model).')
        return
    tag = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
    path = os.path.join(args.result_folder, 'tp_free_DICT_final_%s.pkl' % tag)
    if one_step.save_dict_snapshot(path):
        logging.info('TP_FREE DICT final snapshot: %s', path)


def _dump_shared_tp_free_dict_on_exit(shared_dict):
    """Multi-GPU main process: dump merged Manager dict (default on; TP_FREE_DICT_DUMP_ON_EXIT=0 to skip)."""
    if os.getenv('TP_FREE_DICT_DUMP_ON_EXIT', '1') != '1':
        return
    tag = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
    path = os.path.join(args.result_folder, 'tp_free_DICT_final_%s.pkl' % tag)
    with open(path, 'wb') as f:
        pickle.dump(dict(shared_dict), f)
    logging.info('TP_FREE shared DICT final snapshot: %s', path)


def retro_plan():
    runtime_gpu, device = _resolve_runtime_config()

    _ensure_tp_free_dict_dump_dir()

    starting_mols = prepare_starting_molecules(args.starting_molecules)
    routes = _load_routes(args.test_routes)
    if getattr(args, 'route_limit', 0) and args.route_limit > 0:
        routes = routes[: args.route_limit]
        logging.info('route_limit=%d: using first %d targets', args.route_limit, len(routes))

    target_mols = [route[0].split('>')[0] for route in routes]

    if not os.path.exists(args.result_folder):
        os.mkdir(args.result_folder)

    # Multi-GPU path: build a config dict instead of loading the model in the main process.
    if args.gpu_list and args.one_step_type == 'template_free':
        logging.info(
            'Multi-GPU mode: gpu_list=%s, pool=%d, iterations=%d, topk=%d',
            args.gpu_list, args.parallel_num, args.iterations, args.expansion_topk,
        )
        one_step_cfg = _build_one_step_cfg()
        return _run_multi_gpu(target_mols, starting_mols, one_step_cfg)

    one_step = _build_one_step_model(runtime_gpu)
    value_fn = _build_value_fn(device)

    if args.multi_pool:
        logging.info(
            'Running multi-pool planner: pool=%d, iterations=%d, topk=%d',
            args.parallel_num, args.iterations, args.expansion_topk,
        )
        out = _run_parallel(target_mols, starting_mols, one_step, value_fn)
        _dump_tp_free_dict_on_exit(one_step)
        return out

    logging.info('Running legacy serial planner.')
    out = _run_serial(target_mols, starting_mols, one_step, value_fn)
    _dump_tp_free_dict_on_exit(one_step)
    return out


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    retro_plan()
