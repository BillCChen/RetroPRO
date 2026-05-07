"""Serializable inference / planning parameters for correlating results with configuration."""
from __future__ import print_function

import ast
import os


def build_inference_run_params(args):
    """
    Build a JSON/pickle-friendly dict of settings that affect template-free DICT and planning.

    Parameters
    ----------
    args : argparse.Namespace or similar
        Typically ``common.parse_args.args``.
    """
    rd_raw = getattr(args, 'RD_list', None)
    rd_parsed = None
    if rd_raw is not None:
        try:
            rd_parsed = ast.literal_eval(rd_raw) if isinstance(rd_raw, str) else list(rd_raw)
        except Exception:
            rd_parsed = None

    tp_free_env = {k: os.environ[k] for k in sorted(os.environ) if k.startswith('TP_FREE_')}

    return {
        'test_routes': getattr(args, 'test_routes', None),
        'route_limit': int(getattr(args, 'route_limit', 0) or 0),
        'starting_molecules': getattr(args, 'starting_molecules', None),
        'one_step_type': getattr(args, 'one_step_type', None),
        'retro_topk': int(getattr(args, 'retro_topk', 0) or 0),
        'forward_topk': int(getattr(args, 'forward_topk', 0) or 0),
        'expansion_topk': int(getattr(args, 'expansion_topk', 0) or 0),
        'RD_list': rd_raw,
        'RD_list_parsed': rd_parsed,
        'CSS': bool(getattr(args, 'CSS', False)),
        'DICT': bool(getattr(args, 'DICT', False)),
        'iterations': int(getattr(args, 'iterations', 0) or 0),
        'parallel_num': int(getattr(args, 'parallel_num', 0) or 0),
        'multi_pool': bool(getattr(args, 'multi_pool', False)),
        'gpu': int(getattr(args, 'gpu', -1)),
        'gpu_list': str(getattr(args, 'gpu_list', '') or ''),
        'seed': int(getattr(args, 'seed', 0) or 0),
        'retro_model_path': getattr(args, 'retro_model_path', None),
        'forward_model_path': getattr(args, 'forward_model_path', None),
        'mlp_templates': getattr(args, 'mlp_templates', None),
        'mlp_model_dump': getattr(args, 'mlp_model_dump', None),
        'use_value_fn': bool(getattr(args, 'use_value_fn', False)),
        'value_model': getattr(args, 'value_model', None),
        'result_folder': getattr(args, 'result_folder', None),
        'tp_free_env': tp_free_env,
    }
