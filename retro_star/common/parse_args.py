import argparse
import logging
import os
import torch
import sys
from datetime import datetime


parser = argparse.ArgumentParser()
# for single test
# python retro_plan_single.py --seed 42 --use_value_fn --expansion_topk 8 --one_step_type template_free --CSS --DICT --iterations 101 --viz --gpu 0 --test_routes self_defined
# for template based model test
# python retro_plan_single.py --seed 42 --use_value_fn --expansion_topk 8 --one_step_type template_based --iterations 101 --viz --gpu 0 --test_routes self_defined
# for template free model test
# python retro_plan_single.py --seed 42 --use_value_fn --expansion_topk 8 --one_step_type template_free --CSS --DICT --iterations 101 --viz --gpu 0 --test_routes self_defined
# ===================== gpu id ===================== #
parser.add_argument('--gpu', type=int, required=True)

# =================== random seed ================== #
parser.add_argument('--seed', type=int, default=1234)

# ==================== dataset ===================== #
parser.add_argument('--test_routes',
                    default='uspto190')
parser.add_argument('--starting_molecules', default='dataset/origin_dict.csv')

# ================== value dataset ================= #
parser.add_argument('--value_root', default='dataset')
parser.add_argument('--value_train', default='train_mol_fp_value_step')
parser.add_argument('--value_val', default='val_mol_fp_value_step')
# ==mopdel type 
parser.add_argument('--one_step_type')
# ================== one-step model ================ #
parser.add_argument('--mlp_model_dump',
                    default='one_step_model/saved_rollout_state_1_2048.ckpt')
parser.add_argument('--mlp_templates',
                    default='one_step_model/template_rules_1.dat')
# ================== tempalte free model ================ #
parser.add_argument('--retro_model_path',
                    default='one_step_model/USPTO_full_PtoR.pt')
parser.add_argument('--retro_topk', type=int, default=3)
parser.add_argument('--forward_model_path',
                    default='one_step_model/USPTO-MIT_RtoP_mixed.pt')
parser.add_argument('--forward_topk', type=int, default=1)

parser.add_argument('--CSS', action='store_true')
# parser.add_argument('--R_list', type=str, default=[9])
# parser.add_argument('--D_list', type=str, default=[0])
parser.add_argument('--RD_list', type=str, default="[(9,0)]")
parser.add_argument('--DICT', action='store_true')

# ===================== all algs =================== #
parser.add_argument('--iterations', type=int, default=100)
parser.add_argument('--expansion_topk', type=int, default=50)
parser.add_argument('--viz', action='store_true')


# ===================== model ====================== #
parser.add_argument('--fp_dim', type=int, default=2048)
parser.add_argument('--n_layers', type=int, default=1)
parser.add_argument('--latent_dim', type=int, default=128)

# ==================== training ==================== #
parser.add_argument('--n_epochs', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--save_epoch_int', type=int, default=1)
parser.add_argument('--save_folder', default='saved_models')

# ==================== evaluation =================== #
parser.add_argument('--use_value_fn', action='store_true')
parser.add_argument('--value_model', default='best_epoch_final_4.pt')


# 并行化参数
parser.add_argument('--parallel_num', type=int, default=64, help='Number of parallel molecules in pool scheduler')
parser.add_argument('--parallel_expansions', type=int, default=1, help='Number of parallel expansions per molecule')
parser.add_argument('--use_priority_queue', action='store_true', help='Use priority queue for node selection')
parser.add_argument('--multi_pool', action='store_true', help='Enable fixed-width multi-molecule parallel planning')
parser.add_argument('--gpu_list', default='', help='Comma-separated GPU IDs for multi-GPU data-parallel mode, e.g. "0,1,2,3". Overrides --gpu when set.')

# 输出目录参数允许外部覆盖；未提供时在解析后动态生成默认值。
parser.add_argument('--result_folder', default=None)
parser.add_argument('--viz_dir', default=None)

args = parser.parse_args()
timestamp = datetime.now().strftime('%m%d_%H%M')
profix = f'{args.test_routes}/{args.test_routes}_plan_{args.one_step_type}_iter{args.iterations}_topk{args.expansion_topk}_{timestamp}'
if not args.result_folder:
    args.result_folder = f'results/{profix}'
if not args.viz_dir:
    args.viz_dir = f'{args.result_folder}/viz'

os.makedirs(args.result_folder, exist_ok=True)
logging_path = os.path.join(args.result_folder, 'log.txt')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%m-%d %H:%M',
                    handlers=[
                        logging.FileHandler(logging_path),
                        logging.StreamHandler()
                    ])
logging.info('Parsed args save to %s' % args.result_folder)
standard_args ={
    'gpu': args.gpu,
    'seed': args.seed,
    'iterations': args.iterations,
    'expansion_topk': args.expansion_topk,
    'model type': args.one_step_type,
    'multi pool': args.multi_pool,
    'parallel num': args.parallel_num,
    'use CSS': args.CSS,
    'use DICT': args.DICT
}
for k, v in standard_args.items():
    logging.info(f'{k}: {v}')
# 把所有的参数写到yaml文件中
import yaml
with open(os.path.join(args.result_folder, 'args.yaml'), 'w') as f:
    yaml.dump(vars(args), f)
# setup device
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
