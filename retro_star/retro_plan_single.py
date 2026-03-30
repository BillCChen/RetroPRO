import numpy as np
import torch
import random
import logging
import time
import pickle
import os
# from common import args, prepare_starting_molecules, prepare_mlp,prepare_r_smiles, \
#     prepare_molstar_planner, smiles_to_fp
from common.parse_args import args
from common.prepare_utils import prepare_starting_molecules , prepare_mlp , prepare_r_smiles , prepare_molstar_planner
from common.prepare_utils import *
from common.smiles_to_fp import smiles_to_fp, batch_smiles_to_fp
from model import ValueMLP
from utils import setup_logger


def retro_plan():
    device = torch.device('cuda' if args.gpu >= 0 else 'cpu')

    starting_mols = prepare_starting_molecules(args.starting_molecules)
    if args.one_step_type == "template_based":
        one_step = prepare_mlp(args.mlp_templates, args.mlp_model_dump)
    elif args.one_step_type == "template_free": 
        import ast
        args.RD_list = ast.literal_eval(args.RD_list)
        logging.info("Parsed RD_list: %s" % args.RD_list)
        one_step = prepare_r_smiles(args.retro_model_path,args.retro_topk,args.forward_model_path,args.forward_topk,args.CSS,args.RD_list,args.DICT)
    else:
        raise ValueError("Unknown one step model type: %s" % args.one_step_type)
    # create result folder
    if not os.path.exists(args.result_folder):
        os.mkdir(args.result_folder)

    if args.use_value_fn:
        model = ValueMLP(
            n_layers=args.n_layers,
            fp_dim=args.fp_dim,
            latent_dim=args.latent_dim,
            dropout_rate=0.1,
            device=device
        ).to(device)
        model_f = '%s/%s' % (args.save_folder, args.value_model)
        logging.info('Loading value nn from %s' % model_f)
        model.load_state_dict(torch.load(model_f,  map_location=device))
        model.eval()

        def value_fn(mol):
            fp = smiles_to_fp(mol, fp_dim=args.fp_dim).reshape(1,-1)
            fp = torch.FloatTensor(fp).to(device)
            v = model(fp).item()
            return v
    else:
        value_fn = lambda x: 0.

    plan_handle = prepare_molstar_planner(
        one_step=one_step,
        value_fn=value_fn,
        starting_mols=starting_mols,
        expansion_topk=args.expansion_topk,
        iterations=args.iterations,
        viz=args.viz,
        viz_dir=args.viz_dir
    )

    result = {
        'succ': [],
        'cumulated_time': [],
        'iter': [],
        'routes': [],
        'route_costs': [],
        'route_lens': []
    }
    # haimofenjing
    # smiles_list = ["ClC1=CC=C2C(N=C(/C=C/CC3C(C)(C4=C(N3C)C=CC=C4)C)N2C5=CC=CC=C5)=C1"]
    # smiles_list = ["Cc(n1)n(c2ccccc2)c3c1cc(Cl)cc3"]

    # pth 83
    # smiles_list = ["Cc(c1)c(CCO)c[nH]c1=O"]
    # uspto-120
    # smiles_list = ["COc(c1)c(OCCNCCO)cc2c1c(Oc3c(c4nc(C)ccc4)nc(C)c(C)c3)ccn2"]
    # 2025年11月24日16:31:59 zheng rq师姐的一个分子
    # smiles_list = ["CCN(C1=CC(S(C2=CC=CC=C2)(=O)=O)=C(NC(C3=CC=C(OC)C=C3)=O)C=C1N4CC)C4=O"]
    # smiles_list = ['CCn1c(=O)n(CC)c2cc([N+](=O)[O-])c(Sc3ccccc3)cc21']
    # 2026 年 1 月 18 日 15:09:15 临时一个拆分的分子
    # smiles_list = ['Brc1ccc(C2CCNCC2)o1']
    # smiles_list = ['O=C(N1CCC(C2=CC=C(C3=CC=CN=C3N)O2)CC1)CC4=CC=CC=N4']
    # smiles_list = ['O=C(N1CCN(C2=CC=C(C3=CC=CN=C3N)O2)CC1)CC4=CC=CC=N4']
    # smiles_list = ["O=C(Cc1ccccn1)N1CCNCC1"]
    # smiles_list = ['Cn1nccc1-c1ccc2c(c1)CCN2']
    # 2026 年 1 月 19 日 13:57:02 
    # smiles_list = ['O=C(N1CCC2=CC(C3=CC(C4=CC=CN=C4N)=NN3C)=CC=C12)CC5=CC=CC=N5']
    # 2026 年 1 月 20 日 20:20:02
    # smiles_list = ['NC1=C(C(NC2=CN(C)C3=CC(COC4=NC=CC=C4)=CC=C32)=O)C=CC=N1']
    # smiles_list = ['Nc1ncccc1-n1cc2ccccc2n1'] 
    # smiles_list = ['NC1=C(N2C=C(CC3=CC=C(COC4=NC=CC=C4)C=C3)C(C)=N2)C=CC=N1']
    # smiles_list = ['NC1=C(C(NCC2=CC=C(COC3=NC=CC=C3)C=C2)=O)C=CC=N1']
    # smiles_list = ['NC(N=CC=C1)=C1C2=NC(CC(C=C3)=CC=C3COC4=CC=CC=N4)=CO2']
    # smiles_list = ['NC1=C(C(NC2=C(C3=CC=C(OC4=NC=CC=C4)C=C3)C=NN=C2)=O)C=CC=N1','NC1=C(C(NC2=CC3=C(SC(OC4=NC=CC=C4)=C3)C=C2)=O)C=CC=N1','NC1=NC=CC=C1N2N=CC(C3=CC=C(N=C(N4)OC5=CC=CC=N5)C4=C3)=C2']
    # smiles_list = ['NC1=NC=CC=C1N2N=CC(C3=CC=C(N=C(N4C)OC5=CC=CC=N5)C4=C3)=C2']
    # smiles_list = ['NC1=NC=CC=C1C2=CSC(C(C=C3)=CC=C3OCC4=CC=CC=N4)=N2']
    # smiles_list = ['O=C(C1=C(C2C3C(C(O)=O)=CN2)C=CC=C1)C3=O']
    # 2026 年 1 月 27 日 21:15:21
    smiles_list = ['NC1=NC=CC=C1N2N=CC(C3=CC=C(N=C(N4C)OC5=CC=CC=N5)C4=C3)=C2']
    num_targets = len(smiles_list)
    t0 = time.time()
    for (i, target_mol) in enumerate(smiles_list):

        try:
            succ, msg = plan_handle(target_mol, i)
        except Exception as e:
            logging.info(f"Error planning for target {i}: {e}")
            succ = False
            msg = (None, 101)

        result['succ'].append(succ)
        result['cumulated_time'].append(time.time() - t0)
        result['iter'].append(msg[1])
        result['routes'].append(msg[0])
        if succ:
            result['route_costs'].append(msg[0].total_cost)
            result['route_lens'].append(msg[0].length)
        else:
            result['route_costs'].append(None)
            result['route_lens'].append(None)

        tot_num = i + 1
        tot_succ = np.array(result['succ']).sum()
        avg_time = (time.time() - t0) * 1.0 / tot_num
        avg_iter = np.array(result['iter'], dtype=float).mean()
        logging.info('Succ: %d/%d/%d | avg time: %.2f s | avg iter: %.2f' %
                     (tot_succ, tot_num, num_targets, avg_time, avg_iter))

        f = open(args.result_folder + '/plan.pkl', 'wb')
        pickle.dump(result, f)
        f.close()
        logging.info('Results saved to \n  ====> %s/plan.pkl' % args.result_folder)

if __name__ == '__main__':
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    # setup_logger('plan.log')

    retro_plan()
# python retro_plan_single.py --use_value_fn --one_step_type template_free 
# --iterations 101 --viz --gpu 0 
# --expansion_topk 16 
# --CSS --RD_list "[(7,2),(3,0)]" --DICT 
# --test_routes self_defined 