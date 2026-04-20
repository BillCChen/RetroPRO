import pickle
import pandas as pd
import logging
import sys
sys.path.append("/home/chenqixuan/retro_star/retro_star/packages/mlp_retrosyn")
from mlp_retrosyn.mlp_inference import MLPModel

sys.path.append("/home/chenqixuan/retro_star/retro_star/alg")
from alg import molstar

def prepare_starting_molecules(filename):
    logging.info('Loading starting molecules from %s' % filename)

    if filename[-3:] == 'csv':
        starting_mols = set(list(pd.read_csv(filename)['mol']))
        # exclude_smiles = set(['Brc1ccc(C2CCNCC2)o1'])
        exclude_smiles = set()
        starting_mols = starting_mols - exclude_smiles
        logging.info(f'''
                     !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                     !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                     !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                     Warning: excluding starting mols for testing
                     {exclude_smiles}
                     !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                     !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                     !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                     ''')
    else:
        assert filename[-3:] == 'pkl'
        with open(filename, 'rb') as f:
            starting_mols = pickle.load(f)

    logging.info('%d starting molecules loaded' % len(starting_mols))
    return starting_mols

def prepare_mlp(templates, model_dump, device=-1):
    logging.info('Templates: %s' % templates)
    logging.info('Loading trained mlp model from %s' % model_dump)
    one_step = MLPModel(model_dump, templates, device=device)
    return one_step
def prepare_r_smiles(retro_model_path,retro_topk,forward_model_path,forward_topk,CSS,RD_list,DICT, device=-1):
    from mlp_retrosyn.tp_free_inference import TP_free_Model
    logging.info('Loading trained R-SMILES model from %s' % retro_model_path)
    logging.info('Loading trained Forward model from %s' % forward_model_path)
    logging.info('CSS: %s' % CSS)
    logging.info('DICT: %s' % DICT)
    one_step = TP_free_Model(retro_model_path,retro_topk,forward_model_path,forward_topk,CSS,RD_list,DICT, device=device)
    return one_step
def prepare_molstar_planner(one_step, value_fn, starting_mols, expansion_topk,
                            iterations, viz=False, viz_dir=None, progress_callback=None):
    expansion_handle = lambda x: one_step.run(x, topk=expansion_topk)

    plan_handle = lambda x, y=0: molstar(
        target_mol=x,
        target_mol_id=y,
        starting_mols=starting_mols,
        expand_fn=expansion_handle,
        value_fn=value_fn,
        iterations=iterations,
        viz=viz,
        viz_dir=viz_dir,
        progress_callback=progress_callback,
    )
    return plan_handle
