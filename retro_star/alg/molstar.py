import os
import numpy as np
import logging
import sys
sys.path.append("/home/chenqixuan/retro_star/retro_star/alg")
from alg.mol_tree import MolTree
import time 
import rdkit
from rdkit import Chem
def molstar(target_mol, target_mol_id, starting_mols, expand_fn, value_fn,
            iterations, viz=False, viz_dir=None, progress_callback=None):
    mol_tree = MolTree(
        target_mol=target_mol,
        known_mols=starting_mols,
        value_fn=value_fn
    )

    def emit_progress(iteration_idx, iteration_elapsed=None, status='running'):
        if progress_callback is None:
            return
        mol_nodes = getattr(mol_tree, 'mol_nodes', [])
        reaction_nodes = getattr(mol_tree, 'reaction_nodes', [])
        expanded_nodes = max(len(mol_nodes) + len(reaction_nodes) - 1, 0)
        max_depth = max((node.depth for node in mol_nodes), default=0)
        try:
            progress_callback(
                {
                    'status': status,
                    'current_iteration': int(iteration_idx),
                    'total_iterations': int(iterations),
                    'expanded_nodes': int(expanded_nodes),
                    'max_depth': int(max_depth),
                    'iteration_elapsed_seconds': iteration_elapsed,
                }
            )
        except Exception as exc:
            logging.info('Progress callback failed: %s', exc)

    i = -1
    route_order = 1
    emit_progress(0, iteration_elapsed=0.0, status='running')
    if not mol_tree.succ:
        for i in range(iterations):
            begin = time.time()
            if i % 20 == 0:
                logging.info('Search nodes num: |%d| in iter |%d|' % (len(mol_tree.mol_nodes),i+1))
            scores = []
            for m in mol_tree.mol_nodes:
                if m.open:
                    scores.append(m.v_target())
                else:
                    scores.append(np.inf)
            scores = np.array(scores)
            if np.min(scores) == np.inf:
                logging.info('No open nodes!')
                break
            metric = scores
            mol_tree.search_status = np.min(metric)
            m_next = mol_tree.mol_nodes[np.argmin(metric)]

            assert m_next.open

            result = expand_fn(m_next.mol)
            # logging.info("done")
            if result is not None and (len(result['scores']) > 0):
                reactants = result['reactants']
                scores = result['scores']
                costs = 0.0 - np.log(np.clip(np.array(scores), 1e-3, 1.0))
                # costs = 1.0 - np.array(scores)
                if 'templates' in result.keys():
                    templates = result['templates']
                else:
                    templates = result['template']
                cost = list(costs)
                costs_list = []
                reactant_lists = []
                templates_list = []
                for j in range(len(scores)):
                    reactant_list = list(set(reactants[j].split('.')))
                    # 检查每个反应物是否都有效
                    valid = True
                    for r in reactant_list:
                        if Chem.MolFromSmiles(r) is None:
                            valid = False
                            logging.info('Invalid reactant %s from expansion of %s' % (r, m_next.mol))
                            break
                    if valid:
                        reactant_lists.append(reactant_list)
                        templates_list.append(templates[j])
                        costs_list.append(cost[j])


                assert m_next.open
                succ = mol_tree.expand(m_next, reactant_lists, costs_list, templates_list)

                if succ:
                    break
                    # best_route = mol_tree.get_best_route()
                    # f = '%s/mol_%d_route_%d' % (viz_dir, target_mol_id, route_order)
                    # best_route.viz_route(f)
                    # logging.info('=================================>Found route %d with cost %.2f in %d iterations' % (route_order, best_route.total_cost, i+1))
                    # route_order += 1
                # found optimal route
                if mol_tree.root.succ_value <= mol_tree.search_status:
                    break

            else:
                mol_tree.expand(m_next, None, None, None)
                logging.info('Expansion fails on %s!' % m_next.mol)
            end = time.time()
            logging.info('%s : %.1f s' % (m_next.mol,end - begin))
            emit_progress(i + 1, iteration_elapsed=end - begin, status='running')
        logging.info('Final search status | success value | iter: %s | %s | %d'
                     % (str(mol_tree.search_status), str(mol_tree.root.succ_value), i+1))

    best_route = None
    if mol_tree.succ:
        best_route = mol_tree.get_best_route()
        assert best_route is not None

    if viz:
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)

        if mol_tree.succ:
            if best_route.optimal:
                f = '%s/mol_%d_route_optimal' % (viz_dir, target_mol_id)
            else:
                f = '%s/mol_%d_route_single' % (viz_dir, target_mol_id)
            best_route.viz_route(f)

        f = '%s/mol_%d_search_tree' % (viz_dir, target_mol_id)
        mol_tree.viz_search_tree(f)
    end_total_nodes = len(mol_tree.mol_nodes)
    emit_progress(max(i + 1, 0), iteration_elapsed=None, status='completed' if mol_tree.succ else 'finished')
    print(f"Total searched nodes: |-{end_total_nodes}-|")
    return mol_tree.succ, (best_route, i+1, end_total_nodes)
