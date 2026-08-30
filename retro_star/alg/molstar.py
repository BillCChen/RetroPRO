import os
import numpy as np
import logging
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from alg.mol_tree import MolTree
import time 
import rdkit
from rdkit import Chem
def molstar(target_mol, target_mol_id, starting_mols, expand_fn, value_fn,
            iterations, viz=False, viz_dir=None, progress_callback=None,
            expansion_collector=None):
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
                candidate_records = []
                ancestors = m_next.get_ancestors()
                for j in range(len(scores)):
                    reactant_list = list(set(reactants[j].split('.')))
                    # 检查每个反应物是否都有效
                    valid = True
                    invalid_reactants = []
                    for r in reactant_list:
                        if Chem.MolFromSmiles(r) is None:
                            valid = False
                            invalid_reactants.append(r)
                            logging.info('Invalid reactant %s from expansion of %s' % (r, m_next.mol))
                            break
                    has_ancestor_reactant = any(r in ancestors for r in reactant_list)
                    if expansion_collector is not None:
                        candidate_records.append({
                            'target_id': int(target_mol_id),
                            'target_mol': target_mol,
                            'iteration': int(i + 1),
                            'expanded_mol': m_next.mol,
                            'expanded_mol_id': int(m_next.id),
                            'expanded_mol_depth': int(m_next.depth),
                            'reaction_depth': int(m_next.depth + 1),
                            'candidate_rank': int(j + 1),
                            'reactants_raw': reactants[j],
                            'reactants': reactant_list,
                            'score': float(scores[j]),
                            'cost': float(cost[j]),
                            'template': templates[j],
                            'valid': bool(valid),
                            'invalid_reactants': invalid_reactants,
                            'has_ancestor_reactant': bool(has_ancestor_reactant),
                            'expected_added_to_tree': bool(valid and not has_ancestor_reactant),
                        })
                    if valid:
                        reactant_lists.append(reactant_list)
                        templates_list.append(templates[j])
                        costs_list.append(cost[j])

                if expansion_collector is not None:
                    for record in candidate_records:
                        expansion_collector.record_reaction_candidate(record)
                    expansion_collector.record_node_expansion(
                        target_id=target_mol_id,
                        target_mol=target_mol,
                        iteration=i + 1,
                        expanded_mol_node=m_next,
                        search_status=mol_tree.search_status,
                        root_succ_value=mol_tree.root.succ_value,
                        num_model_candidates=len(scores),
                        num_valid_candidates=len(costs_list),
                        num_expected_added_candidates=sum(
                            1 for record in candidate_records
                            if record['expected_added_to_tree']
                        ),
                        failure_reason=None if len(costs_list) > 0 else 'no_valid_candidates',
                    )

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
                if expansion_collector is not None:
                    failure_reason = 'no_model_output'
                    if result is not None and ('scores' not in result or len(result.get('scores', [])) == 0):
                        failure_reason = 'no_scores'
                    expansion_collector.record_node_expansion(
                        target_id=target_mol_id,
                        target_mol=target_mol,
                        iteration=i + 1,
                        expanded_mol_node=m_next,
                        search_status=mol_tree.search_status,
                        root_succ_value=mol_tree.root.succ_value,
                        num_model_candidates=0,
                        num_valid_candidates=0,
                        num_expected_added_candidates=0,
                        failure_reason=failure_reason,
                    )
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
        try:
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
        except Exception as exc:
            logging.info(
                'Visualization failed for target_mol_id=%s (install system graphviz so `dot` is on PATH; '
                'planning result is unchanged): %s',
                target_mol_id,
                exc,
            )
    end_total_nodes = len(mol_tree.mol_nodes)
    emit_progress(max(i + 1, 0), iteration_elapsed=None, status='completed' if mol_tree.succ else 'finished')
    print(f"Total searched nodes: |-{end_total_nodes}-|")
    return mol_tree.succ, (best_route, i+1, end_total_nodes)
