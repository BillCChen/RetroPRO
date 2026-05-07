import logging
import os
import numpy as np
from rdkit import Chem

import sys
sys.path.append("/home/chenqixuan/retro_star/retro_star/alg")
from alg.mol_tree import MolTree


class MolStarTask:
    """Step-wise retro planning task for one target molecule."""

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
        self.target_mol = target_mol
        self.target_mol_id = target_mol_id
        self.iterations = iterations
        self.viz = viz
        self.viz_dir = viz_dir

        self.mol_tree = MolTree(
            target_mol=target_mol,
            known_mols=starting_mols,
            value_fn=value_fn,
        )

        self.steps_taken = 0
        self.current_node = None
        self.done = self.mol_tree.succ
        self.error = None

    def _select_next_open_node(self):
        scores = []
        for mol_node in self.mol_tree.mol_nodes:
            if mol_node.open:
                scores.append(mol_node.v_target())
            else:
                scores.append(np.inf)
        scores = np.array(scores)
        if np.min(scores) == np.inf:
            return None, None
        search_status = float(np.min(scores))
        selected = self.mol_tree.mol_nodes[int(np.argmin(scores))]
        return selected, search_status

    def ready_for_expansion(self):
        return (not self.done) and (self.current_node is None)

    def next_frontier_smiles(self):
        if self.done:
            return None
        if self.current_node is not None:
            return self.current_node.mol
        if self.steps_taken >= self.iterations:
            self.done = True
            return None

        selected, search_status = self._select_next_open_node()
        if selected is None:
            logging.info("No open nodes for target_id=%d", self.target_mol_id)
            # Keep iteration accounting consistent with legacy molstar loop,
            # where the "no open nodes" check happens inside an iteration.
            self.steps_taken += 1
            self.done = True
            return None

        self.mol_tree.search_status = search_status
        self.current_node = selected
        self.steps_taken += 1
        return selected.mol

    @staticmethod
    def _normalize_expansion_result(result, frontier_mol):
        if result is None:
            return None, None, None
        if 'scores' not in result or len(result['scores']) == 0:
            return None, None, None

        reactants = result.get('reactants', [])
        scores = result.get('scores', [])
        templates = result.get('templates', result.get('template', []))

        costs = 0.0 - np.log(np.clip(np.array(scores), 1e-3, 1.0))
        costs = list(costs)

        reactant_lists = []
        costs_list = []
        templates_list = []

        for i in range(len(scores)):
            reactant_list = list(set(reactants[i].split('.')))
            valid = True
            for reactant in reactant_list:
                if Chem.MolFromSmiles(reactant) is None:
                    valid = False
                    logging.info(
                        'Invalid reactant %s from expansion of %s',
                        reactant,
                        frontier_mol,
                    )
                    break
            if valid:
                reactant_lists.append(reactant_list)
                templates_list.append(templates[i])
                costs_list.append(costs[i])

        if len(costs_list) == 0:
            return None, None, None

        return reactant_lists, costs_list, templates_list

    def apply_expansion(self, result):
        if self.done:
            return
        if self.current_node is None:
            return

        frontier = self.current_node
        self.current_node = None

        try:
            reactant_lists, costs_list, templates_list = self._normalize_expansion_result(
                result,
                frontier.mol,
            )
            if costs_list is None:
                self.mol_tree.expand(frontier, None, None, None)
                logging.info('Expansion fails on %s!', frontier.mol)
            else:
                succ = self.mol_tree.expand(frontier, reactant_lists, costs_list, templates_list)
                if succ:
                    self.done = True
                    return
                if self.mol_tree.root.succ_value <= self.mol_tree.search_status:
                    self.done = True
                    return

            if self.steps_taken >= self.iterations:
                self.done = True
        except Exception as exc:
            logging.info(
                'Error applying expansion for target_id=%d: %s',
                self.target_mol_id,
                exc,
            )
            self.error = exc
            self.done = True

    def is_done(self):
        return self.done

    def finalize_message(self):
        best_route = None
        if self.mol_tree.succ:
            best_route = self.mol_tree.get_best_route()

        if self.viz:
            try:
                if not os.path.exists(self.viz_dir):
                    os.makedirs(self.viz_dir)

                if self.mol_tree.succ and best_route is not None:
                    if best_route.optimal:
                        route_file = '%s/mol_%d_route_optimal' % (self.viz_dir, self.target_mol_id)
                    else:
                        route_file = '%s/mol_%d_route_single' % (self.viz_dir, self.target_mol_id)
                    best_route.viz_route(route_file)

                tree_file = '%s/mol_%d_search_tree' % (self.viz_dir, self.target_mol_id)
                self.mol_tree.viz_search_tree(tree_file)
            except Exception as exc:
                logging.info(
                    'Visualization failed for target_id=%d (install system graphviz so `dot` is on PATH; '
                    'planning result is unchanged): %s',
                    self.target_mol_id,
                    exc,
                )

        end_total_nodes = len(self.mol_tree.mol_nodes)
        return self.mol_tree.succ, (best_route, self.steps_taken, end_total_nodes)
