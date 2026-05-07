from __future__ import print_function
import os
import threading
import numpy as np
import torch
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
from rdkit import Chem
from rdkit.Chem import AllChem
import rdchiral
from rdchiral.main import rdchiralRunText, rdchiralRun
from rdchiral.initialization import rdchiralReaction, rdchiralReactants
from .mlp_policies import load_parallel_model, preprocess
from collections import defaultdict, OrderedDict
import logging


def merge(reactant_d):
    ret = []
    for reactant, l in reactant_d.items():
        ss, ts = zip(*l)
        ret.append((reactant, sum(ss), list(ts)[0]))
    reactants, scores, templates = zip(*sorted(ret, key=lambda item: item[1], reverse=True))
    return list(reactants), list(scores), list(templates)


import time
day_hour_min_sec = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
import pickle
from .tp_free_tools import random_substructure, rand_aug_smiles, repeat_retro_k
from .tp_free_tools import Load_Retro_Model, Load_Forward_Model
from rdchiral import template_extractor as extractor

try:
    from rxnmapper import BatchedMapper
except Exception:
    BatchedMapper = None


class TP_free_Model(object):
    def __init__(
        self,
        retro_model_path,
        retro_topk,
        forward_model_path,
        forwad_topk=1,
        CCS=True,
        RD_list=[(9, 0)],
        DICT=True,
        device=-1,
        shared_dict=None,
        dict_lock=None,
    ):
        super(TP_free_Model, self).__init__()
        self.device = device
        self.use_CCS = CCS
        self.RD_list = RD_list
        self.use_DICT = DICT
        self.retro_topk = int(retro_topk)
        self.forward_topk = int(forwad_topk)
        self.retro_batch_size = int(os.getenv("TP_FREE_RETRO_BATCH_SIZE", "512"))
        self.forward_batch_size = int(os.getenv("TP_FREE_FORWARD_BATCH_SIZE", "512"))
        self.mapper_batch_size = int(os.getenv("TP_FREE_MAPPER_BATCH_SIZE", "256"))
        self.dict_dump_every = int(os.getenv("TP_FREE_DICT_DUMP_EVERY", "0"))
        self.dict_dump_dir = os.getenv("TP_FREE_DICT_DUMP_DIR", "")
        self.dict_dump_with_meta = os.getenv("TP_FREE_DICT_DUMP_WITH_META", "0") == "1"
        self._dict_stats_topk = int(os.getenv("TP_FREE_DICT_STATS_TOPK", "500"))
        self._dict_update_count = 0

        # DICT cache statistics (JSON-serializable report via get_dict_cache_report).
        self._stats_lock = threading.Lock()
        self._per_target_stats = defaultdict(
            lambda: {
                "substructure_lookups": 0,
                "substructure_hits": 0,
                "new_keys": 0,
                "new_template_values": 0,
            }
        )
        self._global_substructure_hit_counts = defaultdict(int)
        self._global_template_hit_counts = defaultdict(int)
        self._global_total_substructure_lookups = 0
        self._global_total_substructure_hits = 0
        self._key_first_infer = {}
        self._pair_first_infer = {}
        self._pair_key_sep = "|||"

        # DICT storage: use external shared proxy when running multi-GPU workers,
        # otherwise use a local defaultdict.
        if shared_dict is not None:
            self._dict_ref = shared_dict
            self._dict_lock = dict_lock
            self._dict_is_shared = True
        else:
            self._dict_ref = defaultdict(list)
            self._dict_lock = None
            self._dict_is_shared = False

        # Keep self.DICT as an alias for backward-compat with code that reads it directly.
        self.DICT = self._dict_ref

        gpu_device = int(device) if isinstance(device, int) and device >= 0 else -1
        self.retro_model = Load_Retro_Model(
            retro_model_path,
            beam_size=10,
            n_best=self.retro_topk,
            batch_size=self.retro_batch_size,
            gpu_device=gpu_device,
        )
        logging.info("Loaded retro model with batch_size=%d, gpu=%d", self.retro_batch_size, gpu_device)
        self.forward_model = Load_Forward_Model(
            forward_model_path,
            beam_size=10,
            n_best=self.forward_topk,
            batch_size=self.forward_batch_size,
            gpu_device=gpu_device,
        )
        logging.info("Loaded forward model with batch_size=%d, gpu=%d", self.forward_batch_size, gpu_device)
        if BatchedMapper is None:
            raise ImportError("rxnmapper is required for template_free inference. Please install `rxnmapper`.")
        self.mapper = BatchedMapper(batch_size=self.mapper_batch_size)
        logging.info("Loaded mapper with batch_size=%d", self.mapper_batch_size)

    def random_sampling(self, x, RD_list, topk):
        output = set()
        rd_size = max(len(RD_list), 1)
        each_num = max(1, int(topk) // rd_size)
        for R, D in RD_list:
            if self.use_CCS:
                sub_smiles = random_substructure(x, r=R, d=D, num=each_num)
            else:
                sub_smiles = [x for _ in range(each_num)]
            for sub_smi in sub_smiles:
                mol = Chem.MolFromSmiles(sub_smi)
                if mol is not None:
                    rand_smi = Chem.MolToSmiles(mol, doRandom=True)
                    output.add(rand_smi)
        if len(output) < 1 and self.use_CCS:
            logging.info(f"Random substructure extraction failed for {x} with RD_list {RD_list}. Using original molecule.")
            output.add(x)
        return list(output)

    def check_smiles_valid(self, smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol
        except Exception:
            return None

    def invalid_retro_filter(self, smiles, retro):
        valid_smiles = []
        valid_retro = []
        for smi, r_smi in zip(smiles, retro):
            if '.' not in r_smi:
                r_mol = self.check_smiles_valid(r_smi)
                if r_mol is not None:
                    valid_smiles.append(smi)
                    valid_retro.append(r_smi)
            else:
                parts = r_smi.split('.')
                all_valid = True
                for part in parts:
                    r_mol = self.check_smiles_valid(part)
                    if r_mol is None:
                        all_valid = False
                        break
                if all_valid:
                    valid_smiles.append(smi)
                    valid_retro.append(r_smi)
        return valid_smiles, valid_retro

    def mol2cano_smiles(self, mol):
        return Chem.MolToSmiles(mol, isomericSmiles=False)

    def smi2cano_smiels(self, smiles):
        if '.' not in smiles:
            mol = Chem.MolFromSmiles(smiles)
            return self.mol2cano_smiles(mol)
        else:
            parts = smiles.split('.')
            cano_parts = []
            for part in parts:
                mol = Chem.MolFromSmiles(part)
                cano_parts.append(self.mol2cano_smiles(mol))
            cano_parts = sorted(cano_parts)
            return '.'.join(cano_parts)

    def filter(self, x, smiles, retro, forward):
        cano_smi_reactions = []
        for CCS_smi, r_smi, f_smi in zip(smiles, retro, forward):
            f_mol = self.check_smiles_valid(f_smi)
            if f_mol is None:
                continue
            smi_mol = Chem.MolFromSmiles(CCS_smi)
            if self.mol2cano_smiles(smi_mol) == self.mol2cano_smiles(f_mol):
                cano_smi_reactions.append(
                    (self.mol2cano_smiles(smi_mol), f"{CCS_smi}>>{self.smi2cano_smiels(r_smi)}")
                )
        logging.info(f"Filtered {len(cano_smi_reactions)} valid reactions from {len(smiles)} candidates.")
        return cano_smi_reactions

    def _extract_templates_from_mapped(self, ccs_smiles, mapped_reactions):
        """Extract templates from pre-mapped reactions (no mapper call here)."""
        templates = []
        for cano_smi, mapped_rxn in zip(ccs_smiles, mapped_reactions):
            if mapped_rxn is None:
                continue
            mapped_rxn_dict = {
                'reactants': mapped_rxn.split('>>')[1],
                'products': mapped_rxn.split('>>')[0],
                '_id': 'test',
            }
            template = extractor.extract_from_reaction(mapped_rxn_dict)
            if 'reaction_smarts' not in template:
                continue
            templates.append((cano_smi, template['reaction_smarts']))
        return templates

    def _pair_key_str(self, cano_smi, rule):
        return "%s%s%s" % (cano_smi, self._pair_key_sep, rule)

    def _record_renew_increment(self, key, rule, is_new_key, task_id, target_smiles):
        """Record first-infer attribution and per-target delta counts (under _stats_lock)."""
        src = None
        if task_id is not None:
            src = {"task_id": int(task_id), "target_smiles": target_smiles}
        if src is not None:
            if is_new_key and key not in self._key_first_infer:
                self._key_first_infer[key] = dict(src)
            pk = self._pair_key_str(key, rule)
            if pk not in self._pair_first_infer:
                self._pair_first_infer[pk] = dict(src)
        if task_id is not None:
            pt = self._per_target_stats[task_id]
            if is_new_key:
                pt["new_keys"] += 1
            pt["new_template_values"] += 1

    def renew_DICT(self, templates, task_id=None, target_smiles=None):
        updated = 0
        if self._dict_is_shared:
            for key, rule in templates:
                with self._dict_lock:
                    existing = list(self._dict_ref.get(key, []))
                    is_new_key = len(existing) == 0
                    if rule not in existing:
                        self._dict_ref[key] = existing + [rule]
                        updated += 1
                        with self._stats_lock:
                            self._record_renew_increment(
                                key, rule, is_new_key, task_id, target_smiles
                            )
        else:
            for key, rule in templates:
                existing = self._dict_ref[key]
                is_new_key = len(existing) == 0
                if rule not in existing:
                    self._dict_ref[key].append(rule)
                    updated += 1
                    with self._stats_lock:
                        self._record_renew_increment(
                            key, rule, is_new_key, task_id, target_smiles
                        )

        self._dict_update_count += updated
        if (
            self.dict_dump_every > 0
            and self._dict_update_count >= self.dict_dump_every
            and self.dict_dump_dir
            and not self._dict_is_shared
        ):
            os.makedirs(self.dict_dump_dir, exist_ok=True)
            save_dir = os.path.join(self.dict_dump_dir, f"tp_free_DICT_{day_hour_min_sec}.pkl")
            payload = dict(self._dict_ref)
            if self.dict_dump_with_meta:
                payload = {"rules": payload, "stats": self.get_dict_cache_report()}
            pickle.dump(payload, open(save_dir, 'wb'))
            self._dict_update_count = 0

    def save_dict_snapshot(self, filepath, with_meta=None):
        """Write current DICT rules to a pickle file (local dict only; not for shared Manager dict)."""
        if self._dict_is_shared:
            return False
        if with_meta is None:
            with_meta = self.dict_dump_with_meta
        parent = os.path.dirname(os.path.abspath(filepath))
        if parent:
            os.makedirs(parent, exist_ok=True)
        payload = dict(self._dict_ref)
        if with_meta:
            payload = {'rules': payload, 'stats': self.get_dict_cache_report()}
        with open(filepath, 'wb') as f:
            pickle.dump(payload, f)
        return True

    def _canonicalize_target(self, x):
        try:
            mol = Chem.MolFromSmiles(x)
            if mol is None:
                return x
            return Chem.MolToSmiles(mol, isomericSmiles=False)
        except Exception:
            return x

    def _is_valid_retro(self, retro_smi):
        if not retro_smi:
            return False
        if '.' not in retro_smi:
            return self.check_smiles_valid(retro_smi) is not None
        parts = retro_smi.split('.')
        for part in parts:
            if self.check_smiles_valid(part) is None:
                return False
        return True

    def _record_cache_lookup(self, cano_smi, cached_rules, task_id):
        with self._stats_lock:
            self._global_total_substructure_lookups += 1
            if task_id is not None:
                self._per_target_stats[task_id]["substructure_lookups"] += 1
            if len(cached_rules) > 0:
                self._global_total_substructure_hits += 1
                self._global_substructure_hit_counts[cano_smi] += 1
                if task_id is not None:
                    self._per_target_stats[task_id]["substructure_hits"] += 1
                for r in cached_rules:
                    self._global_template_hit_counts[r] += 1

    def _prepare_single_target(self, x, topk, task_id=None):
        target = self._canonicalize_target(x)
        sampled = self.random_sampling(target, self.RD_list, topk)

        aug_smiles = []
        dict_rules = []
        if self.use_DICT:
            for smi in sampled:
                cano_smi = self.smi2cano_smiels(smi)
                cached_rules = self._dict_ref.get(cano_smi, [])
                self._record_cache_lookup(cano_smi, cached_rules, task_id)
                if len(cached_rules) > 0:
                    dict_rules.extend(cached_rules)
                    aug_smiles.extend(rand_aug_smiles(smi, 1))
                else:
                    aug_smiles.extend(rand_aug_smiles(smi, 5))
        else:
            for smi in sampled:
                aug_smiles.extend(rand_aug_smiles(smi, 5))

        if len(aug_smiles) == 0:
            aug_smiles = rand_aug_smiles(target, max(int(topk), 1))
        if len(aug_smiles) == 0:
            aug_smiles = [target]
        return {
            'target': target,
            'aug_smiles': aug_smiles,
            'dict_rules': dict_rules,
        }

    def _align_forward_outputs(self, forward_raw, expected_size):
        if expected_size == 0:
            return []
        if len(forward_raw) == expected_size:
            return list(forward_raw)
        if self.forward_topk > 1 and len(forward_raw) >= expected_size * self.forward_topk:
            outputs = []
            for idx in range(expected_size):
                outputs.append(forward_raw[idx * self.forward_topk])
            return outputs

        logging.info(
            "Forward output size mismatch: expected=%d got=%d; truncating/padding.",
            expected_size,
            len(forward_raw),
        )
        outputs = list(forward_raw[:expected_size])
        while len(outputs) < expected_size:
            outputs.append("")
        return outputs

    def _rules_to_result(self, x, rule_list):
        reactants = []
        scores = []
        templates = []
        for rule in rule_list:
            out1 = []
            try:
                all_out = AllChem.ReactionFromSmarts(rule).RunReactants((Chem.MolFromSmiles(x),))
                if len(all_out) == 0:
                    continue
                out1 = [Chem.MolToSmiles(mol) for mol in all_out[0]]
                for smi in out1:
                    if Chem.MolFromSmiles(smi) is None:
                        out1 = []
                        break
                if len(out1) == 0:
                    continue
                out1 = ['.'.join(sorted(out1))]
                for reactant in out1:
                    reactants.append(reactant)
                    scores.append(1.0)
                    templates.append(rule)
            except ValueError:
                pass

        if len(reactants) == 0:
            return None

        reactants_d = defaultdict(list)
        for r, s, t in zip(reactants, scores, templates):
            if '.' in r:
                str_list = sorted(r.strip().split('.'))
                reactants_d['.'.join(str_list)].append((s, t))
            else:
                reactants_d[r].append((s, t))

        reactants, scores, templates = merge(reactants_d)
        total = sum(scores)
        if total <= 0:
            return None
        scores = [s / total for s in scores]
        return {
            'reactants': reactants,
            'scores': scores,
            'template': templates,
        }

    def _dict_num_keys(self):
        return len(self._dict_ref)

    def _dict_num_values(self):
        n = 0
        for _k, rules in self._dict_ref.items():
            n += len(rules)
        return n

    def _topk_int_dict(self, counter_dict, k):
        items = sorted(counter_dict.items(), key=lambda kv: (-kv[1], kv[0]))
        if k > 0:
            items = items[:k]
        return {str(a): int(b) for a, b in items}

    def get_dict_cache_report(self):
        """Return a JSON-serializable summary of DICT cache statistics."""
        with self._stats_lock:
            per_target = []
            for tid in sorted(self._per_target_stats.keys()):
                row = dict(self._per_target_stats[tid])
                lu = row["substructure_lookups"]
                row["task_id"] = int(tid)
                row["substructure_hit_rate"] = (
                    float(row["substructure_hits"]) / float(lu) if lu else None
                )
                per_target.append(row)
            sub_hits = dict(self._global_substructure_hit_counts)
            tpl_hits = dict(self._global_template_hit_counts)
            g_lookups = int(self._global_total_substructure_lookups)
            g_hit = int(self._global_total_substructure_hits)
            key_first = {str(k): dict(v) for k, v in self._key_first_infer.items()}
            pair_first = {str(k): dict(v) for k, v in self._pair_first_infer.items()}
            topk = self._dict_stats_topk

        agg = "full"
        if self._dict_is_shared:
            agg = "per_worker_partial"

        return {
            "aggregation_mode": agg,
            "global": {
                "dict_num_keys": int(self._dict_num_keys()),
                "dict_num_values": int(self._dict_num_values()),
                "substructure_lookups_total": g_lookups,
                "substructure_hits_total": g_hit,
                "substructure_hit_rate": (float(g_hit) / float(g_lookups) if g_lookups else None),
                "substructure_hit_counts_topk": self._topk_int_dict(sub_hits, topk),
                "template_hit_counts_topk": self._topk_int_dict(tpl_hits, topk),
            },
            "per_target": per_target,
            "first_infer": {
                "key": key_first,
                "key_rule": pair_first,
            },
        }

    def run_batch(self, x_list, topk=20, task_ids=None):
        if x_list is None or len(x_list) == 0:
            return []

        if task_ids is not None and len(task_ids) != len(x_list):
            raise ValueError("task_ids must be the same length as x_list")

        # --- Preparation (CPU-bound RDKit) -- parallelized across molecules ---
        max_workers = min(len(x_list), 8)

        def _prep_one(i):
            tid = task_ids[i] if task_ids is not None else None
            return self._prepare_single_target(x_list[i], topk, task_id=tid)

        if max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                prepared = list(ex.map(_prep_one, range(len(x_list))))
        else:
            prepared = [_prep_one(0)]

        # --- Retro inference (single batched GPU call) ---
        flat_aug_smiles = []
        flat_owner = []
        for owner_idx, item in enumerate(prepared):
            for smi in item['aug_smiles']:
                flat_aug_smiles.append(smi)
                flat_owner.append(owner_idx)

        owner_buckets = defaultdict(lambda: {'smiles': [], 'retro': [], 'forward': []})

        if len(flat_aug_smiles) > 0:
            try:
                retro = self.retro_model.inference(flat_aug_smiles)
            except Exception as exc:
                logging.info("Retro model inference error for batch size %d: %s", len(flat_aug_smiles), exc)
                retro = []

            expanded_aug = repeat_retro_k(flat_aug_smiles, self.retro_topk)
            expanded_owner = repeat_retro_k(flat_owner, self.retro_topk)
            if len(retro) != len(expanded_aug):
                valid_len = min(len(retro), len(expanded_aug))
                logging.info(
                    "Retro output size mismatch: expected=%d got=%d; truncating to %d",
                    len(expanded_aug), len(retro), valid_len,
                )
                retro = list(retro[:valid_len])
                expanded_aug = list(expanded_aug[:valid_len])
                expanded_owner = list(expanded_owner[:valid_len])

            valid_owner = []
            valid_smiles = []
            valid_retro = []
            for owner_idx, ccs_smi, retro_smi in zip(expanded_owner, expanded_aug, retro):
                if self._is_valid_retro(retro_smi):
                    valid_owner.append(owner_idx)
                    valid_smiles.append(ccs_smi)
                    valid_retro.append(retro_smi)

            # --- Forward inference (single batched GPU call) ---
            if len(valid_retro) > 0:
                try:
                    forward_raw = self.forward_model.inference(valid_retro)
                except Exception as exc:
                    logging.info("Forward model inference error for batch size %d: %s", len(valid_retro), exc)
                    forward_raw = []
            else:
                forward_raw = []
            valid_forward = self._align_forward_outputs(forward_raw, len(valid_retro))

            for owner_idx, ccs_smi, retro_smi, forward_smi in zip(
                valid_owner, valid_smiles, valid_retro, valid_forward
            ):
                owner_buckets[owner_idx]['smiles'].append(ccs_smi)
                owner_buckets[owner_idx]['retro'].append(retro_smi)
                owner_buckets[owner_idx]['forward'].append(forward_smi)

        # --- Phase A: CPU filter per molecule, collect reactions ---
        per_mol_reactions = []
        for owner_idx, item in enumerate(prepared):
            target = item['target']
            bucket = owner_buckets.get(owner_idx, {'smiles': [], 'retro': [], 'forward': []})
            reactions = self.filter(target, bucket['smiles'], bucket['retro'], bucket['forward'])
            per_mol_reactions.append(reactions)

        # --- Phase B: Single mapper call across ALL molecules ---
        flat_rxns = [rxn for reactions in per_mol_reactions for _, rxn in reactions]
        flat_ccs = [ccs for reactions in per_mol_reactions for ccs, _ in reactions]
        mol_offsets = []
        offset = 0
        for reactions in per_mol_reactions:
            mol_offsets.append(offset)
            offset += len(reactions)
        mol_offsets.append(offset)  # sentinel

        if flat_rxns:
            try:
                mapped_all = list(self.mapper.map_reactions(flat_rxns))
                logging.info("Mapper called once for %d reactions across %d molecules.", len(flat_rxns), len(prepared))
            except Exception as exc:
                logging.info("Mapper error for %d reactions: %s", len(flat_rxns), exc)
                mapped_all = [None] * len(flat_rxns)
        else:
            mapped_all = []

        # --- Phase C: Per-molecule template extraction, DICT update, result building ---
        outputs = []
        for owner_idx, item in enumerate(prepared):
            target = item['target']
            start = mol_offsets[owner_idx]
            end = mol_offsets[owner_idx + 1]

            ccs_slice = flat_ccs[start:end]
            mapped_slice = mapped_all[start:end] if mapped_all else []

            templates = self._extract_templates_from_mapped(ccs_slice, mapped_slice)

            if self.use_DICT:
                _tid = task_ids[owner_idx] if task_ids is not None else None
                self.renew_DICT(templates, task_id=_tid, target_smiles=target)

            rule_k = [rule for _, rule in templates]
            sum_rule = (item['dict_rules'] + rule_k) if self.use_DICT else rule_k
            outputs.append(self._rules_to_result(target, sum_rule))

        return outputs

    def run(self, x, topk=20, task_id=None):
        if task_id is None:
            outputs = self.run_batch([x], topk=topk, task_ids=None)
        else:
            outputs = self.run_batch([x], topk=topk, task_ids=[task_id])
        if len(outputs) == 0:
            return None
        return outputs[0]
