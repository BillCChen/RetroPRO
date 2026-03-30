from __future__ import print_function
import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem
import rdchiral
from rdchiral.main import rdchiralRunText, rdchiralRun
from rdchiral.initialization import rdchiralReaction, rdchiralReactants
from .mlp_policies import load_parallel_model , preprocess
from collections import defaultdict, OrderedDict
import logging
def merge(reactant_d):
    ret = []
    for reactant, l in reactant_d.items():
        ss, ts = zip(*l)
        ret.append((reactant, sum(ss), list(ts)[0]))
    reactants, scores, templates = zip(*sorted(ret,key=lambda item : item[1], reverse=True))
    return list(reactants), list(scores), list(templates)
import time
day_hour_min_sec = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
import pickle
from .tp_free_tools import random_substructure,rand_aug_smiles,repeat_retro_k
from .tp_free_tools import Load_Retro_Model, Load_Forward_Model
from rdchiral import template_extractor as extractor
from rxnmapper import RXNMapper,BatchedMapper
class TP_free_Model(object):
    def __init__(self,retro_model_path,retro_topk,forward_model_path,forwad_topk=1,CCS=True,RD_list=[(9,0)],DICT=True, device=-1):
        super(TP_free_Model, self).__init__()
        self.device = device
        self.use_CCS = CCS
        self.RD_list = RD_list
        self.use_DICT = DICT
        self.DICT = defaultdict(list)
        self.retro_model = Load_Retro_Model(retro_model_path,beam_size=10, n_best=retro_topk, batch_size=20)
        print("Successing Loading Retro Model!")
        self.retro_topk = retro_topk
        self.forward_model = Load_Forward_Model(forward_model_path,beam_size=10, n_best=forwad_topk, batch_size=20)
        print("Successing Loading forward Model!")
        self.mapper = BatchedMapper(batch_size=10)
        print("Successing Loading Reaction-mapping Model!")
    def random_sampling(self, x,RD_list,topk):
        """
        对输入分子x进行随机子结构采样，并通过随机SMILES增强生成多样化的候选反应物。
        """
        # R,D = 4,5
        # R, D = 9,0
        output = set()
        for R,D in RD_list:
            if self.use_CCS:
                sub_smiles = random_substructure(x, r=R, d=D,num = topk//len(RD_list))
            else:
                sub_smiles = [x for _ in range(topk//len(RD_list))]
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
        """
        检查SMILES字符串的有效性
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol
        except:
            return None

    def invalid_retro_filter(self, smiles, retro):
        """
        对逆合成结果进行过滤，去除包含无效SMILES的结果。
        制作成「CCS_mol」->「retro_smiles」的对应关系。
        """
        # assert len(smiles) == len(retro),f"Length mismatch: smiles({len(smiles)}) vs retro({len(retro)})"
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
    def filter(self,x, smiles, retro, forward):
        x_mol = Chem.MolFromSmiles(x)
        cano_smi_reactions = []
        for CCS_smi, r_smi, f_smi in zip(smiles, retro, forward):
            f_mol = self.check_smiles_valid(f_smi)
            if f_mol is None:
                continue
            smi_mol = Chem.MolFromSmiles(CCS_smi)
            if self.mol2cano_smiles(smi_mol) == self.mol2cano_smiles(f_mol):
            # if x_mol.hasSubstructMatch(f_mol):
                cano_smi_reactions.append(  (self.mol2cano_smiles(smi_mol), f"{CCS_smi}>>{self.smi2cano_smiels(r_smi)}")  )
        logging.info(f"Filtered {len(cano_smi_reactions)} valid reactions from {len(smiles)} candidates.")
        # for s_r in smi_reactions:
        #     print(f"Valid reaction: {s_r[1]}")
        return cano_smi_reactions
    def extract_templates(self, cano_smi_reactions):
        reactions = [rxn for ccs_smi,rxn in cano_smi_reactions]
        ccs_smi = [ccs_smi for ccs_smi,rxn in cano_smi_reactions]
        mapped_reactions = list(self.mapper.map_reactions(list(reactions)))
        templates = []
        for cano_smi,mapped_rxn in zip(ccs_smi, mapped_reactions):
            if mapped_rxn is None:
                continue
            mapped_rxn_dict = {
                'reactants':mapped_rxn.split('>>')[1], # 
                'products':mapped_rxn.split('>>')[0],
                '_id':'test'
                  }
            template = extractor.extract_from_reaction(mapped_rxn_dict)
            # print(template)
            if 'reaction_smarts' not in template.keys():
                continue
            template = template['reaction_smarts']
            templates.append( (cano_smi,template)  )
        return templates
    def renew_DICT(self, templates):
        for key,rule in templates:
            # self.DICT = defaultdict(list)
            self.DICT[key].append(rule)
        save_dir = f"results/tp_free_DICT_{day_hour_min_sec}.pkl"
        pickle.dump(self.DICT, open(save_dir, 'wb'))
    def run(self, x, topk=20):
        # 去除x的立体化学信息
        x = Chem.MolToSmiles(Chem.MolFromSmiles(x), isomericSmiles=False)
        smiles = self.random_sampling(x,self.RD_list, topk)
        aug_smiles = []
        if self.use_DICT:
            rule_k_from_DICT = []
            for smi in smiles:
                if self.smi2cano_smiels(smi) in self.DICT.keys():
                    rule_k_from_DICT.extend(self.DICT[smi])
                    smiles.remove(smi)
                    aug_smiles.extend(rand_aug_smiles(smi,1))
                else:
                    aug_smiles.extend(rand_aug_smiles(smi,5))
        else:
            for smi in smiles:
                aug_smiles.extend(rand_aug_smiles(smi,5))
        # print(f"smiles: {smiles}")
        if len(aug_smiles) == 0:
            aug_smiles = rand_aug_smiles(x,topk)
        if len(aug_smiles) == 0:
            aug_smiles = [x]
        try:
            retro = self.retro_model.inference(aug_smiles)
        except Exception as e:
            print(f"Retro model inference error: {e}")
            print(f"Input smiles: {aug_smiles}")
            # raise e
        aug_smiles = repeat_retro_k(aug_smiles,self.retro_topk)
        # print(f"{self.retro_topk} * aug_smiles: {aug_smiles[:5]}")
        smiles, retro = self.invalid_retro_filter(aug_smiles, retro)
        # s_r = [f"{s}.{r}" for s,r in zip(smiles,retro)]
        # print(f"filter retro: {retro[:5]}")
        if len(retro) > 0:
            forward = self.forward_model.inference(retro)
        else:
            forward = []
        print(f"forward: {forward[:5]}")
        reactions = self.filter(x,aug_smiles, retro, forward)
        templates = self.extract_templates(reactions)
        if self.use_DICT:
            self.renew_DICT(templates)
        rule_k = [rule for _,rule in templates]
        reactants = []
        scores = []
        templates = []
        if self.use_DICT:
            sum_rule = rule_k_from_DICT + rule_k
        else:
            sum_rule = rule_k
        for i , rule in enumerate(sum_rule):
            out1 = []
            try:
                # print(f'rule{i}: {rule}')
                # rdchiralRunText(rule, x)
                all_out = AllChem.ReactionFromSmarts(rule).RunReactants((Chem.MolFromSmiles(x),))
                if len(all_out) == 0: continue
                # if len(out1) > 1: print("more than two reactants."),print(out1)
                out1 = [Chem.MolToSmiles(x) for x in all_out[0]]
                for smi in out1:
                    if Chem.MolFromSmiles(smi) is None:
                        continue
                out1 = ['.'.join(sorted(out1))]
                # out1 = sorted(out1)
                for reactant in out1:
                    reactants.append(reactant)
                    scores.append(1.0)
                    templates.append(rule)
            except ValueError as e:
                # print(f"Error processing rule: {rule}, error: {e}")
                pass
            # except KeyError:
            #     pass
        print(f" groups num : {len(reactants)}")
        if len(reactants) == 0: 
            print(f" groups num : {len(reactants)}")
            return None
        reactants_d = defaultdict(list)
        for r, s, t in zip(reactants, scores, templates):
            if '.' in r:
                str_list = sorted(r.strip().split('.'))
                reactants_d['.'.join(str_list)].append((s, t))
            else:
                reactants_d[r].append((s, t))

        reactants, scores, templates = merge(reactants_d)
        print(f"final groups num : {len(reactants)}")
        total = sum(scores)
        scores = [s / total for s in scores]
        return {'reactants':reactants,
                'scores' : scores,
                'template' : templates}


