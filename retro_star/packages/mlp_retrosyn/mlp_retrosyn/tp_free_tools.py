import os
import re

try:
    from onmt.translate.translator import build_translator
except Exception:
    build_translator = None
from types import SimpleNamespace
from rdkit import Chem
import random
from collections import defaultdict, deque
def repeat_retro_k(smi_list, k):
    assert type(smi_list) == list
    out = []
    for smi in smi_list:
        out.extend([smi]*k)
    return out

def rand_aug_smiles(smi,num=1):
    """
    输入一个SMILES字符串，随机生成num个不同的SMILES字符串作为增强数据。
    
    参数:
    smi: 输入的SMILES字符串
    num: 需要生成的增强SMILES数量
    
    返回:
    包含num个不同增强SMILES字符串的列表
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return []
    
    smiles_set = set()
    attempts = 0
    max_attempts = num * 10  # 防止死循环，设置最大尝试次数
    
    while len(smiles_set) < num and attempts < max_attempts:
        rand_smi = Chem.MolToSmiles(mol, doRandom=True)
        if rand_smi != smi:
            smiles_set.add(rand_smi)
        attempts += 1
    
    return list(smiles_set)
def random_substructure(smiles, r=4, d=2, num=1):
    """
    输入一个RDKit分子对象，拓展半径r和搜索半径d，随机输出一个子结构的SMILES字符串。
    
    参数:
    mol: RDKit的Mol对象
    r: 整数，拓展半径
    d: 整数，搜索半径
    
    返回:
    子结构的SMILES字符串
    """
    # 设定随机种子以确保可重复性
    # random.seed(42)
    
    mol = Chem.MolFromSmiles(smiles)
    # 如果分子中没有键，则返回整个分子的SMILES
    if mol.GetNumBonds() == 0:
        return Chem.MolToSmiles(mol)
    
    # 预计算所有芳香环信息
    rings = mol.GetRingInfo().AtomRings()
    aromatic_rings = []
    for ring in rings:
        is_aromatic = True
        for idx in ring:
            atom = mol.GetAtomWithIdx(idx)
            if not atom.GetIsAromatic():
                is_aromatic = False
                break
        if is_aromatic:
            aromatic_rings.append(ring)
    
    # 构建原子到所属芳香环所有原子的映射
    atom_to_aromatic_atoms = defaultdict(set)
    for ring in aromatic_rings:
        for idx1 in ring:
            for idx2 in ring:
                atom_to_aromatic_atoms[idx1].add(idx2)
    
    # 随机选择一个键
    bonds = list(mol.GetBonds())
    if num > len(bonds):
        select_bond = bonds + random.choices(bonds, k=num - len(bonds))
    else:
        select_bond = random.sample(bonds, num)
    smiles_list = []
    for select in select_bond:
        start_atoms = [select.GetBeginAtom(), select.GetEndAtom()]
        
        # 第一步BFS: 拓展距离r内的原子，遇到芳香原子时添加整个芳香环
        visited1 = set()
        dist_dict1 = {}
        queue1 = deque()
        for atom in start_atoms:
            idx = atom.GetIdx()
            visited1.add(idx)
            dist_dict1[idx] = 0
            queue1.append(atom)
        
        while queue1:
            atom = queue1.popleft()
            idx = atom.GetIdx()
            current_dist = dist_dict1[idx]
            if current_dist < r:
                for neighbor in atom.GetNeighbors():
                    nidx = neighbor.GetIdx()
                    if nidx not in visited1:
                        visited1.add(nidx)
                        dist_dict1[nidx] = current_dist + 1
                        queue1.append(neighbor)
            # 如果当前原子是芳香原子，则添加整个芳香环
            if atom.GetIsAromatic():
                aromatic_atoms = atom_to_aromatic_atoms.get(idx, set())
                for aidx in aromatic_atoms:
                    if aidx not in visited1:
                        visited1.add(aidx)
                        # 注意：芳香环原子不加入队列，不分配距离
        
        # 第二步BFS: 搜索距离当前子结构小于d的非芳香原子
        visited2 = set(visited1)
        dist_dict2 = {}
        queue2 = deque()
        for idx in visited1:
            atom = mol.GetAtomWithIdx(idx)
            dist_dict2[idx] = 0
            queue2.append(atom)
        
        while queue2:
            atom = queue2.popleft()
            idx = atom.GetIdx()
            current_dist = dist_dict2[idx]
            for neighbor in atom.GetNeighbors():
                nidx = neighbor.GetIdx()
                if nidx in visited2:
                    continue
                if neighbor.GetIsAromatic():
                    continue  # 忽略芳香原子
                new_dist = current_dist + 1
                if new_dist < d:  # 距离小于d才添加
                    visited2.add(nidx)
                    dist_dict2[nidx] = new_dist
                    queue2.append(neighbor)
        
        # 生成子结构的SMILES
        smiles = Chem.MolFragmentToSmiles(mol, list(visited2))
        smiles_list.append(smiles)
    return smiles_list

class Load_Retro_Model:
    def __init__(self, model_path,beam_size=10, n_best=3, batch_size=25, gpu_device=0 ):
        if build_translator is None:
            raise ImportError("OpenNMT-py is required for template_free inference. Please install `OpenNMT-py==2.2.0`.")
        self.model_path = model_path
        self.gpu_device = gpu_device
        self.beam_size = beam_size
        self.n_best = n_best
        self.batch_size = batch_size
        print("Loading Retro Model from %s" % model_path)
        
        # # 构建 Translator,只加载一次
        opt = SimpleNamespace(
            models=[self.model_path],gpu=self.gpu_device,beam_size=self.beam_size,n_best=self.n_best,batch_size=self.batch_size,
            batch_type="sents",max_length=256,seed=42,block_ngram_repeat=0,ignore_when_blocking=[],replace_unk=True,verbose=False,
            report_align=False,report_time=False,attn_debug=False,align_debug=False,dump_beam="",ban_unk_token=False,phrase_table="",
            log_file="",log_file_level="0",min_length=0,max_sent_length=None,coverage_penalty="none",alpha=0.0,beta=0.0,
            stepwise_penalty=False,length_penalty="none",ratio=0.0,random_sampling_topk=0,random_sampling_topp=0.0,
            random_sampling_temp=1.0,avg_raw_probs=False,data_type="text",src=None,src_feats=None,tgt=None,tgt_prefix=False,
            shard_size=0,output="/root/z-trash/onmt_out.txt",fp32=False,int8=False,
        )
        
        self.inference_model = build_translator(opt, report_score=False, out_file=open(os.devnull, "w"))
        
    def smi_tokenizer(self,smi):
        pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)
        tokens = [token for token in regex.findall(smi)]
        # 删去 : 这个token
        tokens = [token for token in tokens if token != ':']
        assert smi == ''.join(tokens)
        return ' '.join(tokens)
    def inference(self, input_smiles):
        # 对输入做 token 化
        tokenized = [self.smi_tokenizer(smi) for smi in input_smiles]

        # 模型推理
        scores, predictions = self.inference_model.translate(
            src=tokenized,
            tgt=None,
            batch_size=self.batch_size,
            attn_debug=False,
            batch_type="sents"
        )

        # 整理输出
        translated = []
        for pred_list in predictions:  # 每个输入样本
            for smi in pred_list:      # n_best 个预测
                translated.append(smi.replace(" ", ""))

        return translated
    
class Load_Forward_Model:
    def __init__(self, model_path,beam_size=10, n_best=1, batch_size=25, gpu_device=0 ):
        if build_translator is None:
            raise ImportError("OpenNMT-py is required for template_free inference. Please install `OpenNMT-py==2.2.0`.")
        self.model_path = model_path
        self.gpu_device = gpu_device
        self.beam_size = beam_size
        self.n_best = n_best
        self.batch_size = batch_size

        # 构建 OpenNMT 参数
        opt = SimpleNamespace(
            models=[self.model_path],
            gpu=self.gpu_device,
            beam_size=self.beam_size,
            n_best=self.n_best,
            batch_size=self.batch_size,
            max_length=256,
            seed=42,
            # === 需要补充的字段 ===
            block_ngram_repeat=0,
            ignore_when_blocking=[],
            replace_unk=True,
            verbose=False,
            report_align=False,
            report_time=False,
            attn_debug=False,
            align_debug=False,
            dump_beam="",
            ban_unk_token=False,
            phrase_table="",
            log_file="",
            log_file_level="0",
            batch_type="sents",
            min_length=0,
            max_sent_length=None,
            coverage_penalty="none",
            alpha=0.0,
            beta=0.0,
            stepwise_penalty=False,
            length_penalty="none",
            ratio=0.0,
            random_sampling_topk=0,
            random_sampling_topp=0.0,
            random_sampling_temp=1.0,
            avg_raw_probs=False,
            data_type="text",
            src=None,
            src_feats=None,
            tgt=None,
            tgt_prefix=False,
            shard_size=0,
            output="/root/z-trash/onmt_out.txt",
            fp32=False,
            int8=False,
        )

        # 构建 Translator（只加载一次）
    

        # 构建 Translator（只加载一次）
        self.inference_model = build_translator(opt, report_score=False, out_file=open(os.devnull, "w"))

    def smi_tokenizer(self,smi):
        if smi == '':
            return ''
        else:
            pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
            regex = re.compile(pattern)
            tokens = [token for token in regex.findall(smi)]
            # 删去 : 这个token
            tokens = [token for token in tokens if token != ':']
            assert smi == ''.join(tokens)
            return ' '.join(tokens)
    def inference(self, input_smiles):
        # 对输入做 token 化
        tokenized = [self.smi_tokenizer(smi) for smi in input_smiles]

        # 模型推理
        scores, predictions = self.inference_model.translate(
            src=tokenized,
            tgt=None,
            batch_size=self.batch_size,
            attn_debug=False,
            batch_type="sents"
        )

        # 整理输出
        translated = []
        for pred_list in predictions:  # 每个输入样本
            for smi in pred_list:      # n_best 个预测
                translated.append(smi.replace(" ", ""))

        return translated
