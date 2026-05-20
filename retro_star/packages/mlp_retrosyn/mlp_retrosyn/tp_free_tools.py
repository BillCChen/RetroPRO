import os
import pickle
import re
from contextlib import contextmanager

try:
    from onmt.translate.translator import build_translator
except Exception:
    build_translator = None
from types import SimpleNamespace
from rdkit import Chem
import random
from collections import defaultdict, deque


@contextmanager
def _onmt_torch_load_compat_context():
    """Temporarily patch torch.load for OpenNMT 2.2 + PyTorch>=2.6."""
    try:
        import torch
    except Exception:
        yield
        return

    original_load = torch.load

    def _compat_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        if kwargs.get("pickle_module") is None:
            kwargs["pickle_module"] = pickle
        return original_load(*args, **kwargs)

    torch.load = _compat_load
    try:
        yield
    finally:
        torch.load = original_load


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
# def random_substructure(smiles, r=4, d=2, num=1):
#     """
#     输入一个RDKit分子对象，拓展半径r和搜索半径d，随机输出一个子结构的SMILES字符串。
    
#     参数:
#     mol: RDKit的Mol对象
#     r: 整数，拓展半径
#     d: 整数，搜索半径
    
#     返回:
#     子结构的SMILES字符串
#     """
#     # 设定随机种子以确保可重复性
#     # random.seed(42)
    
#     mol = Chem.MolFromSmiles(smiles)
#     # 如果分子中没有键，则返回整个分子的SMILES
#     if mol.GetNumBonds() == 0:
#         return Chem.MolToSmiles(mol)
    
#     # 预计算所有芳香环信息
#     rings = mol.GetRingInfo().AtomRings()
#     aromatic_rings = []
#     for ring in rings:
#         is_aromatic = True
#         for idx in ring:
#             atom = mol.GetAtomWithIdx(idx)
#             if not atom.GetIsAromatic():
#                 is_aromatic = False
#                 break
#         if is_aromatic:
#             aromatic_rings.append(ring)
    
#     # 构建原子到所属芳香环所有原子的映射
#     atom_to_aromatic_atoms = defaultdict(set)
#     for ring in aromatic_rings:
#         for idx1 in ring:
#             for idx2 in ring:
#                 atom_to_aromatic_atoms[idx1].add(idx2)
    
#     # 随机选择一个键
#     bonds = list(mol.GetBonds())
#     if num > len(bonds):
#         select_bond = bonds + random.choices(bonds, k=num - len(bonds))
#     else:
#         select_bond = random.sample(bonds, num)
#     smiles_list = []
#     for select in select_bond:
#         start_atoms = [select.GetBeginAtom(), select.GetEndAtom()]
        
#         # 第一步BFS: 拓展距离r内的原子，遇到芳香原子时添加整个芳香环
#         visited1 = set()
#         dist_dict1 = {}
#         queue1 = deque()
#         for atom in start_atoms:
#             idx = atom.GetIdx()
#             visited1.add(idx)
#             dist_dict1[idx] = 0
#             queue1.append(atom)
        
#         while queue1:
#             atom = queue1.popleft()
#             idx = atom.GetIdx()
#             current_dist = dist_dict1[idx]
#             if current_dist < r:
#                 for neighbor in atom.GetNeighbors():
#                     nidx = neighbor.GetIdx()
#                     if nidx not in visited1:
#                         visited1.add(nidx)
#                         dist_dict1[nidx] = current_dist + 1
#                         queue1.append(neighbor)
#             # 如果当前原子是芳香原子，则添加整个芳香环
#             if atom.GetIsAromatic():
#                 aromatic_atoms = atom_to_aromatic_atoms.get(idx, set())
#                 for aidx in aromatic_atoms:
#                     if aidx not in visited1:
#                         visited1.add(aidx)
#                         # 注意：芳香环原子不加入队列，不分配距离
        
#         # 第二步BFS: 搜索距离当前子结构小于d的非芳香原子
#         visited2 = set(visited1)
#         dist_dict2 = {}
#         queue2 = deque()
#         for idx in visited1:
#             atom = mol.GetAtomWithIdx(idx)
#             dist_dict2[idx] = 0
#             queue2.append(atom)
        
#         while queue2:
#             atom = queue2.popleft()
#             idx = atom.GetIdx()
#             current_dist = dist_dict2[idx]
#             for neighbor in atom.GetNeighbors():
#                 nidx = neighbor.GetIdx()
#                 if nidx in visited2:
#                     continue
#                 if neighbor.GetIsAromatic():
#                     continue  # 忽略芳香原子
#                 new_dist = current_dist + 1
#                 if new_dist < d:  # 距离小于d才添加
#                     visited2.add(nidx)
#                     dist_dict2[nidx] = new_dist
#                     queue2.append(neighbor)
        
#         # 生成子结构的SMILES
#         smiles = Chem.MolFragmentToSmiles(mol, list(visited2))
#         smiles_list.append(smiles)
#     return smiles_list
def random_substructure(smiles, r=4, d=2, num=1, seed=None, return_info=False):
    """
    从输入 SMILES 中随机选择 num 条键，并基于每条键采样子结构。

    采样逻辑：
    1. 以随机选中的化学键为起点，计算所有原子到该键的最短图距离。
       起始键两端原子的距离定义为 0。
    2. 取距离 <= r 的原子作为第一阶段核心原子。
    3. 在第一阶段基础上，加入距离 <= r + d 的原子作为二阶段距离扩展原子。
       注意：这里不使用芳香环补全原子作为 BFS 扩展起点。
    4. 对距离驱动得到的原子进行芳香环补全，保证芳香环片段更容易生成合法 SMILES。
    5. 最终输出：
       final_atoms = distance_atoms | aromatic_completion_atoms

    检验：
    除 aromatic_completion_atoms 外，所有采样原子到起始键的图距离一定 <= r + d。

    参数:
    smiles: str
        输入分子的 SMILES。
    r: int
        第一阶段采样半径。
    d: int
        第二阶段扩展半径。
    num: int
        随机采样的起始键数量。
    seed: int or None
        随机种子。默认 None，不固定随机性。
    return_info: bool
        是否返回调试信息，包括原子编号、距离、补全原子等。

    返回:
    smiles_list 或 (smiles_list, info_list)
    """

    if r < 0 or d < 0:
        raise ValueError("r 和 d 必须是非负整数。")

    rng = random.Random(seed) if seed is not None else random

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"无法解析输入 SMILES: {smiles}")

    # 没有键时，返回整个分子
    if mol.GetNumBonds() == 0:
        out = [Chem.MolToSmiles(mol)] * num
        if return_info:
            return out, []
        return out

    # 预计算芳香环
    aromatic_rings = []
    for ring in mol.GetRingInfo().AtomRings():
        ring_set = set(ring)
        if all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring_set):
            aromatic_rings.append(ring_set)

    def shortest_distances_from_bond(bond):
        """
        计算所有原子到起始键的最短图距离。
        起始键两端原子距离为 0。
        """
        start_idxs = [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]

        dist = {idx: 0 for idx in start_idxs}
        queue = deque(start_idxs)

        while queue:
            idx = queue.popleft()
            atom = mol.GetAtomWithIdx(idx)

            for neighbor in atom.GetNeighbors():
                nidx = neighbor.GetIdx()
                if nidx not in dist:
                    dist[nidx] = dist[idx] + 1
                    queue.append(nidx)

        return dist

    def complete_aromatic_rings(seed_atoms):
        """
        对 seed_atoms 直接接触到的芳香环进行补全。

        注意：
        - seed_atoms 是距离驱动得到的原子集合。
        - 芳香环补全原子不会继续触发新的距离扩展。
        - 这里也不让补全原子递归触发更多芳香环补全，避免过度扩张。
        """
        completion_atoms = set()

        for ring in aromatic_rings:
            if ring & seed_atoms:
                completion_atoms.update(ring)

        return completion_atoms

    # 随机选择起始键
    bonds = list(mol.GetBonds())

    if num > len(bonds):
        selected_bonds = bonds + rng.choices(bonds, k=num - len(bonds))
    else:
        selected_bonds = rng.sample(bonds, num)

    smiles_list = []
    info_list = []

    for bond in selected_bonds:
        dist = shortest_distances_from_bond(bond)

        # 第一阶段：真实图距离 <= r
        radius_atoms = {
            idx for idx, atom_dist in dist.items()
            if atom_dist <= r
        }

        # 第二阶段：真实图距离 <= r + d
        # 这等价于从 radius_atoms 出发再扩展 d，
        # 但不会让芳香环补全原子参与扩展。
        distance_atoms = {
            idx for idx, atom_dist in dist.items()
            if atom_dist <= r + d
        }

        d_atoms = distance_atoms - radius_atoms

        # 芳香环补全：只用于最终输出，不参与距离传播
        aromatic_completed_atoms = complete_aromatic_rings(distance_atoms)

        final_atoms = distance_atoms | aromatic_completed_atoms

        # 检验：除芳香补全原子外，所有距离驱动原子都必须 <= r + d
        aromatic_extra_atoms = final_atoms - distance_atoms

        for idx in distance_atoms:
            if dist[idx] > r + d:
                raise AssertionError(
                    f"距离约束失败: atom {idx}, dist={dist[idx]}, r+d={r+d}"
                )

        frag_smiles = Chem.MolFragmentToSmiles(
            mol,
            atomsToUse=sorted(final_atoms),
            canonical=True,
            isomericSmiles=True
        )

        smiles_list.append(frag_smiles)

        if return_info:
            info_list.append({
                "selected_bond": (
                    bond.GetBeginAtomIdx(),
                    bond.GetEndAtomIdx()
                ),
                "radius_atoms": sorted(radius_atoms),
                "d_atoms": sorted(d_atoms),
                "distance_atoms": sorted(distance_atoms),
                "aromatic_extra_atoms": sorted(aromatic_extra_atoms),
                "final_atoms": sorted(final_atoms),
                "max_distance_in_distance_atoms": max(
                    dist[idx] for idx in distance_atoms
                ) if distance_atoms else None,
                "distances": dict(sorted(dist.items()))
            })

    if return_info:
        return smiles_list, info_list

    return smiles_list

class Load_Retro_Model:
    def __init__(self, model_path, beam_size=10, n_best=3, batch_size=512, gpu_device=0):
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
        
        with _onmt_torch_load_compat_context():
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
    def __init__(self, model_path, beam_size=10, n_best=1, batch_size=512, gpu_device=0):
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
        with _onmt_torch_load_compat_context():
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
