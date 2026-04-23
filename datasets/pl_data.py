import torch
import torch_scatter
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import pdb

FOLLOW_BATCH = ('protein_element', 'ligand_element', 'ligand_bond_type',) 
# PyG DataLoader 옵션. (batching 할 때 각 노드가 어떤 batch에 속하는지도 같이 저장~) (, : 튜플임을 명확히 하기 위한거임. 원소가 1개일 때를 대비한 안전한 습관)

class ProteinLigandData(Data):

    def __init__(self, *args, **kwargs): 
        # **kwargs: 이름이 정해지지 않은 여러 개의 키워드 인자들을 딕셔너리로 받는 문법
        # *args: 위치 인자 여러개 -> 튜플
        # *kwargs: 키워드 인자 여러 개 -> 딕셔너리
        super().__init__(*args, **kwargs)

    @staticmethod 
    # class 안에 있지만, self나 class를 자동으로 받지 않는 함수. 그냥 클래스 안에 있는 일반 함수. (클래스와 '논리적으로 관련'은 있지만, self를 쓰지 않을 때 사용함.)
    def from_protein_ligand_dicts(protein_dict=None, ligand_dict=None, **kwargs):
        # data.py에서 만든 dictionary를 PyG Data 객체로 변환.
        instance = ProteinLigandData(**kwargs)

        if protein_dict is not None:
            for key, item in protein_dict.items():
                instance['protein_' + key] = item

        if ligand_dict is not None:
            for key, item in ligand_dict.items():
                instance['ligand_' + key] = item

        instance['ligand_nbh_list'] = {i.item(): [j.item() for k, j in enumerate(instance.ligand_bond_index[1])
                                                  if instance.ligand_bond_index[0, k].item() == i]
                                       for i in instance.ligand_bond_index[0]}
        return instance

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'ligand_bond_index':
            return self['ligand_element'].size(0)
        else:
            return super().__inc__(key, value)

class ProteinLigandDataLoader(DataLoader):

    def __init__(
            self,
            dataset,
            batch_size=1,
            shuffle=False,
            follow_batch=FOLLOW_BATCH,
            **kwargs
    ):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, follow_batch=follow_batch, **kwargs)


def torchify_dict(data):
    output = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            output[k] = torch.from_numpy(v)
        else:
            output[k] = v
    return output


def get_batch_connectivity_matrix(ligand_batch, ligand_bond_index, ligand_bond_type, ligand_bond_batch):
    batch_ligand_size = torch_scatter.segment_coo(
        torch.ones_like(ligand_batch),
        ligand_batch,
        reduce='sum',
    )
    batch_index_offset = torch.cumsum(batch_ligand_size, 0) - batch_ligand_size
    batch_size = len(batch_index_offset)
    batch_connectivity_matrix = []
    for batch_index in range(batch_size):
        start_index, end_index = ligand_bond_index[:, ligand_bond_batch == batch_index]
        start_index -= batch_index_offset[batch_index]
        end_index -= batch_index_offset[batch_index]
        bond_type = ligand_bond_type[ligand_bond_batch == batch_index]
        # NxN connectivity matrix where 0 means no connection and 1/2/3/4 means single/double/triple/aromatic bonds.
        connectivity_matrix = torch.zeros(batch_ligand_size[batch_index], batch_ligand_size[batch_index],
                                          dtype=torch.int)
        for s, e, t in zip(start_index, end_index, bond_type):
            connectivity_matrix[s, e] = connectivity_matrix[e, s] = t
        batch_connectivity_matrix.append(connectivity_matrix)
    return batch_connectivity_matrix
