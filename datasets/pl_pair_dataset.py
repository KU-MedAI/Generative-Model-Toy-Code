import os
import pickle
import lmdb
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.data import PDBProtein, parse_sdf_file
from .pl_data import ProteinLigandData, torchify_dict

class PocketLigandPairDataset(Dataset):

    def __init__(self, raw_path, transform=None, version='final'):
        super().__init__()
        self.raw_path = raw_path.rstrip('/')
        self.index_path = os.path.join(self.raw_path, 'index.pkl')
        self.processed_path = os.path.join(os.path.dirname(self.raw_path),
                                           os.path.basename(self.raw_path) + f'_processed_{version}.lmdb') # 전처리 결과LMDB 파일 경로
        self.transform = transform
        self.db = None

        self.keys = None

        if not os.path.exists(self.processed_path):
            print(f'{self.processed_path} does not exist, begin processing data')
            self._process()

    def _connect_db(self):

        """
            Establish read-only database connection
            학습할 때는 쓰기가 필요 없으니까 read only
        """
        assert self.db is None, 'A connection has already been opened.'
        self.db = lmdb.open(
            self.processed_path,
            map_size=10*(1024*1024*1024),   # 10GB
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))

    def _close_db(self):
        self.db.close()
        self.db = None
        self.keys = None
        
    def _process(self):
        '''
        원본 -> LMDB로 '전처리/캐싱'
        '''
        db = lmdb.open(
            self.processed_path,
            map_size=10*(1024*1024*1024),   # 10GB
            create=True,
            subdir=False,
            readonly=False,  # Writable
        )
        with open(self.index_path, 'rb') as f:
            index = pickle.load(f)

        num_skipped = 0
        with db.begin(write=True, buffers=True) as txn:
            for i, (pocket_fn, ligand_fn, *_) in enumerate(tqdm(index)):
                # index의 각 행은 최소 pocket_fn, ligand_fn, 그 뒤에 추가정보(*_로 무시하긴함)로 구성
                if pocket_fn is None: continue
                try:
                    # data_prefix = '/data/work/jiaqi/binding_affinity'
                    data_prefix = self.raw_path
                    pocket_dict = PDBProtein(os.path.join(data_prefix, pocket_fn)).to_dict_atom()
                    ligand_dict = parse_sdf_file(os.path.join(data_prefix, ligand_fn))
                    data = ProteinLigandData.from_protein_ligand_dicts(
                        protein_dict=torchify_dict(pocket_dict),
                        ligand_dict=torchify_dict(ligand_dict),
                    )
                    data.protein_filename = pocket_fn
                    data.ligand_filename = ligand_fn
                    data = data.to_dict()  # avoid torch_geometric version issue
                    txn.put(                             # txn.put : LMDB 트랜잭선(txn)에 데이터를 저장하는 메서드. 첫번째 인자:key, 두번째인자: value
                        key=str(i).encode(),
                        value=pickle.dumps(data)
                    )
                except:
                    num_skipped += 1
                    print('Skipping (%d) %s, Error: %s' % (num_skipped, ligand_fn, str(e)))
                    continue
        db.close()
    def __len__(self):
        if self.db is None:
            self._connect_db()
        return len(self.keys)

    def __getitem__(self, idx):
        data = self.get_ori_data(idx)   #원본데이터(=LMDB에서 복원) 가져옴
        if self.transform is not None:
            data = self.transform(data)
        return data

    def get_ori_data(self, idx):
        if self.db is None:
            self._connect_db()
        key = self.keys[idx]
        data = pickle.loads(self.db.begin().get(key))
        data = ProteinLigandData(**data)
        data.id = idx
        assert data.protein_pos.size(0) > 0
        return data
        

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    args = parser.parse_args()

    dataset = PocketLigandPairDataset(args.path)
    print(len(dataset), dataset[0])
