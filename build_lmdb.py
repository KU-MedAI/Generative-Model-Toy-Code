import os
import glob
import pickle
import lmdb
import numpy as np
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem.rdchem import BondType, HybridizationType
from rdkit import RDConfig
from tqdm import tqdm

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

AA_NAME_NUMBER = {
    'ALA': 0, 'CYS': 1, 'ASP': 2, 'GLU': 3, 'PHE': 4,
    'GLY': 5, 'HIS': 6, 'ILE': 7, 'LYS': 8, 'LEU': 9,
    'MET': 10, 'ASN': 11, 'PRO': 12, 'GLN': 13, 'ARG': 14,
    'SER': 15, 'THR': 16, 'VAL': 17, 'TRP': 18, 'TYR': 19,
}
BACKBONE_NAMES = {"CA", "C", "N", "O"}

BOND_TYPES = {
    BondType.UNSPECIFIED: 0,
    BondType.SINGLE: 1,
    BondType.DOUBLE: 2,
    BondType.TRIPLE: 3,
    BondType.AROMATIC: 4,
}

ATOM_FAMILIES = [
    'Acceptor', 'Donor', 'Aromatic', 'Hydrophobe',
    'LumpedHydrophobe', 'NegIonizable', 'PosIonizable', 'ZnBinder',
]
ATOM_FAMILIES_ID = {s: i for i, s in enumerate(ATOM_FAMILIES)}

FEAT_FACTORY = ChemicalFeatures.BuildFeatureFactory(
    os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
)

# ──────────────────────────────────────────────
# Protein Parsing
# ──────────────────────────────────────────────

def parse_pdb(path):
    """atom-level 정보들 parsing."""
    ptable = Chem.GetPeriodicTable()
    elements, positions, is_backbone, atom_to_aa = [], [], [], []

    with open(path, 'r') as f:
        for line in f:
            if line[:6].strip() == 'ENDMDL':
                break
            if line[:6].strip() != 'ATOM':
                continue

            elem_symb = line[76:78].strip().capitalize()
            if not elem_symb:
                elem_symb = line[13:14]
            res_name = line[17:20].strip()
            atom_name = line[12:16].strip()

            if res_name not in AA_NAME_NUMBER:
                continue

            elements.append(ptable.GetAtomicNumber(elem_symb))
            positions.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            is_backbone.append(atom_name in BACKBONE_NAMES)
            atom_to_aa.append(AA_NAME_NUMBER[res_name])

    return {
        'element': np.array(elements, dtype=np.int64),
        'pos': np.array(positions, dtype=np.float32),
        'is_backbone': np.array(is_backbone, dtype=bool),
        'atom_to_aa_type': np.array(atom_to_aa, dtype=np.int64),
    }


# ──────────────────────────────────────────────
# Ligand Parsing
# ──────────────────────────────────────────────

def parse_sdf(path):
    """atom/bond-level arrays + pharmacophoric features 정보 parsing."""
    mol = Chem.MolFromMolFile(path, sanitize=False)
    Chem.SanitizeMol(mol)
    mol = Chem.RemoveHs(mol)
    ptable = Chem.GetPeriodicTable()

    num_atoms = mol.GetNumAtoms()
    pos = np.array(mol.GetConformer().GetPositions(), dtype=np.float32)

    feat_mat = np.zeros([num_atoms, len(ATOM_FAMILIES)], dtype=np.int64)
    for feat in FEAT_FACTORY.GetFeaturesForMol(mol):
        feat_mat[feat.GetAtomIds(), ATOM_FAMILIES_ID[feat.GetFamily()]] = 1

    elements = []
    hybridization = []
    accum_pos, accum_mass = np.zeros(3, dtype=np.float64), 0.0
    for i in range(num_atoms):
        atom = mol.GetAtomWithIdx(i)
        anum = atom.GetAtomicNum()
        elements.append(anum)
        hybridization.append(str(atom.GetHybridization()).split('.')[-1])
        w = ptable.GetAtomicWeight(anum)
        accum_pos += pos[i] * w
        accum_mass += w
    center_of_mass = (accum_pos / accum_mass).astype(np.float32)

    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt = BOND_TYPES[bond.GetBondType()]
        row += [i, j]
        col += [j, i]
        edge_type += [bt, bt]
    edge_index = np.array([row, col], dtype=np.int64)
    edge_type = np.array(edge_type, dtype=np.int64)

    perm = (edge_index[0] * num_atoms + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]

    return {
        'element': np.array(elements, dtype=np.int64),
        'pos': pos,
        'bond_index': edge_index,
        'bond_type': edge_type,
        'center_of_mass': center_of_mass,
        'atom_feature': feat_mat,
        'hybridization': hybridization,
        'smiles': Chem.MolToSmiles(mol),
    }


# ──────────────────────────────────────────────
# LMDB Builder
# ──────────────────────────────────────────────

def build_lmdb(complex_dir, output_path, map_size=10 * (1024**3)):
    """
        {uniprot_id}_pocket.pdb   (1 file)
        {ligand_id}_ligand.sdf    (3 files)
    하나의 LMDB entry는 (pocket, ligand) pair에 대한 정보를 포함
    """
    complex_dirs = sorted(glob.glob(os.path.join(complex_dir, '*')))
    db = lmdb.open(output_path, map_size=map_size, create=True, subdir=False, readonly=False)   # 데이터베이스 열기

    num_skipped = 0
    idx = 0   # 인덱스 초기화
    with db.begin(write=True, buffers=True) as txn:
        for cdir in tqdm(complex_dirs, desc='Building LMDB'):   # 디렉토리 순회
            if not os.path.isdir(cdir):
                continue

            pdb_files = glob.glob(os.path.join(cdir, '*_pocket.pdb')) # 포켓 파일 검색
            sdf_files = sorted(glob.glob(os.path.join(cdir, '*_ligand.sdf'))) # 리간드 파일 검색
            if not pdb_files or not sdf_files:
                num_skipped += 1 # 건너뛴 파일 수 증가
                continue

            pocket_path = pdb_files[0] # 포켓 파일 경로
            try:
                protein_dict = parse_pdb(pocket_path) # 포켓 파일 파싱
            except Exception as e:
                num_skipped += len(sdf_files) # 건너뛴 파일 수 증가
                print(f'Skipped pocket ({os.path.basename(cdir)}) — {e}')
                continue

            for ligand_path in sdf_files:
                try:
                    ligand_dict = parse_sdf(ligand_path) # 리간드 파일 파싱

                    data = {}
                    for k, v in protein_dict.items(): # 포켓 파일 정보 추가
                        data[f'protein_{k}'] = v
                    for k, v in ligand_dict.items():
                        data[f'ligand_{k}'] = v # 리간드 파일 정보 추가

                    data['protein_filename'] = os.path.basename(pocket_path) # 포켓 파일 이름 추가
                    data['ligand_filename'] = os.path.basename(ligand_path) # 리간드 파일 이름 추가
                    data['complex_name'] = os.path.basename(cdir)
                    # 데이터베이스에 저장
                    txn.put(key=str(idx).encode(), value=pickle.dumps(data))
                    idx += 1
                except Exception as e:
                    num_skipped += 1
                    print(f'Skipped ({num_skipped}): {os.path.basename(ligand_path)} — {e}')

    db.close()
    print(f'Done. {idx} entries saved to {output_path} (skipped {num_skipped})')


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

if __name__ == '__main__':
    COMPLEX_DIR = '/home/tech/project/chanju/targetdiff_reproduction/reproduction/best_affinity_ligand_complex'
    OUTPUT_PATH = '/home/tech/project/chanju/targetdiff_reproduction/reproduction/best_affinity_complex_processed.lmdb'
    build_lmdb(complex_dir=COMPLEX_DIR, output_path=OUTPUT_PATH)
