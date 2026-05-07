"""
Molecular reconstruction from 3D coordinates and atom types.
Adapted from https://github.com/mattragoza/liGAN (GPL v2.0)
"""
import itertools

import numpy as np
from rdkit.Chem import AllChem as Chem
from rdkit import Geometry
from openbabel import openbabel as ob
from scipy.spatial.distance import pdist, squareform


class MolReconsError(Exception):
    pass


def make_obmol(xyz, atomic_numbers):
    mol = ob.OBMol()
    mol.BeginModify()
    atoms = []
    for (x, y, z), anum in zip(xyz, atomic_numbers):
        atom = mol.NewAtom()
        atom.SetAtomicNum(anum)
        atom.SetVector(x, y, z)
        atoms.append(atom)
    return mol, atoms


def reachable_r(a, b, seenbonds):
    for nbr in ob.OBAtomAtomIter(a):
        bond = a.GetBond(nbr).GetIdx()
        if bond not in seenbonds:
            seenbonds.add(bond)
            if nbr == b:
                return True
            elif reachable_r(nbr, b, seenbonds):
                return True
    return False


def reachable(a, b):
    if a.GetExplicitDegree() == 1 or b.GetExplicitDegree() == 1:
        return False
    seenbonds = set([a.GetBond(b).GetIdx()])
    return reachable_r(a, b, seenbonds)


def forms_small_angle(a, b, cutoff=60):
    for nbr in ob.OBAtomAtomIter(a):
        if nbr != b:
            if b.GetAngle(a, nbr) < cutoff:
                return True
    return False


def count_nbrs_of_elem(atom, atomic_num):
    count = 0
    for nbr in ob.OBAtomAtomIter(atom):
        if nbr.GetAtomicNum() == atomic_num:
            count += 1
    return count


def connect_the_dots(mol, atoms, indicators, covalent_factor=1.3):
    pt = Chem.GetPeriodicTable()
    if len(atoms) == 0:
        return
    mol.BeginModify()

    coords = np.array([(a.GetX(), a.GetY(), a.GetZ()) for a in atoms])
    dists = squareform(pdist(coords))

    for i, j in itertools.combinations(range(len(atoms)), 2):
        a, b = atoms[i], atoms[j]
        a_r = ob.GetCovalentRad(a.GetAtomicNum()) * covalent_factor
        b_r = ob.GetCovalentRad(b.GetAtomicNum()) * covalent_factor
        if dists[i, j] < a_r + b_r:
            flag = ob.OB_AROMATIC_BOND if (indicators and indicators[i] and indicators[j]) else 0
            mol.AddBond(a.GetIdx(), b.GetIdx(), 1, flag)

    atom_maxb = {}
    for i, a in enumerate(atoms):
        maxb = min(ob.GetMaxBonds(a.GetAtomicNum()), pt.GetDefaultValence(a.GetAtomicNum()))
        if a.GetAtomicNum() == 16 and count_nbrs_of_elem(a, 8) >= 2:
            maxb = 6
        atom_maxb[a.GetIdx()] = maxb

    for bond in ob.OBMolBondIter(mol):
        a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
        if atom_maxb[a1.GetIdx()] == 1 and atom_maxb[a2.GetIdx()] == 1:
            mol.DeleteBond(bond)

    def get_bond_info(biter):
        binfo = []
        for bond in biter:
            a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
            ideal = ob.GetCovalentRad(a1.GetAtomicNum()) + ob.GetCovalentRad(a2.GetAtomicNum())
            binfo.append((bond.GetLength() / ideal, bond))
        binfo.sort(reverse=True, key=lambda t: t[0])
        return binfo

    for stretch, bond in get_bond_info(ob.OBMolBondIter(mol)):
        a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
        if stretch > 1.2 or forms_small_angle(a1, a2) or forms_small_angle(a2, a1):
            if not reachable(a1, a2):
                continue
            mol.DeleteBond(bond)

    hypers = [(atom_maxb[a.GetIdx()], a.GetExplicitValence() - atom_maxb[a.GetIdx()], a) for a in atoms]
    hypers.sort(key=lambda aa: (aa[0], -aa[1]))
    for mb, diff, a in hypers:
        if a.GetExplicitValence() <= atom_maxb[a.GetIdx()]:
            continue
        for stretch, bond in get_bond_info(ob.OBAtomBondIter(a)):
            if stretch < 0.9:
                continue
            a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
            if a1.GetExplicitValence() > atom_maxb[a1.GetIdx()] or a2.GetExplicitValence() > atom_maxb[a2.GetIdx()]:
                if not reachable(a1, a2):
                    continue
                mol.DeleteBond(bond)
                if a.GetExplicitValence() <= atom_maxb[a.GetIdx()]:
                    break
    mol.EndModify()


def fixup(atoms, mol, indicators):
    mol.SetAromaticPerceived(True)
    for i, atom in enumerate(atoms):
        if indicators is not None:
            if indicators[i]:
                atom.SetAromatic(True)
                atom.SetHyb(2)
            else:
                atom.SetAromatic(False)
        if (atom.GetAtomicNum() in (7, 8)) and atom.IsInRing():
            acnt = sum(1 for nbr in ob.OBAtomAtomIter(atom) if nbr.IsAromatic())
            if acnt > 1:
                atom.SetAromatic(True)


def calc_valence(rdatom):
    return sum(bond.GetBondTypeAsDouble() for bond in rdatom.GetBonds())


UPGRADE_BOND_ORDER = {Chem.BondType.SINGLE: Chem.BondType.DOUBLE, Chem.BondType.DOUBLE: Chem.BondType.TRIPLE}


def convert_ob_mol_to_rd_mol(ob_mol):
    ob_mol.DeleteHydrogens()
    n_atoms = ob_mol.NumAtoms()
    rd_mol = Chem.RWMol()
    rd_conf = Chem.Conformer(n_atoms)

    for ob_atom in ob.OBMolAtomIter(ob_mol):
        rd_atom = Chem.Atom(ob_atom.GetAtomicNum())
        if ob_atom.IsAromatic() and ob_atom.IsInRing() and ob_atom.MemberOfRingSize() <= 6:
            rd_atom.SetIsAromatic(True)
        i = rd_mol.AddAtom(rd_atom)
        v = ob_atom.GetVector()
        rd_conf.SetAtomPosition(i, Geometry.Point3D(v.GetX(), v.GetY(), v.GetZ()))
    rd_mol.AddConformer(rd_conf)

    for ob_bond in ob.OBMolBondIter(ob_mol):
        i, j = ob_bond.GetBeginAtomIdx() - 1, ob_bond.GetEndAtomIdx() - 1
        order = ob_bond.GetBondOrder()
        bt = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}[order]
        rd_mol.AddBond(i, j, bt)
        if ob_bond.IsAromatic():
            rd_mol.GetBondBetweenAtoms(i, j).SetIsAromatic(True)

    rd_mol = Chem.RemoveHs(rd_mol, sanitize=False)
    pt = Chem.GetPeriodicTable()

    nonsingles = []
    positions = rd_mol.GetConformer().GetPositions()
    for bond in rd_mol.GetBonds():
        if bond.GetBondType() in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE):
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            nonsingles.append((np.linalg.norm(positions[i] - positions[j]), bond))
    nonsingles.sort(reverse=True, key=lambda t: t[0])

    for d, bond in nonsingles:
        a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
        if calc_valence(a1) > pt.GetDefaultValence(a1.GetAtomicNum()) or \
                calc_valence(a2) > pt.GetDefaultValence(a2.GetAtomicNum()):
            btype = Chem.BondType.DOUBLE if bond.GetBondType() == Chem.BondType.TRIPLE else Chem.BondType.SINGLE
            bond.SetBondType(btype)

    for atom in rd_mol.GetAtoms():
        if atom.GetAtomicNum() == 7 and atom.GetDegree() == 4:
            atom.SetFormalCharge(1)

    rd_mol = Chem.AddHs(rd_mol, addCoords=True)
    positions = rd_mol.GetConformer().GetPositions()
    center = np.mean(positions[np.all(np.isfinite(positions), axis=1)], axis=0)
    for atom in rd_mol.GetAtoms():
        i = atom.GetIdx()
        if not np.all(np.isfinite(positions[i])):
            rd_mol.GetConformer().SetAtomPosition(i, center)

    try:
        Chem.SanitizeMol(rd_mol, Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE)
    except Exception:
        raise MolReconsError()

    for bond in rd_mol.GetBonds():
        a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
        if bond.GetIsAromatic():
            if not a1.GetIsAromatic() or not a2.GetIsAromatic():
                bond.SetIsAromatic(False)
        elif a1.GetIsAromatic() and a2.GetIsAromatic():
            bond.SetIsAromatic(True)

    return rd_mol


def postprocess_rd_mol(rdmol):
    rdmol = Chem.RemoveHs(rdmol)

    nbh_list = {}
    for bond in rdmol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        nbh_list.setdefault(i, []).append(j)
        nbh_list.setdefault(j, []).append(i)

    for atom in rdmol.GetAtoms():
        idx = atom.GetIdx()
        num_radical = atom.GetNumRadicalElectrons()
        if num_radical > 0:
            for j in nbh_list.get(idx, []):
                if j <= idx:
                    continue
                nb_atom = rdmol.GetAtomWithIdx(j)
                nb_radical = nb_atom.GetNumRadicalElectrons()
                if nb_radical > 0:
                    bond = rdmol.GetBondBetweenAtoms(idx, j)
                    bond.SetBondType(UPGRADE_BOND_ORDER[bond.GetBondType()])
                    nb_atom.SetNumRadicalElectrons(nb_radical - 1)
                    num_radical -= 1
            atom.SetNumRadicalElectrons(0)
            num_hs = atom.GetNumExplicitHs()
            atom.SetNumExplicitHs(num_hs + num_radical)

    rdmol_edit = Chem.RWMol(rdmol)
    rings = [set(r) for r in rdmol.GetRingInfo().AtomRings()]
    for ring_a in rings:
        if len(ring_a) == 3:
            non_carbon = [i for i in ring_a if rdmol.GetAtomWithIdx(i).GetSymbol() != 'C']
            if len(non_carbon) == 2:
                rdmol_edit.RemoveBond(*non_carbon)
    rdmol = rdmol_edit.GetMol()

    for atom in rdmol.GetAtoms():
        if atom.GetFormalCharge() > 0:
            atom.SetFormalCharge(0)

    return rdmol


def reconstruct_from_generated(xyz, atomic_nums, aromatic=None):
    indicators = aromatic

    mol, atoms = make_obmol(xyz, atomic_nums)
    fixup(atoms, mol, indicators)
    connect_the_dots(mol, atoms, indicators, covalent_factor=1.3)
    fixup(atoms, mol, indicators)

    mol.AddPolarHydrogens()
    mol.PerceiveBondOrders()
    fixup(atoms, mol, indicators)

    for a in atoms:
        ob.OBAtomAssignTypicalImplicitHydrogens(a)
    fixup(atoms, mol, indicators)

    mol.AddHydrogens()
    fixup(atoms, mol, indicators)

    for ring in ob.OBMolRingIter(mol):
        if 5 <= ring.Size() <= 6:
            carbon_cnt = sum(1 for ai in ring._path if mol.GetAtom(ai).GetAtomicNum() == 6)
            aromatic_ccnt = sum(1 for ai in ring._path if mol.GetAtom(ai).GetAtomicNum() == 6 and mol.GetAtom(ai).IsAromatic())
            if aromatic_ccnt >= carbon_cnt / 2 and aromatic_ccnt != ring.Size():
                for ai in ring._path:
                    mol.GetAtom(ai).SetAromatic(True)

    for bond in ob.OBMolBondIter(mol):
        if bond.GetBeginAtom().IsAromatic() and bond.GetEndAtom().IsAromatic():
            bond.SetAromatic(True)

    mol.PerceiveBondOrders()
    rd_mol = convert_ob_mol_to_rd_mol(mol)
    try:
        rd_mol = postprocess_rd_mol(rd_mol)
    except Exception:
        raise MolReconsError()
    return rd_mol
