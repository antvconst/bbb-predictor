import torch
from rdkit import Chem
from torch_geometric.utils import sort_edge_index
from torch_geometric.data import InMemoryDataset, Data, download_url


atom_feature_map = {
    'atomic_num': list(range(0, 119)),
    'chirality': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER',
    ],
    'degree': list(range(0, 11)),
    'formal_charge': list(range(-5, 7)),
    'num_hs': list(range(0, 9)),
    'num_radical_electrons': list(range(0, 5)),
    'hybridization': [
        'UNSPECIFIED',
        'S',
        'SP',
        'SP2',
        'SP3',
        'SP3D',
        'SP3D2',
        'OTHER',
    ],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],
}


def extract_molecular_features(mol):
    assert(mol is not None)

    atom_attr = []
    for atom in mol.GetAtoms():
        x = []
        x.append(atom_feature_map['atomic_num'].index(atom.GetAtomicNum()))
        x.append(atom_feature_map['chirality'].index(str(atom.GetChiralTag())))
        x.append(atom_feature_map['degree'].index(atom.GetTotalDegree()))
        x.append(atom_feature_map['formal_charge'].index(atom.GetFormalCharge()))
        x.append(atom_feature_map['num_hs'].index(atom.GetTotalNumHs()))
        x.append(atom_feature_map['num_radical_electrons'].index(atom.GetNumRadicalElectrons()))
        x.append(atom_feature_map['hybridization'].index(str(atom.GetHybridization())))
        x.append(atom_feature_map['is_aromatic'].index(atom.GetIsAromatic()))
        x.append(atom_feature_map['is_in_ring'].index(atom.IsInRing()))
        atom_attr.append(x)
    atom_attr = torch.tensor(atom_attr, dtype=torch.long)

    edge_index = []
    for bond in mol.GetBonds():
        src = bond.GetBeginAtomIdx()
        tgt = bond.GetEndAtomIdx()
        edge_index.append([src, tgt])
        edge_index.append([src, tgt])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    edge_index = sort_edge_index(edge_index)

    return edge_index, atom_attr

def molecule_to_graph(smiles, label, name=''):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    edge_index, atom_attr = extract_molecular_features(mol)
    return Data(x=atom_attr,
                edge_index=edge_index,
                y=torch.tensor([label], dtype=torch.long),
                smiles=smiles,
                name=name)
    