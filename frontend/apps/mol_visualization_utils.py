import numpy as np
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Draw


def get_color(t):
    ORANGE = np.array([1., 0.34, 0.2])
    WHITE = np.array([1., 1., 1.])
    return tuple(t*ORANGE + (1-t)*WHITE)


def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    for atom in mol.GetAtoms():
        atom.SetChiralTag(Chem.rdchem.CHI_UNSPECIFIED)
    return mol


def draw_molecule(mol, W=None):
    if W is not None:
        num_atoms = len(W)
        highlights = W / W.max()
        atom_highlights = {atom_id: [get_color(highlights[atom_id])] for atom_id in range(num_atoms)}
        atom_radii = {atom_id: 0.5 for atom_id in range(num_atoms)}
    else:
        atom_highlights = {}
        atom_radii = {}
    drawer = Draw.MolDraw2DSVG(400, 125)
    drawer.DrawMoleculeWithHighlights(mol, '', atom_highlights, {}, atom_radii, {})
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText().replace('svg:', '').partition('\n')[-1]
    return svg