import streamlit as st
import numpy as np
import pandas as pd
from backend.utils import BBBPDataset
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Draw

def get_color(t):
    ORANGE = np.array([1., 0.34, 0.2])
    WHITE = np.array([1., 1., 1.])
    return tuple(t*WHITE + (1-t)*ORANGE)

def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    for atom in mol.GetAtoms():
        atom.SetChiralTag(Chem.rdchem.CHI_UNSPECIFIED)
    return mol

def draw_molecule(mol, W=None):
    if W is not None:
        num_atoms = len(W)
        highlights = (W - W.min()) / (W.max() - W.min())
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

def app():
    st.title('Blood-Brain Barrier Permeability')
    st.header('Inference')

    with st.spinner():
        test_data = BBBPDataset('data', 'test')
        mols = {
            data.name: (data.smiles, data.y.item()) for data in test_data
        }
        drug_names = list(mols.keys())

        option = st.selectbox(
            'Select test molecule:',
            drug_names
        )

        mol = get_mol(mols[option][0])

        st.subheader('Molecule')
        st.image(draw_molecule(mol), use_column_width=True)

        st.subheader('Prediction')
        col1, col2 = st.beta_columns([2, 6])
        with col1:
            st.markdown(f"""
                |                           |               |
                |:-------------------------:|---------------|
                | **Predicted probability** | {0.998989:.3} |
                |    **Predicted class**    | {1}           |
                |      **True class**       | {1}           |
            """)
        with col2:
            num_atoms = len(mol.GetAtoms())
            np.random.seed(1337)
            attn_w = np.random.randn(num_atoms)
            attn_w = np.exp(attn_w / 100) / np.sum(np.exp(attn_w / 100))
            st.image(draw_molecule(mol, attn_w),
                     use_column_width=True,
                     caption='Attention weights')
