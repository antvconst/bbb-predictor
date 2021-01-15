import os
import torch
import streamlit as st
from torch_geometric.data import Batch

from backend.utils import BBBPDataset
from backend import GNN
from .utils import get_mol, draw_molecule


def select_model():
    available_models = os.listdir('saves/bbbp_predictor')
    selection = st.sidebar.selectbox('Model', available_models)
    return os.path.join('saves/bbbp_predictor', selection)

@st.cache(allow_output_mutation=True, show_spinner=True)
def load_data():
    dataset = BBBPDataset('data', 'all')
    mols = {
        data.name: data for data in dataset
    }
    return mols

@st.cache(show_spinner=True)
def load_model(save_dir):
    model_path = os.path.join(save_dir, 'final_model.pt')
    return GNN.load_from_checkpoint(model_path).eval()

@st.cache(show_spinner=True)
def run_inference(model_path, graph):
    model = load_model(model_path)
    with torch.no_grad():
        batch = Batch.from_data_list([graph])
        log_probs, attn_w = model.forward(batch, return_attn_w=True)
    prob = log_probs.exp()[0, 1].item()
    attn_w = attn_w.numpy()
    return prob, attn_w

def app():
    st.title('Blood-Brain Barrier Permeability')
    st.header('Inference')

    model_path = select_model()
    mols = load_data()

    drug_names = list(mols.keys())
    mol_name = st.selectbox(
        'Select molecule:',
        drug_names,
        index=drug_names.index('cocaine')
    )
    graph = mols[mol_name]

    prob, attn_w = run_inference(model_path, graph)
    mol = get_mol(graph.smiles)
    label = graph.y.item()

    st.subheader('Molecule')
    st.image(draw_molecule(mol), use_column_width=True)

    st.subheader('Prediction')
    col1, col2 = st.beta_columns([2, 6])
    with col1:
        st.markdown(f"""
            |                           |               |
            |:-------------------------:|---------------|
            | **Predicted probability** | {prob:.3}     |
            |      **Label**            | {label}       |
        """)
    with col2:
        st.image(draw_molecule(mol, attn_w),
                    use_column_width=True,
                    caption='Attention weights')
