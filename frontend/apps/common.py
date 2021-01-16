import os
import streamlit as st
from backend import GNN
from backend.utils import BBBPDataset

def select_model():
    available_models = os.listdir('saves/bbbp_predictor')
    selection = st.sidebar.selectbox('Model', available_models)
    return os.path.join('saves/bbbp_predictor', selection)

@st.cache(show_spinner=True)
def load_model(save_dir):
    model_path = os.path.join(save_dir, 'final_model.pt')
    return GNN.load_from_checkpoint(model_path).eval()

@st.cache(allow_output_mutation=True, show_spinner=True)
def load_data(part='all'):
    dataset = BBBPDataset('data', part)
    return dataset 