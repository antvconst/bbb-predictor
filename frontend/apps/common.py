import os
import yaml
import streamlit as st
import pandas as pd

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

@st.cache
def load_model_info(save_dir):
    log = pd.read_csv(os.path.join(save_dir, 'metrics.csv'))
    with open(os.path.join(save_dir, 'hparams.yaml')) as f:
        hparams = yaml.safe_load(f)
    return hparams, log