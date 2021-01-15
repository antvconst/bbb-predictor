import streamlit as st
import numpy as np
from sklearn import metrics

def app():
    st.title('Blood-Brain Barrier Permeability')
    st.header('Model summary')

    st.subheader('Training statistics')

    st.subheader('Performance')
