import streamlit as st
from frontend.libs.multiapp import MultiApp
from frontend.apps import inference_app, summary_app

st.set_page_config(page_title='BBBP Predictor', page_icon=':brain:')
app = MultiApp()
app.add_app('Model summary', summary_app.app)
app.add_app('Inference', inference_app.app)
app.run()
