import torch
import streamlit as st
import numpy as np
import plotly.express as px
from sklearn import metrics

from .common import (select_model, load_model,
                    load_data, load_model_info)


@st.cache(show_spinner=True)
def evaluate_model(model_path):
    model = load_model(model_path)
    dataset = load_data('test')
    y_true = torch.cat([data.y for data in dataset]).numpy()
    y_pred = model.predict_on_dataset(dataset)
    return y_true, y_pred


def app():
    selected_model = select_model()
    hparams, log = load_model_info(selected_model)

    st.title('BBB Permeability Predictor')
    st.header('Model summary')

    st.subheader('Configuration')
    st.write(hparams)

    st.subheader('Training & Validation')
    log = log.fillna(method='backfill') 
    
    loss_df = log.melt(id_vars='step',
                       value_vars=['train_loss', 'val_loss'])
    loss_df['variable'].replace({
        'train_loss': 'Train',
        'val_loss': 'Validation'
    }, inplace=True)
    loss_df.rename(columns={
        'step': 'Step',
        'variable': 'Dataset',
        'value': 'Loss'
    }, inplace=True)
    fig_loss = px.line(loss_df, x='Step', y='Loss', color='Dataset', title='Loss')
    st.plotly_chart(fig_loss)

    auroc_df = log.melt(id_vars='step',
                       value_vars=['train_auroc', 'val_auroc'])
    auroc_df['variable'].replace({
        'train_auroc': 'Train',
        'val_auroc': 'Validation'
    }, inplace=True)
    auroc_df.rename(columns={
        'step': 'Step',
        'variable': 'Dataset',
        'value': 'AUROC'
    }, inplace=True)
    fig_auroc = px.line(auroc_df, x='Step', y='AUROC', color='Dataset', title='AUROC')
    st.plotly_chart(fig_auroc)

    st.subheader('Test Performance')
    y_true, y_probs = evaluate_model(selected_model)

    bin_threshold = st.slider('Binarization threshold',
                               min_value=0., max_value=1.,
                               step=0.01, value=.5)
    y_pred = y_probs >= bin_threshold 
    accuracy = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred)
    auroc = metrics.roc_auc_score(y_true, y_probs)
    ap = metrics.average_precision_score(y_true, y_probs)
    roc_curve = metrics.roc_curve(y_true, y_probs)[:-1]
    pr_curve = metrics.precision_recall_curve(y_true, y_probs)[:-1]

    st.markdown(f"""
        |   Score   | Value          |
        |:---------:|----------------|
        | Accuracy  | {accuracy:.3}  |
        | Precision | {precision:.3} |
        | Recall    | {recall:.3}    |
        | F1        | {f1:.3}        |
        | AUROC     | {auroc:.3}     |
        | AP        | {ap:.3}        |
    """)

    roc_fig = px.area(x=roc_curve[0], y=roc_curve[1],
                      labels={'x': 'TPR', 'y': 'FPR'},
                      title='ROC Curve')
    st.plotly_chart(roc_fig)

    roc_fig = px.area(x=pr_curve[0], y=pr_curve[1],
                      labels={'x': 'Recall', 'y': 'Precision'},
                      title='Precision-Recall Curve')
    st.plotly_chart(roc_fig)
