# :brain: Blood-Brain Barrier Permeability Predictor

### Stack:
* Models: PyTorch, PyTorch Geometric, PyTorch Lightning;
* Chemistry: RDKit;
* Web application: Streamlit, Plotly.

### Project structure
* `config.yaml`: model configuration;
* `train.py`: model training script;
* `web.py`: web application root;
* `backend/`: feature extraction, dataset proprocessing, model;
* `frontend/`: web application.

### Usage
* To train the model run `python train.py`;
* To run the web application run `streamlit run web.py`.