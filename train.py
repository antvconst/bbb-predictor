import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import pytorch_lightning.loggers as pl_loggers
from torch_geometric.data import DataLoader
from backend.utils import BBBPDataset
from backend.model import GNN

import argparse
import yaml

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('config', type=str)
    args = argparser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    bbbp_train = BBBPDataset(config['data_root'], 'train')
    bbbp_val = BBBPDataset(config['data_root'], 'val')
    bbbp_test = BBBPDataset(config['data_root'], 'test')

    train_dataloader = DataLoader(bbbp_train, batch_size=config['batch_size'], shuffle=True, num_workers=12)
    val_dataloader = DataLoader(bbbp_val, batch_size=config['batch_size'], num_workers=12)
    test_dataloader = DataLoader(bbbp_test, batch_size=config['batch_size'], num_workers=12)

    csv_logger = pl_loggers.CSVLogger(config['save_dir'], 'bbbp_predictor')
    early_stopping = EarlyStopping(monitor='val_auroc', patience=config['es_patience'], mode='max')
    trainer = pl.Trainer(gpus=1, precision=16,
                         logger=csv_logger,
                         callbacks=[early_stopping],
                         num_sanity_val_steps=0)
    model = GNN(emb_dim=config['emb_dim'],
                hidden_size=config['hidden_size'],
                n_conv=config['n_conv'],
                n_linear=config['n_linear'],
                dropout=config['dropout'],
                lr=float(config['lr']),
                lr_patience=config['lr_patience'],
                lr_factor=float(config['lr_factor']))
    trainer.fit(model,
                train_dataloader=train_dataloader,
                val_dataloaders=val_dataloader)
    trainer.test(test_dataloaders=test_dataloader)