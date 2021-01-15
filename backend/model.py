import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch_geometric.nn as gnn
from sklearn import metrics
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from .utils import GlobalAttention


class GNN(pl.LightningModule):
    def __init__(self, emb_dim, hidden_size, n_conv, n_linear,
                 dropout, lr, lr_factor, lr_patience):
        super().__init__()
        self.dropout = dropout
        self.lr = lr
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.save_hyperparameters()

        self.atom_enc = AtomEncoder(emb_dim)
        self.conv_layers = nn.ModuleList()
        for _ in range(n_conv):
            self.conv_layers.append(gnn.GCNConv(emb_dim, emb_dim))
        
        attn_gate_nn = nn.Sequential(
            nn.Linear(emb_dim, 32),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(32, 1)
        )
        self.attn_pooling = GlobalAttention(attn_gate_nn)

        self.linear_layers = nn.ModuleList()
        self.linear_layers.append(nn.Linear(emb_dim, hidden_size))
        for _ in range(n_linear-1):
            self.linear_layers.append(nn.Linear(hidden_size, hidden_size))
        self.output_layer = nn.Linear(hidden_size, 2)

    def forward(self, data, return_attn_w=False):
        x = self.atom_enc(data.x)
        edge_index = data.edge_index

        for conv in self.conv_layers:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x, attn_w = self.attn_pooling(x, data.batch)

        for linear in self.linear_layers:
            x = linear(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.output_layer(x)
        log_probs = F.log_softmax(x, dim=1)

        if return_attn_w:
            return log_probs, attn_w
        return log_probs

    def training_step(self, batch, batch_idx):
        y = batch.y.long().squeeze()
        out = self.forward(batch)
        loss = F.nll_loss(out, y)

        self.log_dict({
            'train_loss': loss
        })
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            y_pred = self.forward(batch)
        return y_pred.cpu(), batch.y.long().cpu()
        
    def validation_epoch_end(self, outputs):
        y_pred = torch.cat([out[0] for out in outputs]).numpy()[:, 1]
        y_true = torch.cat([out[1] for out in outputs]).numpy()

        self.log_dict({
            'val_auroc': metrics.roc_auc_score(y_true, y_pred)
        }, prog_bar=True)

    test_step = validation_step

    def test_epoch_end(self, outputs):
        y_pred = torch.cat([out[0] for out in outputs]).numpy()[:, 1]
        y_true = torch.cat([out[1] for out in outputs]).numpy()

        self.log_dict({
            'test_auroc': metrics.roc_auc_score(y_true, y_pred)
        }, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=self.lr_factor, patience=self.lr_patience)
        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler,
            'monitor': 'val_auroc'
        }
