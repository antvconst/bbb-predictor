import os
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch_geometric.data import InMemoryDataset, Data, download_url
from .mol_feature_extraction import molecule_to_graph


class BBBPDataset(InMemoryDataset):
    url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv'
    split_part_indices = {
        'all': 0,
        'train': 1,
        'val': 2,
        'test': 3
    }

    def __init__(self, root, split_part='all'):
        super().__init__(root)
        split_part_idx = self.split_part_indices[split_part]
        self.data, self.slices = torch.load(self.processed_paths[split_part_idx])

    @property
    def raw_dir(self):
        return os.path.join(self.root, 'raw')

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'processed')

    @property
    def raw_file_names(self):
        return f'BBBP.csv'

    @property
    def processed_file_names(self):
        return ['all.pt', 'train.pt', 'val.pt', 'test.pt']

    def download(self):
        download_url(self.url, self.raw_dir)

    def process(self):
        bbbp_data = pd.read_csv(self.raw_paths[0])

        # Transform dataframe records into graphs
        graph_data = []
        for _, drug in bbbp_data.iterrows():
            data = molecule_to_graph(drug['smiles'], drug['p_np'], drug['name'])
            if data is not None:
                graph_data.append(data)

        # Produce split indices for train, val and test
        indices = np.arange(len(graph_data))
        targets = np.array([data.y for data in graph_data])
        dev_idx, test_idx = train_test_split(indices, test_size=0.15, stratify=targets)
        train_idx, val_idx = train_test_split(dev_idx, test_size=0.1, stratify=targets[dev_idx])
        
        # Construct all, train, val and test datasets
        dataset_all = self.collate(graph_data)
        dataset_train = self.collate([graph_data[idx] for idx in train_idx])
        dataset_val = self.collate([graph_data[idx] for idx in val_idx])
        dataset_test = self.collate([graph_data[idx] for idx in test_idx])

        # Save the constructed datasets to disk
        torch.save(dataset_all, self.processed_paths[self.split_part_indices['all']])
        torch.save(dataset_train, self.processed_paths[self.split_part_indices['train']])
        torch.save(dataset_val, self.processed_paths[self.split_part_indices['val']])
        torch.save(dataset_test, self.processed_paths[self.split_part_indices['test']])