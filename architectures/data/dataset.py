import os
import pandas as pd
from torch.utils.data import Dataset
import torch
class EmbeddingsDataset(Dataset):
    def __init__(self, embeddings_dir, duplicate=2):
        self.embedding = pd.read_pickle(embeddings_dir)
        self.embeddings = pd.concat([self.embedding]*duplicate, ignore_index=True)
    def __len__(self):
        return self.embeddings.shape[0]

    def __getitem__(self, idx):
        label = torch.from_numpy(self.embeddings.iloc[idx,0])
        return label, label