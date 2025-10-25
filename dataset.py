import torch

from torch.utils.data import Dataset


class SampleDataset(Dataset):
    def __init__(self, features: torch.tensor, labels: torch.tensor):
        self.features = features
        self.labels = labels

    def __getitem__(self, index):
        return (self.features[index], self.labels[index])

    def __len__(self):
        return self.labels.shape[0]
