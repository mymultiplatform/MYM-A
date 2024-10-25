import torch
from torch.utils.data import Dataset
import numpy as np

class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, data, sequence_length, prediction_length, scaler):
        self.data = torch.FloatTensor(data)
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.scaler = scaler

    def __len__(self):
        return len(self.data) - self.sequence_length - self.prediction_length + 1

    def __getitem__(self, idx):
        # Get sequence
        X = self.data[idx:idx + self.sequence_length]
        # Get target (next prediction_length values)
        y = self.data[idx + self.sequence_length:idx + self.sequence_length + self.prediction_length, 0]
        
        return X, y