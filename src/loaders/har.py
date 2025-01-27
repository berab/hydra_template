import pickle
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader, Dataset

class HARLoader:
    def __init__(self, batch_size: int, normalize, debug_level: bool):
        self.name = 'har'
        self.debug = debug_level == 2
        trainset = HARSubset("train")
        testset = HARSubset("test")
        validset = HARSubset("val")

        # Select class to keep 
        self.batch_size = batch_size
        self.train = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        self.test = DataLoader(testset, batch_size=batch_size)
        self.valid = DataLoader(validset, batch_size=batch_size)

        num_workers = 0 if self.debug == 2 else 8

        self.in_chan = 1
        self.in_size = (1, 300)
        self.out_dim = 6

class HARSubset(Dataset):
    def __init__(self, fold="train"):
        self.data = pickle.load(open(f"data/har/{fold}_data.summary", "rb"), encoding="latin1")
        self.labels = pickle.load(open(f"data/har/{fold}_labels.summary", "rb"), encoding="latin1")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        y = np.argmax(y, axis=0)
        return x, y
