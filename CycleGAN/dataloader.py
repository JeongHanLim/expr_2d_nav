from __future__ import print_function, division
import os
import torch
import pickle as pkl
import numpy as np
import scipy.io
import torch.utils.data as data
from torchvision import transforms

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        sample = torch.from_numpy(sample).float()
        return sample


class Transitions(data.Dataset):
    def __init__(self, path, split='train'):
        super().__init__()
        self.path = path
        self.split = split

        with open(path, 'wb') as f:
            self.dataset = pkl.load(f)

    def __getitem__(self, index):
        sample = self.dataset[index]
        return self.transform(sample)

    def __len__(self):
        return len(self.dataset)

    def transform(self, sample):
        composed_transforms = transforms.Compose([ToTensor()])
        return composed_transforms(sample)

    def __str__(self):
        return 'SBDSegmentation(split=' + str(self.split) + ')'