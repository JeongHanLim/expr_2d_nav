from __future__ import print_function, division
import torch
import pickle as pkl
import torch.utils.data as data
import numpy as np
from torchvision import transforms

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        sample = torch.from_numpy(sample).float()
        return sample


class Transitions(data.Dataset):
    def __init__(self, path_1, path_2, split='train'):
        super().__init__()
        self.path_1 = path_1
        self.path_2 = path_2
        self.split = split

        with open(path_1, 'rb') as f:
            self.dataset_1 = np.array(pkl.load(f))
        with open(path_2, 'rb') as f:
            self.dataset_2 = np.array(pkl.load(f))

    def __getitem__(self, index):
        sample_1 = self.dataset_1[index]/256
        sample_2 = self.dataset_2[index]/256
        return self.transform(sample_1), self.transform(sample_2)

    def __len__(self):
        return len(self.dataset_1)

    def transform(self, sample):
        composed_transforms = transforms.Compose([ToTensor()])
        return composed_transforms(sample)

    def __str__(self):
        return 'SBDSegmentation(split=' + str(self.split) + ')'