import glob
import random

import torch
import torch.nn as nn
from skimage import io
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils

from aug import *


class CassavaDataset(Dataset):
    def __init__(self, mode, image_size = 500, transform=None):
        self.image_size = image_size
        self.set_mode(mode)
        self.transform = transform

    def set_mode(self, mode, fold_index = 0):
        '''
        mode 1: "train" train with 80% labeled data
        mode 2: "full_train" train with 80% labeled data and unlabeld data
        mode 3: "validate" valdiate with 20% labeled data
        mode 4: "test" test with unlabeled data
        '''
        self.mode = mode

        if self.mode == 'train':
            self.image_list = glob.glob("./data/train/*.jpg")

        elif self.mode == 'test':
            self.image_list = glob.glob("./data/test/*.jpg")
            self.image_list = sorted(self.image_list, key = lambda url: int(url[21:-4]))

        elif self.mode == "val":
            self.image_list = glob.glob("./data/val/*.jpg")

    def __getitem__(self, index):
        label = -1
        if self.image_list[index].find("cbb") >= 0:
            label = 0
        elif self.image_list[index].find("cbsd") >= 0:
            label = 1
        elif self.image_list[index].find("cgm") >= 0:
            label = 2
        elif self.image_list[index].find("cmd") >= 0:
            label = 3
        elif self.image_list[index].find("healthy") >= 0:
            label = 4
        image = io.imread(self.image_list[index])

        if self.transform:
            image = self.transform(image)
        # if self.mode == "test":
        #     print(self.image_list[index][21:-4])
        return image, label

    def __len__(self):
        return len(self.image_list)


if __name__ == "__main__":
    train_data = CassavaDataset(mode="train")
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=1)
    for data in train_loader:
        img , label = data
