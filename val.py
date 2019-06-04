import time

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision
from torch import optim
from torch.autograd import Function, Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms

from dataloader import *
from inception_resnet import *
from senet import *
from resnet import *


def val():
    print("testing")
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop(500),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    val_data = CassavaDataset(mode="val", transform=val_transform)
    val_loader = DataLoader(val_data, batch_size=2, shuffle=False)

    model = torch.load("densenet121.pt")

    model.eval()



    result_list = []
    label_list = []
    running_loss = 0
    loss_count = 0
    for idx, data in enumerate(val_loader):
        img, label = data
        img = Variable(img).cuda()
        label = Variable(label).cuda()
        batch_size = val_loader.batch_size

        output = model(img)

        output = torch.max(F.softmax(output, dim = 1), dim = 1)[1]
        
        
        result_list += list(output.cpu().numpy())
        label_list += list(label.cpu().numpy())
    
    count = 0
    for i in range(len(result_list)):
        if result_list[i] == label_list[i]:
            count += 1
    print(count, len(result_list))



if __name__ == '__main__':
    val()