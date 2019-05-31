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


def test():
    print("testing")
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop(500),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    test_data = CassavaDataset(mode="test", transform=test_transform)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    model = torch.load("se_resnext.pt")
    model.cuda()
    # model = torch.nn.DataParallel(model)
    model.eval()



    result_list = []
    running_loss = 0
    loss_count = 0
    for idx, data in enumerate(test_loader):
        img, label = data
        img = Variable(img).cuda()
        label = Variable(label).cuda()
        batch_size = test_loader.batch_size

        output = model(img)


        output = torch.max(F.softmax(output, dim = 1), dim = 1)[1]
        if output == 0:
            result_list.append("cbb")
        elif output == 1:
            result_list.append("cbsd")
        elif output == 2:
            result_list.append("cgm")
        elif output == 3:
            result_list.append("cmd")
        elif output == 4:
            result_list.append("healthy")

    df = pd.read_csv("sample_submission_file.csv")
    for i in range(len(result_list)):
        df["Category"][i] = result_list[i]
    df.to_csv("submission_file_senet.csv", index=False)



if __name__ == '__main__':
    test()