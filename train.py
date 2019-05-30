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

def train(resume = False):
    learning_rate = 3e-3
    epoches = 100
    batch_size = 16

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop((500,500), pad_if_needed = True, padding_mode = "reflect"),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])



    train_data = CassavaDataset(mode="train", transform=train_transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    

    if resume:
        learning_rate = 3e-6
        model = torch.load("resnext.pt", map_location=lambda storage, loc: storage)
    else:

        # model = inceptionresnetv2(num_classes=5, pretrained='imagenet')
        model = resnext50_32x4d(pretrained=True, num_classes = 5)
        model = torch.nn.DataParallel(model)


    model.cuda()


    # optimizer = optim.SGD(model.parameters(), lr= learning_rate, momentum=0.9, weight_decay=0.0005)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0005)
    # exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, epoches, eta_min=1e-7)
    # exp_lr_scheduler = CosineWithRestarts(optimizer, T_max = 10, factor = 2)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    # exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode = "min", verbose = True)

    running_loss_list = []

    criterion = nn.CrossEntropyLoss()

    if train:
        print("start training")

        
        for i in range(epoches):

            running_loss = 0.
            running_dice = 0.

            start_time = time.time()
            count = 0
            current_best = 100
            for data in train_loader:
                count +=1

                img , label = data

                img = Variable(img).cuda()
                label = Variable(label).cuda()
                batch_size = train_loader.batch_size

                optimizer.zero_grad()
                output = model(img)

                loss = criterion(output, label)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if loss.item() < current_best:
                    current_best = loss.item()
                    torch.save(model,'resnext.pt')

            exp_lr_scheduler.step(running_loss)
                

            running_loss = running_loss / count
            print(np.floor(time.time() - start_time))
            print("[%d/%d] Loss: %.5f" % (i + 1, epoches, running_loss))
            running_loss_list.append(running_loss)


        file = open("resnext_loss_record.txt","w")
        for l in running_loss_list:
            file.write("%s\n" % l)
        file.close()
        torch.save(model, 'resnext-May30.pt')


    


if __name__ == '__main__':
    train(resume=True)
    # test()
