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
    batch_size = 12

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop((500,500), pad_if_needed = True, padding_mode = "reflect"),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop(500),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
    val_data = CassavaDataset(mode="val", transform=val_transform)
    val_loader = DataLoader(val_data, batch_size=2, shuffle=False)

    train_data = CassavaDataset(mode="train", transform=train_transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    

    if resume:
        learning_rate = 3e-3
        model = torch.load("se_resnext.pt", map_location=lambda storage, loc: storage)
    else:

        # model = inceptionresnetv2(num_classes=5, pretrained='imagenet')
        # model = resnext50_32x4d(pretrained=True, num_classes = 5)
        model = se_resnext101_32x4d(num_classes=5, pretrained='imagenet')
        model = torch.nn.DataParallel(model)


    model.cuda()


    # optimizer = optim.SGD(model.parameters(), lr= learning_rate, momentum=0.9, weight_decay=0.0005)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0005)
    # exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, epoches, eta_min=1e-7)
    # exp_lr_scheduler = CosineWithRestarts(optimizer, T_max = 10, factor = 2)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    # exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode = "min", verbose = True)

    running_loss_list = []
    best_acc = 0
    criterion = nn.CrossEntropyLoss()

    if train:
        print("start training")

        
        for i in range(epoches):

            running_loss = 0.
            running_dice = 0.

            start_time = time.time()
            count = 0
            current_best = 100

            model.train()
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

            acc = count/len(result_list)
            if acc > best_acc:
                torch.save(model,'se_resnext.pt')
                best_acc = acc

            exp_lr_scheduler.step()
                

            running_loss = running_loss / count
            print(np.floor(time.time() - start_time))
            print("[%d/%d] Loss: %.5f" % (i + 1, epoches, running_loss))
            running_loss_list.append(running_loss)


        file = open("se_resnext_loss_record.txt","w")
        for l in running_loss_list:
            file.write("%s\n" % l)
        file.close()
        torch.save(model, 'se_resnext-May30.pt')


    


if __name__ == '__main__':
    train()
    # test()
