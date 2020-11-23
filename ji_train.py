from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
# import pandas as pd
import argparse
from torch.utils.data import Dataset, DataLoader
from utils import AverageMeter, ProgressMeter, accuracy
from resnet import ResNet
from densenet import DenseNet
from efficientnet import EfficientNet


SAVEPATH = './weight/'
WEIGHTDECAY = 1e-4
MOMENTUM = 0.9
BATCHSIZE = 64
LR = 0.1
EPOCHS = 200
PRINTFREQ = 50
VALID_THRESH = 90


def main():
    os.makedirs(SAVEPATH, exist_ok=True)

    # model = ResNet(depth=20)
    # model = DenseNet(depth=52, growthRate=24)
    # model = DenseNet(depth=28, growthRate=40)
    model = EfficientNet.from_name('efficientnet-b0')
    # inputs = torch.rand(1, 3, 32, 32)
    # outpus = model(inputs)
    # return

    ##### optimizer / learning rate scheduler / criterion #####
    optimizer = torch.optim.SGD(model.parameters(), lr=LR,
                                momentum=MOMENTUM, weight_decay=WEIGHTDECAY,
                                nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=EPOCHS, eta_min=0)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [100, 150],
    #                                                  gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss()
    ###########################################################

    model = model.cuda()
    criterion = criterion.cuda()

    # Check number of parameters your model
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {pytorch_total_params}")
    if int(pytorch_total_params) > 2000000:
        print('Your model has the number of parameters more than 2 millions..')
        return

    normalize = transforms.Normalize(mean=[0.47889522, 0.47227842, 0.43047404],
                                     std=[0.24205776, 0.23828046, 0.25874835])
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    train_dataset = torchvision.datasets.ImageFolder(
        './data/train', transform=train_transform)
    train_loader = DataLoader(train_dataset,
                              batch_size=BATCHSIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    
    val_dataset = torchvision.datasets.ImageFolder('./data/valid', transform=valid_transform)
    val_loader = DataLoader(val_dataset,
                              batch_size=BATCHSIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    last_top1_acc = 0
    best_top1_val = 0
    for epoch in range(EPOCHS):
        print("\n----- epoch: {}, lr: {} -----".format(
            epoch, optimizer.param_groups[0]["lr"]))

        # train for one epoch
        start_time = time.time()
        last_top1_acc = train(train_loader, epoch, model, optimizer, criterion)
        if last_top1_acc > VALID_THRESH:
            last_top1_val = valid(val_loader, epoch, model)
        elapsed_time = time.time() - start_time
        print('==> {:.2f} seconds to train this epoch\n'.format(
            elapsed_time))

        # learning rate scheduling
        scheduler.step()

        # Save model each epoch
        if last_top1_acc > VALID_THRESH and last_top1_val > best_top1_val:
            best_top1_val = last_top1_val
            torch.save(model.state_dict(), SAVEPATH + 'model_weight.pth')

    print(f"Best Top-1 Accuracy: {best_top1_val}")
    print(f"Number of parameters: {pytorch_total_params}")



def train(train_loader, epoch, model, optimizer, criterion):
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(len(train_loader),
                             top1, prefix="Epoch: [{}]".format(epoch))
    # switch to train mode
    model.train()

    for i, (inputs, target) in enumerate(train_loader):
        inputs = inputs.cuda()
        target = target.cuda()

        # compute output
        output = model(inputs)
        loss = criterion(output, target)

        # measure accuracy and record loss, accuracy 
        acc1 = accuracy(output, target, topk=(1, ))
        top1.update(acc1[0].item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % PRINTFREQ == 0:
            progress.print(i)

    print('=> Acc@1 {top1.avg:.3f}'
          .format(top1=top1))
    return top1.avg


def valid(val_loader, epoch, model):
    data_time = AverageMeter('Data', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(len(val_loader), data_time,
                             top1, prefix="Epoch: [{}]".format(epoch))
    # switch to eval mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            input = input.cuda()
            target = target.cuda()

            # compute output
            output = model(input)

            # measure accuracy and record loss, accuracy 
            acc1 = accuracy(output, target, topk=(1, ))
            top1.update(acc1[0].item(), input.size(0))

            # measure elapsed time
            end = time.time()

            if i % PRINTFREQ == 0:
                progress.print(i)

    print('=> Acc@1 {top1.avg:.3f}'
          .format(top1=top1))
    return top1.avg


if __name__ == "__main__":
    main()