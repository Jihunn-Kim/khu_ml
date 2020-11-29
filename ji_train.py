from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, transforms
import time
import os
import argparse
from torch.utils.data import Dataset, DataLoader
# from torch.optim.swa_utils import AveragedModel, SWALR
from utils import AverageMeter, ProgressMeter, accuracy
from pytorchcv.model_provider import get_model as ptcv_get_model


SAVEPATH = './weight/'
WEIGHTDECAY = 1e-4
MOMENTUM = 0.9
BATCHSIZE = 128
LR = 0.1
EPOCHS = 300
PRINTFREQ = 50


def main():
    os.makedirs(SAVEPATH, exist_ok=True)

    model = ptcv_get_model("seresnet164bn_cifar10", pretrained=False)
    
    ##### optimizer / learning rate scheduler / criterion #####
    optimizer = torch.optim.SGD(model.parameters(), lr=LR,
                                momentum=MOMENTUM, weight_decay=WEIGHTDECAY,
                                nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=EPOCHS, eta_min=0)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [100, 150],
    #                                                  gamma=0.1)

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.005) # must remove scheduler.step()

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

    train_dataset = torchvision.datasets.ImageFolder(
        './train', transform=train_transform)
    train_loader = DataLoader(train_dataset,
                              batch_size=BATCHSIZE, shuffle=True,
                              num_workers=4, pin_memory=True)

    start_epoch = 0
    if os.path.isfile('weight/latest_checkpoint.pth'):
        checkpoint = torch.load('weight/latest_checkpoint.pth')
        start_epoch = checkpoint['epoch']
        scheduler.load_state_dict(checkpoint['scheduler'])
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(start_epoch, 'load parameter')

    last_top1_acc = 0
    best_top1_val = 0
    for epoch in range(start_epoch, EPOCHS):
        print("\n----- epoch: {}, lr: {} -----".format(
            epoch, optimizer.param_groups[0]["lr"]))

        # train for one epoch
        start_time = time.time()
        last_top1_acc = train(train_loader, epoch, model, optimizer, criterion)
        elapsed_time = time.time() - start_time
        print('==> {:.2f} seconds to train this epoch\n'.format(
            elapsed_time))

        # learning rate scheduling
        scheduler.step()

        checkpoint = { 
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()}
        torch.save(checkpoint, 'weight/latest_checkpoint.pth')
        if epoch % 10 == 0:
            torch.save(checkpoint, 'weight/%d_checkpoint.pth' % epoch)

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


if __name__ == "__main__":
    main()