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
from torch.optim.swa_utils import AveragedModel, SWALR
from utils import AverageMeter, ProgressMeter, accuracy, LabelSmoothingLoss, Cutout
# from pytorchcv.model_provider import get_model as ptcv_get_model
from seresnet import get_seresnet_cifar


SAVEPATH = './weight/'
WEIGHTDECAY = 1e-4
MOMENTUM = 0.9
BATCHSIZE = 128
LR = 0.1
EPOCHS = 300
PRINTFREQ = 100

LABELSMOOTH = False

SWA = True
SWA_LR = 0.02
SWA_START = 200

CUTOUT = True
CUTOUTSIZE = 8

ACTIVATION = 'mish' # 'relu', 'swish'

def main():
    os.makedirs(SAVEPATH, exist_ok=True)
    print('save path:', SAVEPATH)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device:', device)

    print('weight_decay:', WEIGHTDECAY)
    print('momentum:', MOMENTUM)
    print('batch_size:', BATCHSIZE)
    print('lr:', LR)
    print('epoch:', EPOCHS)
    print('Label smoothing:', LABELSMOOTH)
    print('Stochastic Weight Averaging:', SWA)
    if SWA:
        print('Swa lr:', SWA_LR)
        print('Swa start epoch:', SWA_START)
    print('Cutout augmentation:', CUTOUT)
    if CUTOUT:
        print('Cutout size:', CUTOUTSIZE)

    # get model
    # model = ptcv_get_model("seresnet164bn_cifar10", pretrained=False)
    model = get_seresnet_cifar(activation=ACTIVATION)

    # get loss function
    if LABELSMOOTH:
        criterion = LabelSmoothingLoss(classes=10, smoothing=0.1)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
    optimizer = torch.optim.SGD(model.parameters(), lr=LR,
                                momentum=MOMENTUM, weight_decay=WEIGHTDECAY,
                                nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=EPOCHS, eta_min=0)

    model = model.to(device)
    criterion = criterion.to(device)

    # Check number of parameters your model
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {pytorch_total_params}")
    if int(pytorch_total_params) > 2000000:
        print('Your model has the number of parameters more than 2 millions..')
        return

    if SWA:
        # apply swa
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=SWA_LR)
        swa_total_params = sum(p.numel() for p in swa_model.parameters())
        print(f"Swa parameters: {swa_total_params}")
    
    # cinic mean, std
    normalize = transforms.Normalize(mean=[0.47889522, 0.47227842, 0.43047404],
                                     std=[0.24205776, 0.23828046, 0.25874835])

    if CUTOUT:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            Cutout(size=CUTOUTSIZE)
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

    train_dataset = torchvision.datasets.ImageFolder('./train', transform=train_transform)
    train_loader = DataLoader(train_dataset,
                              batch_size=BATCHSIZE, shuffle=True,
                              num_workers=4, pin_memory=True)

    # colab reload
    start_epoch = 0
    if os.path.isfile(os.path.join(SAVEPATH, 'latest_checkpoint.pth')):
        checkpoint = torch.load(os.path.join(SAVEPATH, 'latest_checkpoint.pth'))
        start_epoch = checkpoint['epoch']
        scheduler.load_state_dict(checkpoint['scheduler'])
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if SWA:
            swa_scheduler.load_state_dict(checkpoint['swa_scheduler'])
            swa_model.load_state_dict(checkpoint['swa_model'])
        print(start_epoch, 'load parameter')

    for epoch in range(start_epoch, EPOCHS):
        print("\n----- epoch: {}, lr: {} -----".format(
            epoch, optimizer.param_groups[0]["lr"]))

        # train for one epoch
        start_time = time.time()
        train(train_loader, epoch, model, optimizer, criterion)
        elapsed_time = time.time() - start_time
        print('==> {:.2f} seconds to train this epoch\n'.format(elapsed_time))

        # learning rate scheduling
        if SWA and epoch > SWA_START:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        if SWA:
            checkpoint = { 
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'swa_model': swa_model.state_dict(),
                'swa_scheduler': swa_scheduler.state_dict()}
        else:
            checkpoint = { 
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()}
        torch.save(checkpoint, os.path.join(SAVEPATH, 'latest_checkpoint.pth'))
        if epoch % 10 == 0:
            torch.save(checkpoint, os.path.join(SAVEPATH, '%d_checkpoint.pth' % epoch))


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

    print('=> Acc@1 {top1.avg:.3f}'.format(top1=top1))


if __name__ == "__main__":
    main()