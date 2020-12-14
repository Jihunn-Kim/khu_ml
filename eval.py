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
from seresnet import get_seresnet_cifar
import pandas as pd


SAVEPATH = './weight/'
WEIGHTDECAY = 1e-4
MOMENTUM = 0.9
BATCHSIZE = 128
LR = 0.1
EPOCHS = 300
PRINTFREQ = 100

LABELSMOOTH = True

SWA = True
SWA_LR = 0.02
SWA_START = 200

CUTOUT = True
CUTOUTSIZE = 8

ACTIVATION = 'mish' # 'relu', 'swish', 'mish'


class TestImageFolder(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        # return image path
        return super(TestImageFolder, self).__getitem__(index), self.imgs[index][0].split('/')[-1]


def main():
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
    print('Activation:', ACTIVATION)

    model = get_seresnet_cifar(activation=ACTIVATION)
    if SWA:
        swa_model = AveragedModel(model)

    normalize = transforms.Normalize(mean=[0.47889522, 0.47227842, 0.43047404],
                                     std=[0.24205776, 0.23828046, 0.25874835])

    if SWA:
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

        train_dataset = torchvision.datasets.ImageFolder(
            '/content/train', transform=train_transform)
        train_loader = DataLoader(train_dataset,
                                batch_size=BATCHSIZE, shuffle=True,
                                num_workers=4, pin_memory=True)
        print('swa data ready')

    if os.path.isfile(os.path.join(SAVEPATH, 'latest_checkpoint.pth')):
        checkpoint = torch.load(os.path.join(SAVEPATH, 'latest_checkpoint.pth'))
        model.load_state_dict(checkpoint['model'])
        model = model.to(device)
        model.eval()
        if SWA:
            print('swa update batch norm')
            swa_model.load_state_dict(checkpoint['swa_model'])
            swa_model = swa_model.to(device)
            torch.optim.swa_utils.update_bn(train_loader, swa_model, device)
            swa_model.eval()
        print('model ready')

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    test_dataset = TestImageFolder('/content/test', transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCHSIZE, num_workers=4, shuffle=False)
    print('test data ready')

    print('Make an evaluation csv file for kaggle submission...')
    Category = []
    Id = []
    with torch.no_grad():
      for data in test_loader:
          (input, _), name = data
          
          input = input.to(device)
          output = swa_model(input)
          output = torch.argmax(output, dim=1)
          Id = Id + list(name)
          Category = Category + output.tolist()

    #Id = list(range(0, 90000))
    samples = {
       'Id': Id,
       'Target': Category 
    }
    df = pd.DataFrame(samples, columns=['Id', 'Target'])

    df.to_csv(os.path.join(SAVEPATH, 'submission.csv'), index=False)
    print('Done!!')


if __name__ == "__main__":
    main()