from __future__ import print_function, division
import torch
import numpy as np
import torchvision
from torchvision import transforms
import pandas as pd
from torch.utils.data import DataLoader
from pytorchcv.model_provider import get_model as ptcv_get_model
from utils import AverageMeter, ProgressMeter, accuracy


class TestImageFolder(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        # return image path
        return super(TestImageFolder, self).__getitem__(index), self.imgs[index][0].split('/')[-1]


def eval():
    ########## You can change this part only in this cell ##########
    normalize = transforms.Normalize(mean=[0.47889522, 0.47227842, 0.43047404],
                                     std=[0.24205776, 0.23828046, 0.25874835])

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    ################################################################
    SAVEPATH = './weight/'
    BATCHSIZE = 256

    val_dataset = torchvision.datasets.ImageFolder('./valid', transform=valid_transform)
    val_loader = DataLoader(val_dataset,batch_size=BATCHSIZE, shuffle=False, num_workers=4)

    test_dataset = TestImageFolder('./test', transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCHSIZE, num_workers=4, shuffle=False)

    model = ptcv_get_model("seresnet164bn_cifar10", pretrained=False)

    save_epo = 280
    # checkpoint = torch.load('weight/latest_checkpoint.pth')
    checkpoint = torch.load('weight/%d_checkpoint.pth' % save_epo)
    model.load_state_dict(checkpoint['model'])

    model = model.cuda()
    model.eval()

    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(len(val_loader),
                             top1, prefix="Epoch: [{}]".format(0))

    with torch.no_grad():
        for i, (inputs, target) in enumerate(val_loader):

            inputs = inputs.cuda()
            target = target.cuda()

            # compute output
            output = model(inputs)

            # measure accuracy and record loss, accuracy 
            acc1 = accuracy(output, target, topk=(1, ))
            top1.update(acc1[0].item(), inputs.size(0))

            if i % 100 == 0:
                progress.print(i)

    print('=> Acc@1 {top1.avg:.3f}'
          .format(top1=top1))
    print(top1.avg)

    print('Make an evaluation csv file for kaggle submission...')
    Category = []
    Id = []
    with torch.no_grad():
      for data in test_loader:
          (input, _), name = data
          
          input = input.cuda()
          output = model(input)
          output = torch.argmax(output, dim=1)
          Id = Id + list(name)
          Category = Category + output.tolist()

    #Id = list(range(0, 90000))
    samples = {
       'Id': Id,
       'Target': Category 
    }
    df = pd.DataFrame(samples, columns=['Id', 'Target'])

    # df.to_csv(SAVEPATH + 'submission.csv', index=False)
    df.to_csv('weight/submission_%d.csv' % save_epo, index=False)
    print('Done!!')


if __name__ == "__main__":
    eval()