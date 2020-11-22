from __future__ import print_function, division
import torch
import numpy as np
import torchvision
from torchvision import transforms
import pandas as pd
from torch.utils.data import DataLoader
from resnet import ResNet
from densenet import DenseNet


class TestImageFolder(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        # return image path
        return super(TestImageFolder, self).__getitem__(index), self.imgs[index][0].split('/')[-1]


def eval():
    ########## You can change this part only in this cell ##########
    normalize = transforms.Normalize(mean=[0.47889522, 0.47227842, 0.43047404],
                                     std=[0.24205776, 0.23828046, 0.25874835])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    ################################################################
    SAVEPATH = './weight/'
    BATCHSIZE = 256

    test_dataset = TestImageFolder('./data/test', transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCHSIZE, num_workers=4, shuffle=False)

    # model = ResNet(depth=110)
    model = DenseNet(depth=100, growthRate=12)
    model = model.cuda()
    model.load_state_dict(torch.load(SAVEPATH + 'model_weight.pth'))
    model.eval()

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

    df.to_csv(SAVEPATH + 'submission.csv', index=False)
    print('Done!!')


if __name__ == "__main__":
    eval()