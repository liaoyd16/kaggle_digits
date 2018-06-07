
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import training
from training import training, batchsize

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.conv11 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(3,3), padding=3)
        self.conv12 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(3,3), padding=1)
        self.maxpool21 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.conv22 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(3,3), padding=1)
        self.conv23 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3,3), padding=1)
        self.maxpool31 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.conv32 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), padding=1)
        self.conv33 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3,3), padding=1)
        self.conv34 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3,3), padding=1)
        self.maxpool41 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.conv42 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), padding=1)
        self.conv43 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=1)
        self.conv44 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=1)

        self.fc1 = nn.Linear(512, 10)

        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        x = x.view(batchsize, 1, 28, 28)

        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = self.maxpool21(x)
        x = F.relu(self.conv22(x))
        x = F.relu(self.conv23(x))
        x = self.maxpool31(x)
        x = F.relu(self.conv32(x))
        x = F.relu(self.conv33(x))
        x = F.relu(self.conv34(x))
        x = self.maxpool41(x)
        x = F.relu(self.conv42(x))
        x = F.relu(self.conv43(x))
        x = F.relu(self.conv44(x))

        x = x.view(batchsize, 512)
        x = F.relu(self.fc1(x))

        return x

    def forward_pass(self, x, valid_size):
        x = x.view(valid_size, 1, 28, 28)

        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = self.maxpool21(x)
        x = F.relu(self.conv22(x))
        x = F.relu(self.conv23(x))
        x = self.maxpool31(x)
        x = F.relu(self.conv32(x))
        x = F.relu(self.conv33(x))
        x = F.relu(self.conv34(x))
        x = self.maxpool41(x)
        x = F.relu(self.conv42(x))
        x = F.relu(self.conv43(x))
        x = F.relu(self.conv44(x))

        x = x.view(valid_size, 512)
        x = F.relu(self.fc1(x))

        return x

''' training & testing '''
if __name__=="__main__":

    ''' training entity setup '''
    classifier = VGG16()
    lossF = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(classifier.parameters(), lr=0.005)
    optimizer.zero_grad()
    ''' training entity setup '''

    training(classifier, lossF, optimizer, "vgg_best")