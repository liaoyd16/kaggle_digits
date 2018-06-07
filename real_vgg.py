
#coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

import training
from training import training, batchsize

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()

        # 操作函数的定义处
        self.conv11 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), padding=3)
        self.conv12 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3,3), padding=1)

        self.pool1 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.conv21 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), padding=1)
        self.conv22 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3,3), padding=1)

        self.pool2 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.conv31 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), padding=1)
        self.conv32 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=1)

        self.pool3 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.conv41 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), padding=1)
        self.conv42 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), padding=1)
        self.conv43 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), padding=1)

        # self.pool4 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        # self.conv51 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), padding=1)
        # self.conv52 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), padding=1)
        # self.conv53 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), padding=1)        

        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 10)

        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        x = x.view(batchsize, 1, 28, 28)

        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))

        x = self.pool1(x)
        x = F.relu(self.conv21(x))
        x = F.relu(self.conv22(x))

        x = self.pool2(x)
        x = F.relu(self.conv31(x))
        x = F.relu(self.conv32(x))

        x = self.pool3(x)
        x = F.relu(self.conv41(x))
        x = F.relu(self.conv42(x))
        x = F.relu(self.conv43(x))

        # x = self.pool4(x)
        # x = F.relu(self.conv51(x))
        # x = F.relu(self.conv52(x))
        # x = F.relu(self.conv53(x))

        x = x.view(batchsize, 1024)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # x = self.softmax(x)

        return x

    def forward_pass(self, x, valid_size):
        x = x.view(valid_size, 1, 28, 28)

        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))

        x = self.pool1(x)
        x = F.relu(self.conv21(x))
        x = F.relu(self.conv22(x))

        x = self.pool2(x)
        x = F.relu(self.conv31(x))
        x = F.relu(self.conv32(x))

        x = self.pool3(x)
        x = F.relu(self.conv41(x))
        x = F.relu(self.conv42(x))
        x = F.relu(self.conv43(x))

        # x = self.pool4(x)
        # x = F.relu(self.conv51(x))
        # x = F.relu(self.conv52(x))
        # x = F.relu(self.conv53(x))

        x = x.view(valid_size, 1024)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # x = self.softmax(x)

        return x

if __name__=="__main__":
    vgg = VGG()

    lossF = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(vgg.parameters(), lr=0.005)
    optimizer.zero_grad()

    training(vgg, lossF, optimizer, "real_vgg_best")
