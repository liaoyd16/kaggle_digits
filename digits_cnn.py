
# digits_cnn.py

import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import training
from training import training, batchsize


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5,5), stride=1, padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5,5), stride=1,padding=0)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        x = x.view(batchsize, 1, 28, 28)
        x = self.maxpool1(F.relu(self.conv1(x)))
        x = self.maxpool2(F.relu(self.conv2(x)))
        x = x.view(batchsize, 1, 400)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

    def forward_pass(self, x, xnum):
        x = x.view(xnum, 1, 28, 28)
        x = self.maxpool1(F.relu(self.conv1(x)))
        x = self.maxpool2(F.relu(self.conv2(x)))
        x = x.view(xnum, 1, 400)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


if __name__=="__main__":

    ''' training entity setup '''
    classifier = Classifier()
    lossF = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(classifier.parameters(), lr=0.005)
    optimizer.zero_grad()
    ''' training entity setup '''

    training(classifier, lossF, optimizer, "alex_best")