# ResNet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from training import training, batchsize

class ResBlock(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(ResBlock, self).__init__()

        self.channels_in = channels_in
        self.channels_out = channels_out

        self.conv1 = nn.Conv2d(in_channels=channels_in, out_channels=channels_out, kernel_size=(3,3), padding=1)
        self.conv2 = nn.Conv2d(in_channels=channels_out, out_channels=channels_out, kernel_size=(3,3), padding=1)

        # self.conv1.weight.data.fill_(0.001 * np.random.randn())
        # self.conv1.bias.data.fill_(0.001 * np.random.randn())

        # self.conv2.weight.data.fill_(0.001 * np.random.randn())
        # self.conv2.bias.data.fill_(0.001 * np.random.randn())

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x1 = self.conv2(x1)
        if self.channels_out > self.channels_in:
            x = self.sizematch(self.channels_in, self.channels_out, x)

        return F.relu(x + x1)

    def sizematch(self, channels_in, channels_out, x):
        zeros = torch.tensor( np.zeros( (x.size()[0], channels_out - channels_in, x.size()[2], x.size()[3]) ), dtype=torch.float )
        return torch.cat((x, zeros), dim=1)

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.zeropad = nn.ZeroPad2d(2)
        self.res11 = ResBlock(1, 8)
        self.res12 = ResBlock(8, 8)
        
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.res21 = ResBlock(8, 16)
        self.res22 = ResBlock(16, 16)
        self.res23 = ResBlock(16, 16)
        
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.res31 = ResBlock(16, 32)
        self.res32 = ResBlock(32, 32)
        self.res33 = ResBlock(32, 32)
        
        self.pool3 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.res41 = ResBlock(32, 64)
        self.res42 = ResBlock(64, 64)
        self.res43 = ResBlock(64, 64)
        
        self.pool4 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.res51 = ResBlock(64, 128)
        self.res52 = ResBlock(128, 128)
        self.res53 = ResBlock(128, 128)
        self.res54 = ResBlock(128, 128)

        self.fc1 = nn.Linear(512, 10)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        x = x.view(batchsize, 1, 28, 28)
        x = self.zeropad(x)

        x = self.res11(x)
        x = self.res12(x)

        x = self.pool1(x)
        x = self.res21(x)
        x = self.res22(x)
        x = self.res23(x)

        x = self.pool2(x)
        x = self.res31(x)
        x = self.res32(x)
        x = self.res33(x)

        x = self.pool3(x)
        x = self.res41(x)
        x = self.res42(x)
        x = self.res43(x)

        x = self.pool4(x)
        x = self.res51(x)
        x = self.res52(x)
        x = self.res53(x)
        x = self.res54(x)
        
        x = x.view(batchsize, 1, 512)
        x = self.fc1(x)
        x = self.softmax(x)

        return x

    def forward_pass(self, x, valid_size):
        x = x.view(valid_size, 1, 28, 28)
        x = self.zeropad(x)

        x = self.res11(x)
        x = self.res12(x)

        x = self.pool1(x)
        x = self.res21(x)
        x = self.res22(x)
        x = self.res23(x)

        x = self.pool2(x)
        x = self.res31(x)
        x = self.res32(x)
        x = self.res33(x)

        x = self.pool3(x)
        x = self.res41(x)
        x = self.res42(x)
        x = self.res43(x)

        x = self.pool4(x)
        x = self.res51(x)
        x = self.res52(x)
        x = self.res53(x)
        x = self.res54(x)
        
        x = x.view(valid_size, 1, 512)
        x = self.fc1(x)
        x = self.softmax(x)

        return x

if __name__ == "__main__":
    classifier = ResNet()
    lossF = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(classifier.parameters(), lr=0.005)
    optimizer.zero_grad()

    training(classifier, lossF, optimizer, "resnet_best")
