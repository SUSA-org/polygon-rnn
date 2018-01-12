# Patrick Chao and Noah Gundotra
# 1/11/18
#   MACHINEEEE LEARNINGG

import cv2
import numpy as np
import torch
from torch.optim import SGD
from torchvision import models,transforms, datasets
from torch.autograd import Variable
from skimage import io, transform
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import pdb

#pool 2:   56 x 56 x 128
#pool 3:   28 x 28 x 256
#conv4_3:  28 x 28 x 512
#conv5_3:  14 x 14 x 512
#indices: 9,16,22,29

class TruncatedVGG(nn.Module):
    def __init__(self,pretrainedModel,indices=[9,16,22,29]):
        super(TruncatedVGG, self).__init__()
        self.model = pretrainedModel
        self.indices = indices

    def forward(self, x):
        output=[]
        for i in range(len(self.model)):
            x=self.model[i](x)
            if i in self.indices:
                output.append(x)
            if i == len(self.model)-1 and i not in self.indices:
                output.append(x)
        return output

class PolygonRNN(nn.Module):
    def __init__(self,pretrainedModel):
        super(PolygonRNN, self).__init__()
        self.VGG = TruncatedVGG(pretrainedModel)

        #First VGG block 56 x 56 x 128
        self.mp1 = nn.MaxPool2d(2,2) #28 x 28 x 128
        self.conv1 = nn.Conv2d(128, 128, kernel_size=3,stride=1,padding=1) #28 x 28 x 128

        #Second VGG block 28 x 28 x 256
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3,stride=1,padding=1) #28 x 28 x 128

        #Third VGG block 28 x 28 x 512
        self.conv3 = nn.Conv2d(512, 128, kernel_size=3,stride=1,padding=1) #28 x 28 x 128

        #Fourth VGG block 14 x 14 x 512
        self.conv4 = nn.Conv2d(512, 128, kernel_size=3,stride=1,padding=1) #14 x 14 x 128
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear') # 28 x 28 x 128

        #Fused VGG BLock 28 x 28 x 512
        self.convFused = nn.Conv2d(512, 128, kernel_size=3,stride=1,padding=1) # 28 x 28 x 128
    def forward(self, x):
        VGGOutput = self.VGG.forward(x)

        block1 = VGGOutput[0]
        block1 = self.conv1(self.mp1(block1)) #28 x 28 x 128

        block2 = VGGOutput[1]
        block2 = self.conv2(block2) #28 x 28 x 128

        block3 = VGGOutput[2]
        block3 = self.conv3(block3)

        block4 = VGGOutput[3]
        block4 = self.up4(self.conv4(block4))

        #merged VGG block 28 x 28 x 512
        fused = torch.cat((block1,block2,block3,block4),1) #dimension of channel each is 128
        fused = self.convFused(fused)
        return fused
