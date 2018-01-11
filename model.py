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

model = models.vgg16(pretrained=True)
model=nn.Sequential(*list(model.features.children())[:-1])
print(model)


class Model(nn.Module):
    def __init__(self,pretrainedModel):
        super(Net, self).__init__()
        self.model = pretrainedModel
        self.conv1 = self.model[0]

        self.conv2 = self.model[2]
        self.mp2 = self.model[4]

        self.conv3 = self.model[5]
        self.conv4 = self.model[7]
        self.mp4 = self.model[9]

        self.conv5 = self.model[10]
        self.conv6 = self.model[12]
        self.conv7= self.model[14]
        self.mp7 = self.model[16]

        self.conv8 = self.model[17]
        self.conv9 = self.model[19]
        self.conv10 = self.model[21]
        self.mp10 = self.model[23]

        self.conv11 = self.model[24]
        self.conv12 = self.model[26]
        self.conv13 = self.model[28]


    def forward(self, x):
        x1=F.relu(self.conv1(x))
        x2=self.mp2(F.relu(self.conv2(x)))
        x3=F.relu(self.conv3(x))

        x = self.conv5(x)
        return x1,x2,
