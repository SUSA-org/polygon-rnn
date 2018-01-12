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


#pool 2:   56 x 56 x 128
#pool 3:   28 x 28 x 256
#conv4_3:  28 x 28 x 512
#conv5_3:  14 x 14 x 512
#indices: 9,16,22,29
class Model(nn.Module):
    def __init__(self,pretrainedModel,indices):
        super(Model, self).__init__()
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

class PolygonRnn(nn.Module):
    def __init__(self,pretrainedModel,indices):
        super(Model, self).__init__()
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
