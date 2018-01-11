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

model = models.vgg16(pretrained=True)
model.features
# model=nn.Sequential(*list(model.features.children())[:-1])
print(model)

class Model(nn.Module):
    def __init__(self,pretrainedModel):
        super(Model, self).__init__()
        self.model=pretrainedModel

    def forward(self, x,indices):
        output=[]
        for i in range(len(self.model)):
            x=self.model[i](x)
            if i in indices:
                output += x
        return output
