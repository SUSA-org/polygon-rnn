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
class PolyRNN:
   def __init__(self,pretrainedModel):
       super(Net, self).__init__()
       self.model = pretrainedModel
       
   def forward(self, x):
       x1=F.relu(self.conv1(x))
       x2=self.mp2(F.relu(self.conv2(x)))
       x3=F.relu(self.conv3(x))

       x = self.conv5(x)
       return x1,x2,
