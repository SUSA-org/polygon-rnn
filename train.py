import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torchvision import transforms,models
import numpy as np
import torch.nn as nn
from model import PolygonRNN
from tqdm import tqdm
import time
import pdb

starttime = time.time()

have_cuda = torch.cuda.is_available()
epochs = 5

# Creating Datasets
"""
In the paper there are 3 transforms listed:
    1) Random Flip of Img Crop & Corresponding Label
    2) Expand the bounding box between 10-20% (random)
    3) Random selection of the starting vertex of the polygon annotation

We may have to use the Lambda transforms
"""
transform = transforms.Compose([
    transforms.Rescale(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

image_dir="../leftImg8bit/train"
train_set = torchvision.datasets.ImageFolder(image_dir,transform)
train_set_size = len(train_set)

val_dir = '.../leftImg8bit/val'
val_set = torchvision.datasets.ImageFolder(val_dir,transform)
val_set_size = len(val_set)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, num_workers=1)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=8, shuffle=True, num_workers=1)

# Initializing Model
vgg = models.vgg16(pretrained=True)
vgg = nn.Sequential(*list(vgg.features.children())[:-1])
model = PolygonRNN(vgg)
if have_cuda:
    model.cuda()

if have_cuda:
    model.cuda()

elapsed = time.time() - starttime
print("About to train! Time elapsed: {}".format(elapsed))

# Train Process Pt.1/2
def train(epoch):
    print("Training process has started")
    model.eval()

    for _, data in tqdm(enumerate(train_loader)):
        original_img = data[0].float()
        original_img = Variable(original_img, volatile=True).cuda()
        output = model(original_img)

# Train Process Pt.2/2
for epoch in range(1, epochs + 1):
    print("Epoch: {}".format(epoch))
    train(epoch)
