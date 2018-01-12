import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torchvision import transforms,models
import numpy as np
import torch.nn as nn
from model import Model
import pdb

have_cuda = torch.cuda.is_available()
epochs = 5

transform = transforms.Compose([
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

image_dir="../val"
train_set = torchvision.datasets.ImageFolder(image_dir,transform)
train_set_size = len(train_set)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)

model = models.vgg16(pretrained=True)
model=nn.Sequential(*list(model.features.children())[:-1])
indices = [9,16,22,29]
model = Model(model,indices)



def train(epoch):
    model.eval()

    i = 1
    for _, data in enumerate(train_loader):
        if i==1:

            original_img = data[0].float()
            scale_img = data[0].float()
            original_img, scale_img = Variable(original_img, volatile=True), Variable(scale_img)

            output = model(original_img)
            i=2
        # use the follow method can't get the right image but I don't know why
        # color_img = torch.from_numpy(color_img.transpose((0, 3, 1, 2)))
        # sprite_img = make_grid(color_img)
        # color_name = './colorimg/'+str(i)+'.jpg'
        # save_image(sprite_img, color_name)
        # i += 1
for epoch in range(1, epochs + 1):
    train(epoch)
