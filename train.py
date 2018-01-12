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

transform = transforms.Compose([
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

image_dir="../leftImg8bit/train"
train_set = torchvision.datasets.ImageFolder(image_dir,transform)
train_set_size = len(train_set)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, num_workers=1)

vgg = models.vgg16(pretrained=True)
vgg = nn.Sequential(*list(vgg.features.children())[:-1])
model = PolygonRNN(vgg)
if have_cuda:
    model.cuda()

elapsed = time.time() - starttime
print("About to train! Time elapsed: {}".format(elapsed))

def train(epoch):
    print("Training process has started")
    model.eval()

    i = 1
    for _, data in tqdm(enumerate(train_loader)):
        if i==1:

            original_img = data[0].float()
            original_img = Variable(original_img, volatile=True).cuda()

            output = model(original_img)
            i=2
        # use the follow method can't get the right image but I don't know why
        # color_img = torch.from_numpy(color_img.transpose((0, 3, 1, 2)))
        # sprite_img = make_grid(color_img)
        # color_name = './colorimg/'+str(i)+'.jpg'
        # save_image(sprite_img, color_name)
        # i += 1
for epoch in range(1, epochs + 1):
    print("Epoch: {}".format(epoch))
    train(epoch)
