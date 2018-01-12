import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
import numpy as np

from ImagesFolder import TrainFolder
from model import Model


have_cuda = torch.cuda.is_available()
epochs = 5

original_transform = transforms.Compose([
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    #transforms.ToTensor()
])

image_dir="..val"

train_set = torchvision.datasets.ImageFolder(image_dir)
train_set_size = len(train_set)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)

model = models.vgg16(pretrained=True)
model=nn.Sequential(*list(model.features.children())[:-1])
indices = [9,16,22,29]
color_model = Model(model,indices)



def train(epoch):
    color_model.train()
    try:
        for batch_idx, (data, classes) in enumerate(train_loader):
            messagefile = open('./message.txt', 'a')
            original_img = data[0].unsqueeze(1).float()
            optimizer.zero_grad()
            output = model(original_img)
            n = np.array(output.size(), dtype='int64')
            # print n.dtype
            print(output)
            lossmsg = 'loss: %.9f\n' % (loss.data[0])
            messagefile.write(lossmsg)
            ems_loss.backward(retain_variables=True)
            optimizer.step()
            if batch_idx % 20 == 0:
                message = 'Train Epoch:%d\tPercent:[%d/%d (%.0f%%)]\tLoss:%.9f\n' % (
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data[0])
                messagefile.write(message)
                torch.save(color_model.state_dict(), modelParams)
                print('Train Epoch: {}[{}/{}({:.0f}%)]\tLoss: {:.9f}\n'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.data[0]))
            messagefile.close()
    except Exception:
        logfile = open('log.txt', 'w')
        logfile.write(traceback.format_exc())
        logfile.close()
    finally:
        torch.save(color_model.state_dict(), modelParams)


for epoch in range(1, epochs + 1):
    train(epoch)
