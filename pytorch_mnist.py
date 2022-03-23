# -*- coding: utf-8 -*-
"""
@Time ： 2022/3/17 14:28
@Auth ： Zhe Tang
@File ：pytorch_mnist.py
@IDE ：PyCharm
@Motto：ABC(Always Be Coding)

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy

def get_n_params(model):
    np = 0
    for p in list(model.parameters()):
        np += p.nelement()
    return np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#load the data
input_size = 28 * 28
output_size = 10

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST("./data", train=True,download=True,
                   transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))])),
    batch_size=64,shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST("./data",train=False,transform=transforms.Compose([
        transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))
    ])),
    batch_size=1000,shuffle=True
)
'''
class FC2Layer(nn.Module):
    def __init__(self,input_size,n_hidden,output_size):
        #or nn.Module.__init__(self)
        super(FC2Layer, self).__init__()
        self.input_size = input_size
        self.network = nn.Sequential(
            nn.Linear(input_size,n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden,n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden,output_size),
            nn.LogSoftmax(dim=1)
        )
    def forward(self,x):
        x = x.view(-1,self.input_size)
        return self.network
'''
#define the CNN
class CNN(nn.Module):
    def __init__(self,input_size,n_feature,output_size):
        super(CNN, self).__init__()
        #structure of the CNN
        self.n_feature = n_feature
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=n_feature,kernel_size=5)
        self.conv2 = nn.Conv2d(n_feature,n_feature,kernel_size=5)
        self.fc1 = nn.Linear(n_feature*4*4,50)
        self.fc2 = nn.Linear(50,10)
    # You can repeate the following structure
    def forward(self,x,verbose=False):
        x=self.conv1(x)
        x=F.relu(x)
        x=F.max_pool2d(x,kernel_size=2)
        x=self.conv2(x)
        x=F.relu(x)
        x=F.max_pool2d(x,kernel_size=2)
        x=x.view(-1,self.n_feature*4*4)
        x=self.fc1(x)
        x=F.relu(x)
        x=self.fc2(x)
        x=F.log_softmax(x,dim=1)
        return x;

def train(model):
    model.train()
    for batch_idx, (data,target) in enumerate(train_loader):
        data=data.to(device)
        target=target.to(device)
        optimizer.zero_grad()
        output=model(data)
        loss=F.nll_loss(output,target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train:[{}/{} ({:.0f}%)]\tLoss:{:.6f}'.format(batch_idx * len(data),len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))

def test(model):
    model.eval()
    test_loss=0
    correct=0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += F.nll_loss(output, target, reduction = 'sum').item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))

n_features = 6
model_cnn = CNN(input_size,n_features,output_size)
model_cnn.to(device)
optimizer = optim.SGD(model_cnn.parameters(),lr=0.02,momentum=0.5)
print('Number of parameters: {}'.format(get_n_params(model_cnn)))
train(model_cnn)
test(model_cnn)