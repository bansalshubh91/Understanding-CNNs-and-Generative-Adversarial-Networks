
# coding: utf-8

# In[ ]:



# # coding: utf-8

# # In[1]:

import numpy as np
import h5py
import time
import copy
from random import randint
import torch 
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.autograd as autograd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

def plot(samples):
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(10, 10)
    gs.update(wspace=0.02, hspace=0.02)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample)
    return fig

batch_size = 128


transform_test = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

testset = torchvision.datasets.CIFAR10(root='./', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
testloader = enumerate(testloader)


#Model architecture
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        # input is 3x32x32
        #These variables store the model parameters.
        self.conv1 = nn.Conv2d(3, 196, kernel_size=3, stride=1, padding=1 )
        self.conv1_ln = nn.LayerNorm([32,32])

        self.conv2 = nn.Conv2d(196, 196, kernel_size=3,stride=2, padding=1  )
        self.conv2_ln = nn.LayerNorm([16,16])
        self.conv3 = nn.Conv2d(196 , 196, kernel_size=3,stride=1, padding=1  )
        self.conv3_ln = nn.LayerNorm([16,16])

        self.conv4 = nn.Conv2d(196 , 196, kernel_size=3,stride=2, padding=1  )
        self.conv4_ln = nn.LayerNorm([8,8])
        self.conv5 = nn.Conv2d(196, 196, kernel_size=3,stride=1, padding=1  )
        self.conv5_ln = nn.LayerNorm([8,8])

        self.conv6 = nn.Conv2d(196, 196, kernel_size=3,stride=1, padding=1  )
        self.conv6_ln = nn.LayerNorm([8,8])
        self.conv7 = nn.Conv2d(196, 196, kernel_size=3,stride=1, padding=1  )
        self.conv7_ln = nn.LayerNorm([8,8])
        
        self.conv8 = nn.Conv2d(196, 196, kernel_size=3,stride=2, padding=1  )
        self.conv8_ln = nn.LayerNorm([4,4])
        
        
    
        self.fc1 = nn.Linear(196, 1)
        self.fc10 = nn.Linear(196, 10)

    def forward(self, x, extract_features=0):
        #Here is where the network is specified.
        x = F.leaky_relu(self.conv1_ln(self.conv1(x)))

        x = F.leaky_relu(self.conv2_ln(self.conv2(x)))
        x = F.leaky_relu(self.conv3_ln(self.conv3(x)))
        x = F.leaky_relu(self.conv4_ln(self.conv4(x)))

        if(extract_features==4):
            x = F.max_pool2d(x,8,8)
            x = x.view(-1, 196)
            return x


        x = F.leaky_relu(self.conv5_ln(self.conv5(x)))
        x = F.leaky_relu(self.conv6_ln(self.conv6(x)))
        x = F.leaky_relu(self.conv7_ln(self.conv7(x)))
        x = F.leaky_relu(self.conv8_ln(self.conv8(x)))
       
        if(extract_features==8):
            x = F.max_pool2d(x,4,4)
            x = x.view(-1, 196)
            return x
        
      
        x = F.max_pool2d(x,  kernel_size=4,stride=4)

    
        
        x = x.view(x.size(0), -1)
        fc1 = self.fc1(x)
        fc10 = self.fc10(x)
        return fc1, fc10


#Model architecture
class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        # input is 3x32x32
        #These variables store the model parameters.
        self.fc1 = nn.Linear(100, 196*4*4)

        self.conv1 = nn.ConvTranspose2d(196, 196, kernel_size=4, stride=2, padding=1 )
        self.conv1_bn = nn.BatchNorm2d(196)

        self.conv2 = nn.Conv2d(196, 196, kernel_size=3,stride=1, padding=1  )
        self.conv2_bn = nn.BatchNorm2d(196)
        self.conv3 = nn.Conv2d(196 , 196, kernel_size=3,stride=1, padding=1  )
        self.conv3_bn = nn.BatchNorm2d(196)

        self.conv4 = nn.Conv2d(196 , 196, kernel_size=3,stride=1, padding=1  )
        self.conv4_bn = nn.BatchNorm2d(196)
        self.conv5 = nn.ConvTranspose2d(196, 196, kernel_size=4,stride=2, padding=1  )
        self.conv5_bn = nn.BatchNorm2d(196)

        self.conv6 = nn.Conv2d(196, 196, kernel_size=3,stride=1, padding=1  )
        self.conv6_bn = nn.BatchNorm2d(196)
        self.conv7 = nn.ConvTranspose2d(196, 196, kernel_size=4,stride=2, padding=1  )
        self.conv7_bn = nn.BatchNorm2d(196)
        
        self.conv8 = nn.Conv2d(196, 3, kernel_size=3,stride=1, padding=1  )
      
        
        
    
        

    def forward(self, x):
        #Here is where the network is specified.
        x = F.relu(self.fc1(x))
        x = x.view(-1, 196, 4, 4)

        x = F.relu(self.conv1_bn(self.conv1(x)))

        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.relu(self.conv4_bn(self.conv4(x)))
        x = F.relu(self.conv5_bn(self.conv5(x)))
        x = F.relu(self.conv6_bn(self.conv6(x)))
        x = F.relu(self.conv7_bn(self.conv7(x)))


        
        x = F.tanh(self.conv8(x))
       
    
        return x



model = torch.load('tempD.model')
model.cuda()
model.eval()

batch_idx, (X_batch, Y_batch) = testloader.__next__()
X_batch = Variable(X_batch,requires_grad=True).cuda()
Y_batch_alternate = (Y_batch + 1)%10
Y_batch_alternate = Variable(Y_batch_alternate).cuda()
Y_batch = Variable(Y_batch).cuda()




X = X_batch.mean(dim=0)
X = X.repeat(batch_size,1,1,1)

Y = torch.arange(batch_size).type(torch.int64)
Y = Variable(Y).cuda()


lr = 0.1
weight_decay = 0.001
for i in range(200):
    output = model(X, extract_features=4 )

    loss = -output[torch.arange(batch_size).type(torch.int64),torch.arange(batch_size).type(torch.int64)]
    gradients = torch.autograd.grad(outputs=loss, inputs=X,
                              grad_outputs=torch.ones(loss.size()).cuda(),
                              create_graph=True, retain_graph=False, only_inputs=True)[0]

    prediction = output.data.max(1)[1] # first column has actual prob.
    accuracy = ( float( prediction.eq(Y.data).sum() ) /float(batch_size))*100.0
    print(i,accuracy,-loss)

    X = X - lr*gradients.data - weight_decay*X.data*torch.abs(X.data)
    X[X>1.0] = 1.0
    X[X<-1.0] = -1.0

## save new images
samples = X.data.cpu().numpy()
samples += 1.0
samples /= 2.0
samples = samples.transpose(0,2,3,1)

fig = plot(samples[0:100])
plt.savefig('visualization/max_features_4.png', bbox_inches='tight')
plt.close(fig)