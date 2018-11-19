
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



batch_size = 128


transform_train = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.7, 1.0), ratio=(1.0,1.0)),
    transforms.ColorJitter(
            brightness=0.1*torch.randn(1),
            contrast=0.1*torch.randn(1),
            saturation=0.1*torch.randn(1),
            hue=0.1*torch.randn(1)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = torchvision.datasets.CIFAR10(root='./', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root='./', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)





# In[4]:



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

    def forward(self, x):
        #Here is where the network is specified.
        x = F.leaky_relu(self.conv1_ln(self.conv1(x)))

        x = F.leaky_relu(self.conv2_ln(self.conv2(x)))
        x = F.leaky_relu(self.conv3_ln(self.conv3(x)))
        x = F.leaky_relu(self.conv4_ln(self.conv4(x)))
        x = F.leaky_relu(self.conv5_ln(self.conv5(x)))
        x = F.leaky_relu(self.conv6_ln(self.conv6(x)))
        x = F.leaky_relu(self.conv7_ln(self.conv7(x)))
        x = F.leaky_relu(self.conv8_ln(self.conv8(x)))
       
        
      
        x = F.max_pool2d(x,  kernel_size=4,stride=4)

    
        
        x = x.view(x.size(0), -1)
        fc1 = self.fc1(x)
        fc10 = self.fc10(x)
        return fc1, fc10

model =  discriminator()
model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

num_epochs = 100




model.train()



learning_rate = 0.0001

# In[6]:


#Train Model
for epoch in range(num_epochs):
    if(epoch==50):
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate/10.0
    
    if(epoch==75):
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate/100.0
    
    train_accu = []
    
    for images, labels in trainloader:
        data, target = Variable(images).cuda(), Variable(labels).cuda()
        
        #PyTorch "accumulates gradients", so we need to set the stored gradients to zero when there’s a new batch of data.
        optimizer.zero_grad()
        #Forward propagation of the model, i.e. calculate the hidden units and the output.
        _, output = model(data)
        #The objective function is the negative log-likelihood function.
        loss = criterion(output, target)
        #This calculates the gradients (via backpropagation)
        loss.backward()
        
        #The parameters for the model are updated using stochastic gradient descent.
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                if('step' in state and state['step']>=1024):
                    state['step'] = 1000
        optimizer.step()
        #Calculate accuracy on the training set.
        prediction = output.data.max(1)[1] # first column has actual prob.
        accuracy = ( float( prediction.eq(target.data).sum() ) /float(batch_size))*100.0
        train_accu.append(accuracy)
    accuracy_epoch = np.mean(train_accu)
    print(epoch, accuracy_epoch)


torch.save(model,'cifar10.model')


# # Save and load the entire model.
model = torch.load('cifar10.model')     
model.cuda()    
# In[ ]:




# # In[ ]:

#Calculate accuracy of trained model on the Test Set
model.eval()
test_accu = []

with torch.no_grad():

    for images, labels in testloader:
        data, target = Variable(images).cuda(), Variable(labels).cuda()
        _, output = model(data)
        prediction = output.data.max(1)[1] # first column has actual prob.
        accuracy = ( float( prediction.eq(target.data).sum() ) /float(batch_size))*100.0
        test_accu.append(accuracy)
    accuracy_test = np.mean(test_accu)
    print(accuracy_test)







