from utils_py import *
from data_setup import *

import torch
import torch.nn as nn


# Model

class CNN(nn.Module):
    def __init__(self,num_classes=2):
        super(CNN, self).__init__()

        # Block 1

        self.conv1=nn.Conv2d(3,32,kernel_size=3,stride=1,padding=1)
        self.bn1=nn.BatchNorm2d(32)
        self.relu1=nn.ReLU()
        self.pooling_1=nn.MaxPool2d(kernel_size=2,stride=2)


        # Block 2

        self.conv2=nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1)
        self.bn2=nn.BatchNorm2d(64)
        self.relu2=nn.ReLU()
        self.pooling_2=nn.MaxPool2d(kernel_size=2,stride=2)


        # Block 3

        self.conv3=nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1)
        self.bn3=nn.BatchNorm2d(128)
        self.relu3=nn.ReLU()
        self.pooling_3=nn.MaxPool2d(kernel_size=2,stride=2)

        # Block 4

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier=nn.Sequential(
            nn.Flatten(),
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256,2)
        )
      
    def forward(self, x):
        
      # Block 1
      x = self.pooling_1(self.relu1(self.bn1(self.conv1(x))))
      

      # Block 2
      x = self.pooling_2(self.relu2(self.bn2(self.conv2(x))))
      

      # Block 3
      x = self.pooling_3(self.relu3(self.bn3(self.conv3(x))))
      
      
      #global pooling
      x = self.global_pool(x)
      
      # Classifier
     
      x = self.classifier(x)

      return x




