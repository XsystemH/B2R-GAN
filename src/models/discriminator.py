import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_filters=64):
        super().__init__()
        self.conv1 = self.conv_block(in_channels, num_filters)
        self.conv2 = self.conv_block(num_filters, num_filters * 2)
        self.conv3 = self.conv_block(num_filters * 2, num_filters * 4)
        self.conv4 = self.conv_block(num_filters * 4, num_filters * 8)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc1 = nn.Linear(num_filters * 8, num_filters * 4)
        self.fc2 = nn.Linear(num_filters * 4, out_channels)
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        
        x = F.leaky_relu(self.fc1(x), 0.2, inplace=True)
        x = self.fc2(x)
        
        return x