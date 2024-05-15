import torch
import torch.nn as nn
import torch.nn.functional as F
from .image import SepConv
class Model1(nn.Module): #new
    def __init__(self, with_head=True ,num_classes: int = 1000) -> None:

        super().__init__()


            
        self.Dconvs=nn.Sequential(
            SepConv(1,6,3,stride=2,padding=1,growth_factor=3 ,depth_count=3),
            SepConv(6,9,3,stride=2,padding=1,growth_factor=3 ,depth_count=3),
            SepConv(9,14,3,stride=2,padding=1,growth_factor=3 ,depth_count=3),
            SepConv(14,21,3,stride=2,padding=1,growth_factor=3 ,depth_count=3),# Avgpool-> 56 
            SepConv(21,32,3,stride=2,padding=1,growth_factor=3 ,depth_count=3),# then Avgpool -> 28 
            SepConv(32,48,3,stride=2,padding=1,growth_factor=3 ,depth_count=3), # then concate avgpool -> 14

            # SepConv(160,320,3, padding=1, growth_factor=3,depth_count=2),
            ) # then concate avgpool -> 1
        




        self.fc=nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(48, num_classes)
        ) if with_head else nn.Identity()
        
        



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x=self.Dconvs(x)
        x=self.fc(x)
        return x


class M5(nn.Module):
    def __init__(self, n_input=1, num_classes=35, stride=16, n_channel=32):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return torch.squeeze(x, 1)


