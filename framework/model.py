'''
Description: 模型搭建其实就是继承pytorch官方库中的nn.Module基类，实现自己的网络模型类。
继承基类后需要重新实现两个函数，Init函数和forward函数。
Author: HCQ
Company(School): UCAS
Email: 1756260160@qq.com
Date: 2021-06-05 11:19:36
LastEditTime: 2021-06-10 10:14:57
FilePath: /pointnet-simple/framework/model.py
'''
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import pretty_errors # 


class PointNet(nn.Module):  # 继承自父类nn.Module
  def __init__(self): # 初始化
    super(PointNet, self).__init__() # 
    self.conv1 = nn.Conv1d(3, 64, 1) # conv1 = torch.nn.Conv1d(3, 64, 1)，即输入通道=3，输出通道=64，卷积核大的大小为1,实际上是使用64个3*1的卷积核
    self.conv2 = nn.Conv1d(64, 128, 1)
    self.conv3 = nn.Conv1d(128, 1024, 1)
    self.fc1 = nn.Linear(1024, 512)
    self.fc2 = nn.Linear(512, 256)
    self.fc3 = nn.Linear(256, 40) # 输出40

    self.bn1 = nn.BatchNorm1d(64)
    self.bn2 = nn.BatchNorm1d(128)
    self.bn3 = nn.BatchNorm1d(1024)
    self.bn4 = nn.BatchNorm1d(512)
    self.bn5 = nn.BatchNorm1d(256)

    self.relu = nn.ReLU(inplace=True)
    self.dropout = nn.Dropout(p=0.3)

  # forward函数负责将输入的feature经过一系列网络层的处理后输出。
  def forward(self, x):
    x = self.relu(self.bn1(self.conv1(x))) #  nn.Conv1d(3, 64, 1)→ torch.Size([3, 64, 10000])常用的组合形式如下： CONV/FC -> BN -> ReLu -> Dropout -> CONV/FC
    x = self.relu(self.bn2(self.conv2(x)))# 128
    x = self.relu(self.bn3(self.conv3(x))) # n x 1024→torch.Size([3, 1024, 10000])
    # print(x.shape) # torch.Size([3, 1024, 10000])
    x = torch.max(x, 2, keepdim=True)[0]
    print(x.shape) # torch.Size([3, 1024, 1])
    x = x.view(-1, 1024)
    print(x.shape) # torch.Size([3, 1024])

    x = self.relu(self.bn4(self.dropout(self.fc1(x))))# 512
    x = self.relu(self.bn5(self.dropout(self.fc2(x))))# 256
    x = self.fc3(x) # 40
    
    return x


if __name__ == "__main__":
  net = PointNet()
  sim_data = Variable(torch.rand(3, 3, 10000)) # batch_size = 3 ,三列坐标，10000个点
  out = net(sim_data) # 输入sim_data
  print('gfn', out.size()) # gfn torch.Size([3, 40])