'''
Description: 构造一个数据集类，继承官方的torch.utils.data.Dataset
Author: HCQ
Company(School): UCAS
Email: 1756260160@qq.com
Date: 2021-06-05 11:19:36
LastEditTime: 2021-06-10 10:39:16
FilePath: /pointnet-simple/framework/dataset.py
'''
import torch
import os
import json
from torch.utils.data import Dataset # 官方
from torch.utils.data import DataLoader # 官方
import numpy as np

# 读取pcd（txt）文件
def read_pcd_from_file(file):
    np_pts = np.zeros(0)
    with open(file, 'r') as f:
        pts = []
        for line in f: # '-0.098790,-0.182300,0.163800,0.829000,-0.557200,-0.048180\n'
            one_pt = list(map(float, line[:-1].split(','))) # line[:-1]是把 \n去掉
            pts.append(one_pt[:3]) # 前三列
        np_pts = np.array(pts) # 转成numpy格式
    return np_pts

# 读取文件名，得到文件里面每行的名字
def read_file_names_from_file(file):
  with open(file, 'r') as f:
    files = []
    for line in f:
      files.append(line.split('\n')[0]) # 得到每行的文件名，然后append
  return files


class PointNetDataset(Dataset): # 继承父类Dataset
  def __init__(self, root_dir, train):
    super(PointNetDataset, self).__init__() # 执行父类的构造函数，使得我们能够调用父类的属性。

    self._train = train # 0是训练文件，1是测试文件
    self._classes = []
    # 特征和label
    self._features = []
    self._labels = []

    self.load(root_dir) #数据加载完毕后，所有东西就都在self._features和self._labels ，root_dir： modelnet40_normal_resampled

  def classes(self):
    return self._classes

  def __len__(self): # 返回样本数量：
    return len(self._features)
  
  def __getitem__(self, idx): # 传入一个index，数据增强后，输出对应的特征和标签。
    feature, label = self._features[idx], self._labels[idx]
    # 数据增强(归一化、旋转、高斯噪声)
    #  normalize feature
    center = np.expand_dims(np.mean(feature,axis=0), 0)
    feature = feature-center
    dist = np.max(np.sqrt(np.sum(feature ** 2, axis = 1)),0)
    feature = feature / dist #scale

    # rotation to feature
    theta = np.random.uniform(0, np.pi*2)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]) # 旋转矩阵
    feature[:,[0,2]] = feature[:,[0,2]].dot(rotation_matrix)

    # jitter
    feature += np.random.normal(0, 0.02, size=feature.shape) # np.random.normal(loc=0.0均值, scale=1.0标准差, size=None)
    # 转置 方便后面一维卷积=================================
    feature = torch.Tensor(feature.T) # 转置10000x3变为3x10000============================
    
    # label需要从数字变成一个one hot的向量
    l_lable = [0 for _ in range(len(self._classes))]
    l_lable[self._classes.index(label)] = 1 # index对应位置为1
    label = torch.Tensor(l_lable)

    return feature, label # 返回特征和label
  
  def load(self, root_dir): # 自执行 加载数据
    things = os.listdir(root_dir) # 返回一个由文件名和目录名组成的列表，
    files = []
    for f in things:
      if self._train == 0:
        if f == 'modelnet40_train.txt':
          files = read_file_names_from_file(root_dir + '/' + f) # 得到对应文件名list
      elif self._train == 1:
        if f == 'modelnet40_test.txt':
          files = read_file_names_from_file(root_dir + '/' + f)
      if f == "modelnet40_shape_names.txt":
        self._classes = read_file_names_from_file(root_dir + '/' + f) # 40个类别名
    # for循环结束
    tmp_classes = []
    for file in files: # 遍历 得到对应文件名list
      num = file.split("_")[-1] # file：airplane_0001
      kind = file.split("_" + num)[0] # 标签
      if kind not in tmp_classes:
        tmp_classes.append(kind) # 添加类别
      pcd_file = root_dir + '/' + kind + '/' + file + '.txt' # txt文件全路径
      np_pts = read_pcd_from_file(pcd_file) # 读取txt文件转成np矩阵
      # print(np_pts.shape) # (10000, 3)
      self._features.append(np_pts) # 样本的特征和标签存到成员变量中：
      self._labels.append(kind)
    if self._train == 0: # 训练集
      print("There are " + str(len(self._labels)) + " trian files.") # There are 9843 trian files.
    elif self._train == 1: # 测试集
      print("There are " + str(len(self._labels)) + " test files.")
      

if __name__ == "__main__":
  train_data = PointNetDataset("/home/hcq/data/modelnet40/modelnet40_normal_resampled", train=0) #txt文件   9843
  train_loader = DataLoader(train_data, batch_size=2, shuffle=True) # 官方DataLoader  4922
  cnt = 0
  print(" len(train_loader:", len(train_loader))# 4922
  for pts, label in train_loader: #  batch_size=2  循环2次 Iteration=样本数/batch_size
    print("pts.shape", pts.shape) # torch.Size([2, 3, 10000])
    print("label.shape", label.shape) # torch.Size([2, 40])
    cnt += 1
    if cnt > 3:
      break