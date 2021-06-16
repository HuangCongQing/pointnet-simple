'''
Description: 测试
Author: HCQ
Company(School): UCAS
Email: 1756260160@qq.com
Date: 2021-06-05 11:19:36
LastEditTime: 2021-06-16 15:50:39
FilePath: /pointnet-simple/framework/test.py
'''
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import PointNetDataset
from model import PointNet



SEED = 13
gpus = [0]
batch_size = 1
ckp_path = '../output/latest.pth'

# 加载权重文件
def load_ckp(ckp_path, model):
  state = torch.load(ckp_path)
  model.load_state_dict(state['state_dict'])
  print("model load from %s" % ckp_path)

if __name__ == "__main__":
  torch.manual_seed(SEED) # 固定随机种子
  device = torch.device(f'cuda:{gpus[0]}' if torch.cuda.is_available() else 'cpu') # 使用UPU
  print("Loading test dataset...")
  test_data = PointNetDataset("./dataset/modelnet40_normal_resampled", train=1)
  test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
  model = PointNet().to(device=device)
  if ckp_path:
    load_ckp(ckp_path, model) # 加载权重文件
    model = model.to(device) # .to(device) 使用GPU训练
  
  model.eval() # 在测试模型时都会在前面加上model.eval()

  with torch.no_grad():
    accs = []
    gt_ys = []
    pred_ys = []
    for x, y in test_loader: # Iteration=样本数/batch_size
      x = x.to(device)
      y = y.to(device) # GPU

      # TODO: put x into network and get out
      out =model(x)

      # TODO: get pred_y from out
      pred_y = np.argmax(out.cpu().numpy(), axis=1)#  求每一行最大下标
      gt = np.argmax(y.cpu().numpy(), axis=1) #  求每一行最大下标
      print("pred[" + str(pred_y)+"] gt[" + str(gt) + "]")

      # TODO: calculate acc from pred_y and gt
      acc = np.sum(pred_y == gt)/len(pred_y)
      gt_ys = np.append(gt_ys, gt)
      pred_ys = np.append(pred_ys, pred_y)

      accs.append(acc)

    print("final acc is: " + str(np.mean(accs)))
