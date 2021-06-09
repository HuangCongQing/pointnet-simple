from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch
import numpy as np
import time
import os
import datetime

from dataset import PointNetDataset
from model import PointNet

SEED = 13
batch_size = 32
epochs = 100
decay_lr_factor = 0.95
decay_lr_every = 2
lr = 0.01
gpus = [0]
global_step = 0
show_every = 1
val_every = 3
date = datetime.date.today()
save_dir = "../output"


def save_ckp(ckp_dir, model, optimizer, epoch, best_acc, date):
  os.makedirs(ckp_dir, exist_ok=True)
  state = {
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict()
    }
  ckp_path = os.path.join(ckp_dir, f'date_{date}-epoch_{epoch}-maxacc_{best_acc:.3f}.pth')
  torch.save(state, ckp_path)
  torch.save(state, os.path.join(ckp_dir,f'latest.pth'))
  print('model saved to %s' % ckp_path)


def load_ckp(ckp_path, model, optimizer):
  state = torch.load(ckp_path)
  model.load_state_dict(state['state_dict'])
  optimizer.load_state_dict(state['optimizer'])
  print("model load from %s" % ckp_path)

# forward函数输出的预测结果还需要经过softmax操作才能用于计算loss。
def softXEnt(prediction, real_class):
    # return loss here
    logprobs = torch.nn.functional.log_softmax(prediction, dim=1)
    return -(real_class * logprobs).sum() / prediction.shape[0]


def get_eval_acc_results(model, data_loader, device):
    """
    ACC
    """
    seq_id = 0
    model.eval()

    distribution = np.zeros([5])
    confusion_matrix = np.zeros([5, 5])
    pred_ys = []
    gt_ys = []
    with torch.no_grad():
        accs = []
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)

            # TODO: put x into network and get out
            out = model(x)

            # TODO: get pred_y from out
            pred_y = np.argmax(out.cpu().numpy(), axis=1)
            gt = np.argmax(y.cpu().numpy(), axis=1)

            # TODO: calculate acc from pred_y and gt
            acc = np.sum(pred_y == gt) / len(pred_y)
            gt_ys = np.append(gt_ys, gt)
            pred_ys = np.append(pred_ys, pred_y)
            idx = gt

            accs.append(acc)

        return np.mean(accs)


if __name__ == "__main__":
    writer = SummaryWriter('../output/runs/tersorboard')
    torch.manual_seed(SEED)
    device = torch.device(f'cuda:{gpus[0]}' if torch.cuda.is_available() else 'cpu')
    # 加载数据
    print("Loading train dataset...")
    train_data = PointNetDataset("/home/hcq/data/modelnet40/modelnet40_normal_resampled", train=0)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True) # batch_size = 32
    print("Loading valid dataset...")
    val_data = PointNetDataset("/home/hcq/data/modelnet40/modelnet40_normal_resampled/", train=1)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    # 加载模型
    print("Set model and optimizer...")
    model = PointNet().to(device=device)# 模型
    optimizer = optim.Adam(model.parameters(), lr=lr) # 优化器
    scheduler = optim.lr_scheduler.StepLR(
          optimizer, step_size=decay_lr_every, gamma=decay_lr_factor)

    best_acc = 0.0
    model.train() # 训练train
    print("Start trainning...")
    for epoch in range(epochs): # 100
      acc_loss = 0.0
      num_samples = 0
      start_tic = time.time()
      for x, y in train_loader: # 遍历数据(每次train_loader不一样？)===========================================================
        x = x.to(device)
        y = y.to(device)

        # TODO: set grad to zero
        optimizer.zero_grad()

        # TODO: put x into network and get out
        out = model(x) # 模型输入

        loss = softXEnt(out, y) # 计算损失
        
        # TODO: loss backward
        acc = np.sum(np.argmax(out.cpu().detach().numpy(), axis=1) == np.argmax(y.cpu().detach().numpy(), axis=1)) / len(y)

        # TODO: update network's param
        loss.backward()
        
        acc_loss += batch_size * loss.item()
        num_samples += y.shape[0]
        global_step += 1
        acc = np.sum(np.argmax(out.cpu().detach().numpy(), axis=1) == np.argmax(y.cpu().detach().numpy(), axis=1)) / len(y)
        # print('acc: ', acc)
        if (global_step + 1) % show_every == 0:
          # ...log the running loss
          writer.add_scalar('training loss', acc_loss / num_samples, global_step)
          writer.add_scalar('training acc', acc, global_step)
          # print( f"loss at epoch {epoch} step {global_step}:{loss.item():3f}, lr:{optimizer.state_dict()['param_groups'][0]['lr']: .6f}, time:{time.time() - start_tic: 4f}sec")
      # for循环结束
      scheduler.step()
      print(f"loss at epoch {epoch}:{acc_loss / num_samples:.3f}, lr:{optimizer.state_dict()['param_groups'][0]['lr']: .6f}, time:{time.time() - start_tic: 4f}sec")
      
      if (epoch + 1) % val_every == 0:
        
        acc = get_eval_acc_results(model, val_loader, device)
        print("eval at epoch[" + str(epoch) + f"] acc[{acc:3f}]")
        writer.add_scalar('validing acc', acc, global_step)

        if acc > best_acc:
          best_acc = acc
          save_ckp(save_dir, model, optimizer, epoch, best_acc, date)

          example = torch.randn(1, 3, 10000).to(device)
          traced_script_module = torch.jit.trace(model, example)
          traced_script_module.save("../output/traced_model.pt") # 输出权重文件