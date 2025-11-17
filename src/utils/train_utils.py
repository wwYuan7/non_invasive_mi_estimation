"""
训练工具函数
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json


def dice_coefficient(pred, target, smooth=1e-5):
    """
    计算Dice系数
    Args:
        pred: 预测掩模 (B, H, W) or (B, 1, H, W)
        target: 真实掩模 (B, H, W) or (B, 1, H, W)
    """
    if pred.dim() == 4:
        pred = pred.squeeze(1)
    if target.dim() == 4:
        target = target.squeeze(1)
    
    pred = (pred > 0.5).float()
    target = target.float()
    
    intersection = (pred * target).sum(dim=(1, 2))
    union = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean()


class DiceLoss(nn.Module):
    """Dice损失"""
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        if pred.dim() == 4:
            pred = pred.squeeze(1)
        if target.dim() == 4:
            target = target.squeeze(1)
        
        pred = torch.sigmoid(pred)
        
        intersection = (pred * target).sum(dim=(1, 2))
        union = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class FocalLoss(nn.Module):
    """Focal损失"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        if pred.dim() == 4:
            pred = pred.squeeze(1)
        if target.dim() == 4:
            target = target.squeeze(1)
        
        pred = torch.sigmoid(pred)
        
        # 二分类focal loss
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_weight = (1 - pt) ** self.gamma
        
        bce = nn.functional.binary_cross_entropy(pred, target, reduction='none')
        focal_loss = self.alpha * focal_weight * bce
        
        return focal_loss.mean()


class CombinedSegmentationLoss(nn.Module):
    """组合分割损失：Dice + Focal"""
    def __init__(self, dice_weight=0.5, focal_weight=0.5):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
    
    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        return self.dice_weight * dice + self.focal_weight * focal


class NCCLoss(nn.Module):
    """归一化互相关损失（用于配准）"""
    def __init__(self, win_size=9):
        super().__init__()
        self.win_size = win_size
    
    def forward(self, I, J):
        """
        Args:
            I: 固定图像 (B, 1, H, W)
            J: 浮动图像（扭曲后） (B, 1, H, W)
        """
        # 计算局部均值
        pad = self.win_size // 2
        I_mean = nn.functional.avg_pool2d(I, self.win_size, stride=1, padding=pad)
        J_mean = nn.functional.avg_pool2d(J, self.win_size, stride=1, padding=pad)
        
        # 计算局部方差和协方差
        I_var = nn.functional.avg_pool2d((I - I_mean) ** 2, self.win_size, stride=1, padding=pad)
        J_var = nn.functional.avg_pool2d((J - J_mean) ** 2, self.win_size, stride=1, padding=pad)
        I_J_cov = nn.functional.avg_pool2d((I - I_mean) * (J - J_mean), self.win_size, stride=1, padding=pad)
        
        # NCC
        ncc = (I_J_cov ** 2) / (I_var * J_var + 1e-5)
        
        # 返回负NCC作为损失
        return -ncc.mean()


class GradientLoss(nn.Module):
    """梯度平滑损失（用于配准和运动估计）"""
    def __init__(self):
        super().__init__()
    
    def forward(self, flow):
        """
        Args:
            flow: 位移场或运动场 (B, 2, H, W)
        """
        # 计算x和y方向的梯度
        dy = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :])
        dx = torch.abs(flow[:, :, :, 1:] - flow[:, :, :, :-1])
        
        return dx.mean() + dy.mean()


class AverageMeter:
    """计算并存储平均值和当前值"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, save_dir, filename='checkpoint.pth'):
    """保存检查点"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    filepath = save_dir / filename
    torch.save(state, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(filepath, model, optimizer=None):
    """加载检查点"""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    best_metric = checkpoint.get('best_metric', 0)
    
    print(f"Checkpoint loaded from {filepath} (epoch {epoch})")
    return epoch, best_metric


def set_seed(seed=42):
    """设置随机种子以保证可复现性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_metric):
        score = val_metric
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        
        return self.early_stop
