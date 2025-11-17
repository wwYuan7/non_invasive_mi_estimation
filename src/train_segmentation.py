"""
训练分割模块
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import sys
import argparse
from tqdm import tqdm
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from src.models.attention_unet import AttentionUNet
from src.models.motion_pyramid import MotionPyramidNet
from src.data.dataloader import get_dataloader
from src.utils.train_utils import (
    AverageMeter, save_checkpoint, set_seed, 
    EarlyStopping, CombinedSegmentationLoss, dice_coefficient
)


class SegmentationTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建分割模型（4通道输入）
        self.seg_model = AttentionUNet(in_channels=4, out_channels=1).to(self.device)
        
        # 加载预训练的运动估计模型（如果提供）
        if args.motion_checkpoint:
            self.motion_model = MotionPyramidNet(img_size=(256, 256)).to(self.device)
            checkpoint = torch.load(args.motion_checkpoint, map_location=self.device)
            self.motion_model.load_state_dict(checkpoint['model_state_dict'])
            self.motion_model.eval()
            for param in self.motion_model.parameters():
                param.requires_grad = False
            print(f"Loaded motion model from {args.motion_checkpoint}")
        else:
            self.motion_model = None
            print("Warning: No motion model provided, will use random motion features")
        
        # 优化器
        self.optimizer = optim.AdamW(
            self.seg_model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=args.epochs,
            eta_min=1e-6
        )
        
        # 损失函数
        self.criterion = CombinedSegmentationLoss(
            dice_weight=0.5,
            focal_weight=0.5
        )
        
        # 数据加载器
        self.train_loader = get_dataloader(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            mode='train'
        )
        self.val_loader = get_dataloader(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            mode='val'
        )
        
        # 早停
        self.early_stopping = EarlyStopping(patience=args.patience)
        
        # 最佳指标
        self.best_dice = 0.0
        self.start_epoch = 0
        
        # 创建保存目录
        self.save_dir = Path(args.save_dir) / 'segmentation'
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def compute_strain(self, flow):
        """
        从运动场计算应变图
        Args:
            flow: (B, 2, H, W)
        Returns:
            strain: (B, 1, H, W)
        """
        # 计算梯度（位移梯度张量）
        # du/dx
        dudx = flow[:, 0:1, :, 1:] - flow[:, 0:1, :, :-1]
        # du/dy
        dudy = flow[:, 0:1, 1:, :] - flow[:, 0:1, :-1, :]
        # dv/dx
        dvdx = flow[:, 1:2, :, 1:] - flow[:, 1:2, :, :-1]
        # dv/dy
        dvdy = flow[:, 1:2, 1:, :] - flow[:, 1:2, :-1, :]
        
        # Pad to original size
        dudx = nn.functional.pad(dudx, (0, 1, 0, 0))
        dudy = nn.functional.pad(dudy, (0, 0, 0, 1))
        dvdx = nn.functional.pad(dvdx, (0, 1, 0, 0))
        dvdy = nn.functional.pad(dvdy, (0, 0, 0, 1))
        
        # 计算Frobenius范数作为应变的近似
        strain = torch.sqrt(dudx**2 + dudy**2 + dvdx**2 + dvdy**2 + 1e-8)
        
        return strain
    
    def extract_motion_features(self, cine_ed, cine_es):
        """
        提取运动特征
        Returns:
            flow_x, flow_y, strain: 各为 (B, 1, H, W)
        """
        if self.motion_model is not None:
            with torch.no_grad():
                flow, _, _ = self.motion_model(cine_ed, cine_es)  # (B, 2, H, W)
        else:
            # 如果没有运动模型，生成随机特征（仅用于演示）
            B, _, H, W = cine_ed.shape
            flow = torch.randn(B, 2, H, W, device=self.device) * 0.1
        
        flow_x = flow[:, 0:1, :, :]
        flow_y = flow[:, 1:2, :, :]
        strain = self.compute_strain(flow)
        
        return flow_x, flow_y, strain
    
    def train_epoch(self, epoch):
        self.seg_model.train()
        losses = AverageMeter()
        dices = AverageMeter()
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.args.epochs} [Train]')
        
        for batch in pbar:
            cine_ed = batch['cine_ed'].to(self.device)  # (B, 1, H, W)
            cine_es = batch['cine_es'].to(self.device)  # (B, 1, H, W)
            target = batch['infarct_mask'].unsqueeze(1).to(self.device)  # (B, 1, H, W)
            
            # 提取运动特征
            flow_x, flow_y, strain = self.extract_motion_features(cine_ed, cine_es)
            
            # 拼接为4通道输入
            input_features = torch.cat([cine_ed, flow_x, flow_y, strain], dim=1)  # (B, 4, H, W)
            
            # 前向传播
            pred = self.seg_model(input_features)  # (B, 1, H, W)
            
            # 计算损失
            loss = self.criterion(pred, target)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 计算Dice
            dice = dice_coefficient(torch.sigmoid(pred), target)
            
            # 更新统计
            losses.update(loss.item(), cine_ed.size(0))
            dices.update(dice.item(), cine_ed.size(0))
            
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'dice': f'{dices.avg:.4f}'
            })
        
        return losses.avg, dices.avg
    
    def validate(self, epoch):
        self.seg_model.eval()
        losses = AverageMeter()
        dices = AverageMeter()
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch}/{self.args.epochs} [Val]')
            
            for batch in pbar:
                cine_ed = batch['cine_ed'].to(self.device)
                cine_es = batch['cine_es'].to(self.device)
                target = batch['infarct_mask'].unsqueeze(1).to(self.device)
                
                # 提取运动特征
                flow_x, flow_y, strain = self.extract_motion_features(cine_ed, cine_es)
                
                # 拼接输入
                input_features = torch.cat([cine_ed, flow_x, flow_y, strain], dim=1)
                
                # 前向传播
                pred = self.seg_model(input_features)
                
                # 计算损失和Dice
                loss = self.criterion(pred, target)
                dice = dice_coefficient(torch.sigmoid(pred), target)
                
                losses.update(loss.item(), cine_ed.size(0))
                dices.update(dice.item(), cine_ed.size(0))
                
                pbar.set_postfix({
                    'val_loss': f'{losses.avg:.4f}',
                    'val_dice': f'{dices.avg:.4f}'
                })
        
        return losses.avg, dices.avg
    
    def train(self):
        print(f"Training Segmentation Module")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.seg_model.parameters()):,}")
        
        for epoch in range(self.start_epoch, self.args.epochs):
            # 训练
            train_loss, train_dice = self.train_epoch(epoch + 1)
            
            # 验证
            val_loss, val_dice = self.validate(epoch + 1)
            
            # 更新学习率
            self.scheduler.step()
            
            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Dice={train_dice:.4f} | "
                  f"Val Loss={val_loss:.4f}, Dice={val_dice:.4f}")
            
            # 保存最佳模型
            if val_dice > self.best_dice:
                self.best_dice = val_dice
                save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        'model_state_dict': self.seg_model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'best_dice': self.best_dice,
                    },
                    self.save_dir,
                    'best_segmentation_model.pth'
                )
            
            # 定期保存
            if (epoch + 1) % self.args.save_freq == 0:
                save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        'model_state_dict': self.seg_model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'best_dice': self.best_dice,
                    },
                    self.save_dir,
                    f'segmentation_epoch_{epoch+1}.pth'
                )
            
            # 早停
            if self.early_stopping(val_dice):
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        print(f"Training completed! Best validation Dice: {self.best_dice:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Train Segmentation Module')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--motion_checkpoint', type=str, default=None,
                        help='Path to pretrained motion estimation model')
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    trainer = SegmentationTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
