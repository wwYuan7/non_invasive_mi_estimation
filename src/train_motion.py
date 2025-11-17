"""
训练运动估计模块
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import sys
import argparse
from tqdm import tqdm

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from src.models.motion_pyramid import MotionPyramidNet
from src.data.dataloader import get_dataloader
from src.utils.train_utils import (
    AverageMeter, save_checkpoint, load_checkpoint, 
    set_seed, EarlyStopping, GradientLoss
)


class MotionTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建模型
        self.model = MotionPyramidNet(img_size=(256, 256)).to(self.device)
        
        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
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
        self.photometric_loss = nn.L1Loss()
        self.smoothness_loss = GradientLoss()
        
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
        self.best_loss = float('inf')
        self.start_epoch = 0
        
        # 创建保存目录
        self.save_dir = Path(args.save_dir) / 'motion_estimation'
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def warp_image(self, img, flow):
        """
        使用运动场扭曲图像
        Args:
            img: (B, 1, H, W)
            flow: (B, 2, H, W)
        """
        B, C, H, W = img.shape
        
        # 创建网格
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=self.device),
            torch.arange(W, device=self.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=0).float()  # (2, H, W)
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # (B, 2, H, W)
        
        # 添加运动场
        new_grid = grid + flow
        
        # 归一化到[-1, 1]
        new_grid[:, 0, :, :] = 2.0 * new_grid[:, 0, :, :] / (W - 1) - 1.0
        new_grid[:, 1, :, :] = 2.0 * new_grid[:, 1, :, :] / (H - 1) - 1.0
        
        # 转换为(B, H, W, 2)格式
        new_grid = new_grid.permute(0, 2, 3, 1)
        
        # 采样
        warped = nn.functional.grid_sample(
            img, new_grid, mode='bilinear', padding_mode='border', align_corners=True
        )
        
        return warped
    
    def train_epoch(self, epoch):
        self.model.train()
        losses = AverageMeter()
        photo_losses = AverageMeter()
        smooth_losses = AverageMeter()
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.args.epochs} [Train]')
        
        for batch in pbar:
            # 获取数据
            cine_ed = batch['cine_ed'].to(self.device)  # (B, 1, H, W)
            cine_es = batch['cine_es'].to(self.device)  # (B, 1, H, W)
            
            # 前向传播（MotionPyramidNet接受两个独立的帧作为输入）
            flow, warped_ed, _ = self.model(cine_ed, cine_es)  # flow: (B, 2, H, W), warped_ed: (B, 1, H, W)
            
            # 计算损失
            photo_loss = self.photometric_loss(warped_ed, cine_es)
            smooth_loss = self.smoothness_loss(flow)
            
            loss = photo_loss + self.args.smooth_weight * smooth_loss
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 更新统计
            losses.update(loss.item(), cine_ed.size(0))
            photo_losses.update(photo_loss.item(), cine_ed.size(0))
            smooth_losses.update(smooth_loss.item(), cine_ed.size(0))
            
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'photo': f'{photo_losses.avg:.4f}',
                'smooth': f'{smooth_losses.avg:.4f}'
            })
        
        return losses.avg
    
    def validate(self, epoch):
        self.model.eval()
        losses = AverageMeter()
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch}/{self.args.epochs} [Val]')
            
            for batch in pbar:
                cine_ed = batch['cine_ed'].to(self.device)
                cine_es = batch['cine_es'].to(self.device)
                
                flow, warped_ed, _ = self.model(cine_ed, cine_es)
                photo_loss = self.photometric_loss(warped_ed, cine_es)
                smooth_loss = self.smoothness_loss(flow)
                
                loss = photo_loss + self.args.smooth_weight * smooth_loss
                
                losses.update(loss.item(), cine_ed.size(0))
                
                pbar.set_postfix({'val_loss': f'{losses.avg:.4f}'})
        
        return losses.avg
    
    def train(self):
        print(f"Training Motion Estimation Module")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.start_epoch, self.args.epochs):
            # 训练
            train_loss = self.train_epoch(epoch + 1)
            
            # 验证
            val_loss = self.validate(epoch + 1)
            
            # 更新学习率
            self.scheduler.step()
            
            print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            
            # 保存最佳模型
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'best_loss': self.best_loss,
                    },
                    self.save_dir,
                    'best_motion_model.pth'
                )
            
            # 定期保存检查点
            if (epoch + 1) % self.args.save_freq == 0:
                save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'best_loss': self.best_loss,
                    },
                    self.save_dir,
                    f'motion_epoch_{epoch+1}.pth'
                )
            
            # 早停检查
            if self.early_stopping(-val_loss):  # 使用负值因为我们要最小化loss
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        print(f"Training completed! Best validation loss: {self.best_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Train Motion Estimation Module')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--smooth_weight', type=float, default=0.1, help='Smoothness loss weight')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of data loading workers')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--save_freq', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建训练器并开始训练
    trainer = MotionTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
