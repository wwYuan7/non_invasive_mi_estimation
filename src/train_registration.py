"""
训练配准模块
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import sys
import argparse
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from src.models.voxelmorph_simple import VoxelMorph
from src.data.dataloader import get_dataloader
from src.utils.train_utils import (
    AverageMeter, save_checkpoint, set_seed, 
    EarlyStopping, NCCLoss, GradientLoss, dice_coefficient
)


class RegistrationTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建模型
        self.model = VoxelMorph(img_size=(256, 256)).to(self.device)
        
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
        self.similarity_loss = NCCLoss()
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
        self.best_dice = 0.0
        self.start_epoch = 0
        
        # 创建保存目录
        self.save_dir = Path(args.save_dir) / 'registration'
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def warp_image(self, img, flow):
        """使用变形场扭曲图像"""
        B, C, H, W = img.shape
        
        # 创建网格
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=self.device),
            torch.arange(W, device=self.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=0).float()
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)
        
        # 添加变形场
        new_grid = grid + flow
        
        # 归一化到[-1, 1]
        new_grid[:, 0, :, :] = 2.0 * new_grid[:, 0, :, :] / (W - 1) - 1.0
        new_grid[:, 1, :, :] = 2.0 * new_grid[:, 1, :, :] / (H - 1) - 1.0
        
        # 转换格式
        new_grid = new_grid.permute(0, 2, 3, 1)
        
        # 采样
        warped = nn.functional.grid_sample(
            img, new_grid, mode='bilinear', padding_mode='border', align_corners=True
        )
        
        return warped
    
    def train_epoch(self, epoch):
        self.model.train()
        losses = AverageMeter()
        sim_losses = AverageMeter()
        smooth_losses = AverageMeter()
        dices = AverageMeter()
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.args.epochs} [Train]')
        
        for batch in pbar:
            # 固定图像：Cine ED
            fixed = batch['cine_ed'].to(self.device)  # (B, 1, H, W)
            # 浮动图像：LGE
            moving = batch['lge'].to(self.device)  # (B, 1, H, W)
            # 浮动掩模：心梗掩模
            moving_mask = batch['infarct_mask'].unsqueeze(1).to(self.device)  # (B, 1, H, W)
            
            # 前向传播（VoxelMorph接受moving和fixed两个参数）
            warped_moving, flow = self.model(fixed, moving)  # warped_moving: (B, 1, H, W), flow: (B, 2, H, W)
            
            # 扭曲掩模
            warped_mask = self.warp_image(moving_mask, flow)
            
            # 计算损失
            sim_loss = self.similarity_loss(fixed, warped_moving)
            smooth_loss = self.smoothness_loss(flow)
            
            loss = sim_loss + self.args.smooth_weight * smooth_loss
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 计算Dice（用于监控配准质量）
            # 这里我们假设fixed也有心梗掩模（实际中需要手动标注或从LGE传播）
            # 为了演示，我们使用原始心梗掩模
            target_mask = batch['infarct_mask'].unsqueeze(1).to(self.device)
            dice = dice_coefficient(warped_mask, target_mask)
            
            # 更新统计
            losses.update(loss.item(), fixed.size(0))
            sim_losses.update(sim_loss.item(), fixed.size(0))
            smooth_losses.update(smooth_loss.item(), fixed.size(0))
            dices.update(dice.item(), fixed.size(0))
            
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'sim': f'{sim_losses.avg:.4f}',
                'dice': f'{dices.avg:.4f}'
            })
        
        return losses.avg, dices.avg
    
    def validate(self, epoch):
        self.model.eval()
        losses = AverageMeter()
        dices = AverageMeter()
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch}/{self.args.epochs} [Val]')
            
            for batch in pbar:
                fixed = batch['cine_ed'].to(self.device)
                moving = batch['lge'].to(self.device)
                moving_mask = batch['infarct_mask'].unsqueeze(1).to(self.device)
                
                warped_moving, flow = self.model(fixed, moving)
                warped_mask = self.warp_image(moving_mask, flow)
                
                sim_loss = self.similarity_loss(fixed, warped_moving)
                smooth_loss = self.smoothness_loss(flow)
                loss = sim_loss + self.args.smooth_weight * smooth_loss
                
                target_mask = batch['infarct_mask'].unsqueeze(1).to(self.device)
                dice = dice_coefficient(warped_mask, target_mask)
                
                losses.update(loss.item(), fixed.size(0))
                dices.update(dice.item(), fixed.size(0))
                
                pbar.set_postfix({
                    'val_loss': f'{losses.avg:.4f}',
                    'val_dice': f'{dices.avg:.4f}'
                })
        
        return losses.avg, dices.avg
    
    def train(self):
        print(f"Training Registration Module")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
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
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'best_dice': self.best_dice,
                    },
                    self.save_dir,
                    'best_registration_model.pth'
                )
            
            # 定期保存
            if (epoch + 1) % self.args.save_freq == 0:
                save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'best_dice': self.best_dice,
                    },
                    self.save_dir,
                    f'registration_epoch_{epoch+1}.pth'
                )
            
            # 早停
            if self.early_stopping(val_dice):
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        print(f"Training completed! Best validation Dice: {self.best_dice:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Train Registration Module')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--smooth_weight', type=float, default=0.5)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    trainer = RegistrationTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
