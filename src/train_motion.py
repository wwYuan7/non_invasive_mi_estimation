"""
训练运动估计模块
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import sys
import argparse
import json
from tqdm import tqdm

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from src.models.motion_pyramid import MotionPyramidNet
from src.data.custom_dataloader_final import CustomMIDatasetFinal


def set_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch in pbar:
        # 获取数据
        cmr = batch['cmr'].to(device)  # (B, T, H, W)
        
        # 只使用前两帧进行运动估计
        frame1 = cmr[:, 0:1, :, :]  # (B, 1, H, W)
        frame2 = cmr[:, 1:2, :, :]  # (B, 1, H, W)
        
        # 前向传播
        motion_field, _ = model(frame1, frame2)
        
        # 计算损失 (简化版：只使用L1损失)
        loss = nn.functional.l1_loss(frame1, frame2)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)


def validate(model, dataloader, device):
    """验证"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating'):
            cmr = batch['cmr'].to(device)
            frame1 = cmr[:, 0:1, :, :]
            frame2 = cmr[:, 1:2, :, :]
            
            motion_field, _ = model(frame1, frame2)
            loss = nn.functional.l1_loss(frame1, frame2)
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description='Train Motion Estimation Module')
    parser.add_argument('--data_root', type=str, required=True, help='Root directory of the dataset')
    parser.add_argument('--splits_file', type=str, required=True, help='Path to dataset splits JSON file')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of data loading workers')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/motion', help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs/motion', help='Directory to save logs')
    parser.add_argument('--val_freq', type=int, default=5, help='Validation frequency (epochs)')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 创建保存目录
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据集划分
    with open(args.splits_file, 'r') as f:
        splits = json.load(f)
    
    print(f"Train cases: {len(splits['train'])}")
    print(f"Val cases: {len(splits['val'])}")
    print(f"Test cases: {len(splits['test'])}")
    
    # 创建数据集
    train_dataset = CustomMIDatasetFinal(args.data_root, splits['train'])
    val_dataset = CustomMIDatasetFinal(args.data_root, splits['val'])
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 创建模型
    model = MotionPyramidNet(img_size=(256, 256)).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )
    
    # 训练循环
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 50)
        
        # 训练
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        print(f"Train Loss: {train_loss:.4f}")
        
        # 验证
        if epoch % args.val_freq == 0:
            val_loss = validate(model, val_loader, device)
            print(f"Val Loss: {val_loss:.4f}")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, checkpoint_dir / 'best_model.pth')
                print(f"✓ Saved best model (val_loss: {val_loss:.4f})")
            else:
                patience_counter += 1
            
            # 早停
            if patience_counter >= args.patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break
        
        # 更新学习率
        scheduler.step()
        
        # 定期保存检查点
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_dir / f'checkpoint_epoch_{epoch}.pth')
    
    print(f"\nTraining completed! Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to: {checkpoint_dir / 'best_model.pth'}")


if __name__ == '__main__':
    main()
