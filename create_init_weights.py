"""
创建初始化权重文件
"""

import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.models.motion_pyramid import MotionPyramidNet
from src.models.voxelmorph_simple import VoxelMorph
from src.models.attention_unet import AttentionUNet

def create_initial_weights():
    """创建各模块的初始化权重"""
    
    # 创建checkpoints目录
    checkpoint_dir = Path('checkpoints')
    (checkpoint_dir / 'motion_estimation').mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / 'registration').mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / 'segmentation').mkdir(parents=True, exist_ok=True)
    
    print("Creating initial weights for all modules...")
    
    # 1. 运动估计模块
    print("\n1. Motion Estimation Module")
    motion_model = MotionPyramidNet(img_size=(256, 256))
    motion_checkpoint = {
        'epoch': 0,
        'model_state_dict': motion_model.state_dict(),
        'best_loss': float('inf'),
        'note': 'Initial weights - requires training on real data'
    }
    motion_path = checkpoint_dir / 'motion_estimation' / 'init_motion_model.pth'
    torch.save(motion_checkpoint, motion_path)
    print(f"   Saved to: {motion_path}")
    print(f"   Parameters: {sum(p.numel() for p in motion_model.parameters()):,}")
    
    # 2. 配准模块
    print("\n2. Registration Module")
    registration_model = VoxelMorph(img_size=(256, 256))
    registration_checkpoint = {
        'epoch': 0,
        'model_state_dict': registration_model.state_dict(),
        'best_dice': 0.0,
        'note': 'Initial weights - requires training on real data'
    }
    registration_path = checkpoint_dir / 'registration' / 'init_registration_model.pth'
    torch.save(registration_checkpoint, registration_path)
    print(f"   Saved to: {registration_path}")
    print(f"   Parameters: {sum(p.numel() for p in registration_model.parameters()):,}")
    
    # 3. 分割模块
    print("\n3. Segmentation Module")
    segmentation_model = AttentionUNet(in_channels=4, out_channels=1)
    segmentation_checkpoint = {
        'epoch': 0,
        'model_state_dict': segmentation_model.state_dict(),
        'best_dice': 0.0,
        'note': 'Initial weights - requires training on real data'
    }
    segmentation_path = checkpoint_dir / 'segmentation' / 'init_segmentation_model.pth'
    torch.save(segmentation_checkpoint, segmentation_path)
    print(f"   Saved to: {segmentation_path}")
    print(f"   Parameters: {sum(p.numel() for p in segmentation_model.parameters()):,}")
    
    print("\n" + "="*60)
    print("Initial weights created successfully!")
    print("="*60)
    print("\nNote: These are randomly initialized weights.")
    print("For best performance, train the models on real CMR/LGE data.")
    print("\nTraining commands:")
    print("  1. python src/train_motion.py --epochs 100")
    print("  2. python src/train_registration.py --epochs 100")
    print("  3. python src/train_segmentation.py --epochs 100 --motion_checkpoint checkpoints/motion_estimation/best_motion_model.pth")

if __name__ == '__main__':
    create_initial_weights()
