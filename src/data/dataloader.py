"""
数据加载器和数据增强
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import random


class SyntheticCMRDataset(Dataset):
    """
    合成CMR数据集用于演示训练
    在实际应用中，应替换为真实的CMR数据加载逻辑
    """
    
    def __init__(self, num_samples=100, image_size=(256, 256), num_frames=25, mode='train'):
        """
        Args:
            num_samples: 数据集样本数量
            image_size: 图像尺寸 (H, W)
            num_frames: Cine序列的帧数
            mode: 'train', 'val', or 'test'
        """
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_frames = num_frames
        self.mode = mode
        
    def __len__(self):
        return self.num_samples
    
    def _generate_myocardium_mask(self, size):
        """生成心肌环形掩模"""
        h, w = size
        center_y, center_x = h // 2, w // 2
        
        # 创建坐标网格
        y, x = np.ogrid[:h, :w]
        
        # 计算到中心的距离
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # 创建环形掩模（外半径60，内半径40）
        outer_radius = 60 + np.random.randint(-5, 5)
        inner_radius = 40 + np.random.randint(-5, 5)
        mask = (dist_from_center <= outer_radius) & (dist_from_center >= inner_radius)
        
        return mask.astype(np.float32)
    
    def _generate_infarct_mask(self, myocardium_mask):
        """在心肌掩模内生成心梗区域"""
        h, w = myocardium_mask.shape
        infarct_mask = np.zeros_like(myocardium_mask)
        
        # 随机选择心梗位置（前壁、下壁、侧壁等）
        angle = np.random.uniform(0, 2 * np.pi)
        angular_width = np.random.uniform(np.pi/6, np.pi/3)  # 30-60度
        
        center_y, center_x = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        
        # 计算角度
        angles = np.arctan2(y - center_y, x - center_x)
        
        # 创建扇形区域
        angle_mask = np.abs(angles - angle) < angular_width / 2
        
        # 心梗区域 = 心肌区域 ∩ 扇形区域
        infarct_mask = myocardium_mask * angle_mask
        
        return infarct_mask.astype(np.float32)
    
    def _generate_cine_frame(self, myocardium_mask, infarct_mask, phase):
        """
        生成Cine序列的一帧
        Args:
            phase: 0-1之间，表示心动周期的相位
        """
        h, w = self.image_size
        image = np.zeros((h, w), dtype=np.float32)
        
        # 心肌信号强度
        image[myocardium_mask > 0] = 0.6 + np.random.normal(0, 0.05, np.sum(myocardium_mask > 0))
        
        # 心梗区域在Cine上信号略低（由于运动减弱）
        image[infarct_mask > 0] *= 0.95
        
        # 模拟心脏运动（收缩）
        contraction_factor = 0.8 + 0.2 * np.cos(2 * np.pi * phase)
        
        # 添加血池（左心室腔）
        center_y, center_x = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        blood_pool = dist_from_center <= 35 * contraction_factor
        image[blood_pool] = 0.9
        
        # 添加噪声
        image += np.random.normal(0, 0.02, (h, w))
        
        return np.clip(image, 0, 1)
    
    def _generate_lge_image(self, myocardium_mask, infarct_mask):
        """生成LGE图像"""
        h, w = self.image_size
        image = np.zeros((h, w), dtype=np.float32)
        
        # 正常心肌信号
        image[myocardium_mask > 0] = 0.3 + np.random.normal(0, 0.05, np.sum(myocardium_mask > 0))
        
        # 心梗区域高信号
        image[infarct_mask > 0] = 0.9 + np.random.normal(0, 0.05, np.sum(infarct_mask > 0))
        
        # 血池
        center_y, center_x = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        blood_pool = dist_from_center <= 35
        image[blood_pool] = 0.1
        
        # 添加噪声
        image += np.random.normal(0, 0.02, (h, w))
        
        return np.clip(image, 0, 1)
    
    def __getitem__(self, idx):
        """
        返回一个样本
        Returns:
            dict: {
                'cine_ed': ED帧 (1, H, W)
                'cine_es': ES帧 (1, H, W)
                'cine_sequence': 完整Cine序列 (T, H, W)
                'lge': LGE图像 (1, H, W)
                'myocardium_mask': 心肌掩模 (H, W)
                'infarct_mask': 心梗掩模 (H, W)
            }
        """
        # 生成掩模
        myocardium_mask = self._generate_myocardium_mask(self.image_size)
        infarct_mask = self._generate_infarct_mask(myocardium_mask)
        
        # 生成Cine序列
        cine_sequence = []
        for t in range(self.num_frames):
            phase = t / self.num_frames
            frame = self._generate_cine_frame(myocardium_mask, infarct_mask, phase)
            cine_sequence.append(frame)
        
        cine_sequence = np.stack(cine_sequence, axis=0)  # (T, H, W)
        
        # ED帧（舒张末期，phase=0）和ES帧（收缩末期，phase=0.5）
        ed_idx = 0
        es_idx = self.num_frames // 2
        cine_ed = cine_sequence[ed_idx:ed_idx+1]  # (1, H, W)
        cine_es = cine_sequence[es_idx:es_idx+1]  # (1, H, W)
        
        # 生成LGE图像
        lge = self._generate_lge_image(myocardium_mask, infarct_mask)
        lge = lge[np.newaxis, ...]  # (1, H, W)
        
        # 转换为torch张量
        sample = {
            'cine_ed': torch.from_numpy(cine_ed).float(),
            'cine_es': torch.from_numpy(cine_es).float(),
            'cine_sequence': torch.from_numpy(cine_sequence).float(),
            'lge': torch.from_numpy(lge).float(),
            'myocardium_mask': torch.from_numpy(myocardium_mask).float(),
            'infarct_mask': torch.from_numpy(infarct_mask).float(),
        }
        
        return sample


def get_dataloader(batch_size=4, num_workers=2, mode='train'):
    """
    创建数据加载器
    """
    if mode == 'train':
        dataset = SyntheticCMRDataset(num_samples=100, mode='train')
        shuffle = True
    elif mode == 'val':
        dataset = SyntheticCMRDataset(num_samples=20, mode='val')
        shuffle = False
    else:  # test
        dataset = SyntheticCMRDataset(num_samples=20, mode='test')
        shuffle = False
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


if __name__ == '__main__':
    # 测试数据加载器
    dataloader = get_dataloader(batch_size=2, mode='train')
    
    for batch in dataloader:
        print("Batch shapes:")
        for key, value in batch.items():
            print(f"  {key}: {value.shape}")
        break
