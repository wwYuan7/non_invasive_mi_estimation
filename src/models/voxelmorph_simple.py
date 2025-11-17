"""
简化版VoxelMorph配准网络实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleUNet(nn.Module):
    """简化的U-Net用于变形场预测"""
    
    def __init__(self, in_channels=2):
        super(SimpleUNet, self).__init__()
        
        # 编码器
        self.enc1 = self._make_layer(in_channels, 32)
        self.enc2 = self._make_layer(32, 64)
        self.enc3 = self._make_layer(64, 128)
        self.enc4 = self._make_layer(128, 256)
        
        # 解码器
        self.dec4 = self._make_layer(256 + 128, 128)
        self.dec3 = self._make_layer(128 + 64, 64)
        self.dec2 = self._make_layer(64 + 32, 32)
        self.dec1 = self._make_layer(32, 16)
        
        # 输出层
        self.flow_layer = nn.Conv2d(16, 2, kernel_size=3, padding=1)
        
    def _make_layer(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, x):
        # 编码
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        
        # 解码
        d4 = self.dec4(torch.cat([F.interpolate(e4, scale_factor=2, mode='bilinear', align_corners=True), e3], dim=1))
        d3 = self.dec3(torch.cat([F.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=True), e2], dim=1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=True), e1], dim=1))
        d1 = self.dec1(d2)
        
        # 输出变形场
        flow = self.flow_layer(d1)
        
        return flow


class SpatialTransformer(nn.Module):
    """空间变换层"""
    
    def __init__(self, size):
        super(SpatialTransformer, self).__init__()
        self.size = size
        
        # 创建采样网格
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids)  # (2, H, W)
        grid = grid.unsqueeze(0).float()  # (1, 2, H, W)
        self.register_buffer('grid', grid)
    
    def forward(self, src, flow):
        """
        Args:
            src: (B, C, H, W)
            flow: (B, 2, H, W)
        """
        # 新位置 = 原位置 + 位移
        new_locs = self.grid + flow
        
        # 归一化到[-1, 1]
        shape = flow.shape[2:]
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
        
        # 转换为(B, H, W, 2)格式
        new_locs = new_locs.permute(0, 2, 3, 1)
        new_locs = new_locs[..., [1, 0]]  # 交换x和y
        
        # 采样
        warped = F.grid_sample(src, new_locs, align_corners=True, mode='bilinear')
        
        return warped


class VoxelMorph(nn.Module):
    """简化版VoxelMorph配准网络"""
    
    def __init__(self, img_size=(256, 256)):
        super(VoxelMorph, self).__init__()
        
        self.img_size = img_size
        self.unet = SimpleUNet(in_channels=2)
        self.spatial_transformer = SpatialTransformer(img_size)
    
    def forward(self, moving, fixed):
        """
        Args:
            moving: (B, 1, H, W)
            fixed: (B, 1, H, W)
        Returns:
            warped_moving: (B, 1, H, W)
            flow: (B, 2, H, W)
        """
        # 拼接
        x = torch.cat([moving, fixed], dim=1)
        
        # 预测变形场
        flow = self.unet(x)
        
        # 扭曲
        warped_moving = self.spatial_transformer(moving, flow)
        
        return warped_moving, flow
