"""
VoxelMorph配准网络实现
用于CMR-LGE图像配准，生成变形场将LGE心梗标签映射到CMR空间
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """卷积块：Conv + BatchNorm + LeakyReLU"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))


class UNet(nn.Module):
    """U-Net编码器-解码器网络，用于预测变形场"""
    
    def __init__(self, in_channels=2, enc_channels=[16, 32, 64, 128, 256]):
        super(UNet, self).__init__()
        
        # 编码器
        self.encoders = nn.ModuleList()
        prev_channels = in_channels
        for channels in enc_channels:
            self.encoders.append(ConvBlock(prev_channels, channels))
            prev_channels = channels
        
        # 解码器
        self.decoders = nn.ModuleList()
        dec_channels = enc_channels[::-1]
        for i in range(len(dec_channels) - 1):
            # 上采样 + 跳跃连接
            self.decoders.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    ConvBlock(dec_channels[i] + dec_channels[i+1], dec_channels[i+1])
                )
            )
        
        # 最后一层上采样
        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.final_conv = ConvBlock(dec_channels[-1], dec_channels[-1])
    
    def forward(self, x):
        # 编码路径，保存中间特征用于跳跃连接
        encoder_features = []
        for encoder in self.encoders:
            x = encoder(x)
            encoder_features.append(x)
            x = F.max_pool2d(x, 2)
        
        # 解码路径
        encoder_features = encoder_features[::-1][1:]  # 反转并移除最后一层（已经在x中）
        for i, decoder in enumerate(self.decoders):
            x = decoder(x)
            # 跳跃连接
            if i < len(encoder_features):
                x = torch.cat([x, encoder_features[i]], dim=1)
        
        x = self.final_upsample(x)
        x = self.final_conv(x)
        
        return x


class SpatialTransformer(nn.Module):
    """空间变换层，使用变形场对图像进行扭曲"""
    
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
            src: 源图像 (B, C, H, W)
            flow: 变形场 (B, 2, H, W)
        Returns:
            warped: 扭曲后的图像 (B, C, H, W)
        """
        # 新位置 = 原位置 + 位移
        new_locs = self.grid + flow
        
        # 归一化到[-1, 1]
        for i in range(len(self.size)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (self.size[i] - 1) - 0.5)
        
        # 调整维度顺序为(B, H, W, 2)
        new_locs = new_locs.permute(0, 2, 3, 1)
        new_locs = new_locs[..., [1, 0]]  # 交换x和y
        
        # 使用grid_sample进行双线性插值
        warped = F.grid_sample(src, new_locs, align_corners=True, mode='bilinear')
        
        return warped


class VoxelMorph(nn.Module):
    """
    VoxelMorph配准网络
    
    输入: CMR图像和LGE图像的拼接 (B, 2, H, W)
    输出: 变形场 (B, 2, H, W)
    """
    
    def __init__(self, img_size=(256, 256), enc_channels=[16, 32, 64, 128, 256]):
        super(VoxelMorph, self).__init__()
        
        self.img_size = img_size
        
        # U-Net用于预测变形场
        self.unet = UNet(in_channels=2, enc_channels=enc_channels)
        
        # 最后一层卷积，输出2通道的变形场
        self.flow_layer = nn.Conv2d(enc_channels[0], 2, kernel_size=3, padding=1)
        
        # 空间变换层
        self.spatial_transformer = SpatialTransformer(img_size)
    
    def forward(self, moving, fixed):
        """
        Args:
            moving: 移动图像 (CMR) (B, 1, H, W)
            fixed: 固定图像 (LGE) (B, 1, H, W)
        Returns:
            warped_moving: 配准后的移动图像 (B, 1, H, W)
            flow: 变形场 (B, 2, H, W)
        """
        # 拼接两个图像
        x = torch.cat([moving, fixed], dim=1)  # (B, 2, H, W)
        
        # 通过U-Net提取特征
        x = self.unet(x)
        
        # 预测变形场
        flow = self.flow_layer(x)
        
        # 使用变形场扭曲移动图像
        warped_moving = self.spatial_transformer(moving, flow)
        
        return warped_moving, flow
    
    def warp_mask(self, mask, flow):
        """
        使用变形场扭曲掩模
        
        Args:
            mask: 掩模 (B, 1, H, W)
            flow: 变形场 (B, 2, H, W)
        Returns:
            warped_mask: 扭曲后的掩模 (B, 1, H, W)
        """
        return self.spatial_transformer(mask, flow)


class RegistrationLoss(nn.Module):
    """配准损失函数：相似性损失 + 平滑度损失"""
    
    def __init__(self, similarity_weight=1.0, smoothness_weight=0.1):
        super(RegistrationLoss, self).__init__()
        self.similarity_weight = similarity_weight
        self.smoothness_weight = smoothness_weight
    
    def ncc_loss(self, I, J):
        """归一化互相关损失"""
        # 计算均值
        I_mean = torch.mean(I, dim=[2, 3], keepdim=True)
        J_mean = torch.mean(J, dim=[2, 3], keepdim=True)
        
        # 去中心化
        I_centered = I - I_mean
        J_centered = J - J_mean
        
        # 计算互相关
        cross = torch.sum(I_centered * J_centered, dim=[2, 3])
        I_var = torch.sum(I_centered ** 2, dim=[2, 3])
        J_var = torch.sum(J_centered ** 2, dim=[2, 3])
        
        ncc = cross / (torch.sqrt(I_var * J_var) + 1e-5)
        
        # NCC范围是[-1, 1]，我们希望最大化它，所以返回负值作为损失
        return -torch.mean(ncc)
    
    def gradient_loss(self, flow):
        """变形场平滑度损失（梯度的L2范数）"""
        dy = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :])
        dx = torch.abs(flow[:, :, :, 1:] - flow[:, :, :, :-1])
        
        return torch.mean(dx) + torch.mean(dy)
    
    def forward(self, warped, fixed, flow):
        """
        Args:
            warped: 配准后的图像
            fixed: 目标图像
            flow: 变形场
        """
        sim_loss = self.ncc_loss(warped, fixed)
        smooth_loss = self.gradient_loss(flow)
        
        total_loss = self.similarity_weight * sim_loss + self.smoothness_weight * smooth_loss
        
        return total_loss, sim_loss, smooth_loss


if __name__ == "__main__":
    # 测试代码
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建模型
    model = VoxelMorph(img_size=(256, 256)).to(device)
    loss_fn = RegistrationLoss()
    
    # 创建随机输入
    moving = torch.randn(2, 1, 256, 256).to(device)  # CMR
    fixed = torch.randn(2, 1, 256, 256).to(device)   # LGE
    
    # 前向传播
    warped, flow = model(moving, fixed)
    
    # 计算损失
    total_loss, sim_loss, smooth_loss = loss_fn(warped, fixed, flow)
    
    print(f"Input shape: {moving.shape}")
    print(f"Warped shape: {warped.shape}")
    print(f"Flow shape: {flow.shape}")
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Similarity loss: {sim_loss.item():.4f}")
    print(f"Smoothness loss: {smooth_loss.item():.4f}")
