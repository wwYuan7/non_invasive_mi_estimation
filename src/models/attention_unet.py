"""
Attention U-Net实现
用于心梗分割，融合CMR图像和运动特征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionBlock(nn.Module):
    """注意力门控模块"""
    
    def __init__(self, F_g, F_l, F_int):
        """
        Args:
            F_g: 门控信号的通道数
            F_l: 跳跃连接的通道数
            F_int: 中间层通道数
        """
        super(AttentionBlock, self).__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        """
        Args:
            g: 门控信号（来自解码器）
            x: 跳跃连接（来自编码器）
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi


class ConvBlock(nn.Module):
    """双卷积块"""
    
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class UpConv(nn.Module):
    """上采样卷积块"""
    
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.up(x)


class AttentionUNet(nn.Module):
    """
    Attention U-Net用于心梗分割
    
    输入: CMR图像 + 运动场 + 应变图
    输出: 心梗分割掩模
    """
    
    def __init__(self, in_channels=4, out_channels=1, init_features=64):
        """
        Args:
            in_channels: 输入通道数 (CMR + motion_x + motion_y + strain)
            out_channels: 输出通道数 (1 for binary segmentation)
            init_features: 初始特征数
        """
        super(AttentionUNet, self).__init__()
        
        features = init_features
        
        # 编码器
        self.encoder1 = ConvBlock(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder2 = ConvBlock(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder3 = ConvBlock(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder4 = ConvBlock(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 瓶颈层
        self.bottleneck = ConvBlock(features * 8, features * 16)
        
        # 解码器
        self.upconv4 = UpConv(features * 16, features * 8)
        self.att4 = AttentionBlock(F_g=features * 8, F_l=features * 8, F_int=features * 4)
        self.decoder4 = ConvBlock(features * 16, features * 8)
        
        self.upconv3 = UpConv(features * 8, features * 4)
        self.att3 = AttentionBlock(F_g=features * 4, F_l=features * 4, F_int=features * 2)
        self.decoder3 = ConvBlock(features * 8, features * 4)
        
        self.upconv2 = UpConv(features * 4, features * 2)
        self.att2 = AttentionBlock(F_g=features * 2, F_l=features * 2, F_int=features)
        self.decoder2 = ConvBlock(features * 4, features * 2)
        
        self.upconv1 = UpConv(features * 2, features)
        self.att1 = AttentionBlock(F_g=features, F_l=features, F_int=features // 2)
        self.decoder1 = ConvBlock(features * 2, features)
        
        # 输出层
        self.output = nn.Conv2d(features, out_channels, kernel_size=1)
    
    def forward(self, x):
        """
        Args:
            x: 输入张量 (B, in_channels, H, W)
        Returns:
            output: 分割结果 (B, out_channels, H, W)
        """
        # 编码路径
        enc1 = self.encoder1(x)
        
        enc2 = self.pool1(enc1)
        enc2 = self.encoder2(enc2)
        
        enc3 = self.pool2(enc2)
        enc3 = self.encoder3(enc3)
        
        enc4 = self.pool3(enc3)
        enc4 = self.encoder4(enc4)
        
        # 瓶颈
        bottleneck = self.pool4(enc4)
        bottleneck = self.bottleneck(bottleneck)
        
        # 解码路径（带注意力）
        dec4 = self.upconv4(bottleneck)
        enc4 = self.att4(g=dec4, x=enc4)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        enc3 = self.att3(g=dec3, x=enc3)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        enc2 = self.att2(g=dec2, x=enc2)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        enc1 = self.att1(g=dec1, x=enc1)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.decoder1(dec1)
        
        # 输出
        output = self.output(dec1)
        
        return output


class DiceLoss(nn.Module):
    """Dice损失函数"""
    
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        Args:
            pred: 预测 (B, 1, H, W)
            target: 目标 (B, 1, H, W)
        """
        pred = torch.sigmoid(pred)
        
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        
        dice = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        
        return 1 - dice


class FocalLoss(nn.Module):
    """Focal损失函数，处理类别不平衡"""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        """
        Args:
            pred: 预测 (B, 1, H, W)
            target: 目标 (B, 1, H, W)
        """
        pred = torch.sigmoid(pred)
        
        # 计算二元交叉熵
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        
        # 计算focal权重
        p_t = pred * target + (1 - pred) * (1 - target)
        focal_weight = (1 - p_t) ** self.gamma
        
        # 应用alpha平衡
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        
        focal_loss = alpha_t * focal_weight * bce
        
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """组合损失：Dice + Focal"""
    
    def __init__(self, dice_weight=0.5, focal_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
    
    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        
        total = self.dice_weight * dice + self.focal_weight * focal
        
        return total, dice, focal


if __name__ == "__main__":
    # 测试代码
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建模型
    model = AttentionUNet(in_channels=4, out_channels=1).to(device)
    loss_fn = CombinedLoss()
    
    # 创建随机输入
    # 输入包括: CMR图像 + motion_x + motion_y + strain
    x = torch.randn(2, 4, 256, 256).to(device)
    target = torch.randint(0, 2, (2, 1, 256, 256)).float().to(device)
    
    # 前向传播
    output = model(x)
    
    # 计算损失
    total_loss, dice_loss, focal_loss = loss_fn(output, target)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Target shape: {target.shape}")
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Dice loss: {dice_loss.item():.4f}")
    print(f"Focal loss: {focal_loss.item():.4f}")
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
