"""
Motion Pyramid Networks实现
用于从CMR序列估计心脏运动场
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MotionEncoder(nn.Module):
    """运动编码器，提取多尺度特征"""
    
    def __init__(self, in_channels=2):
        super(MotionEncoder, self).__init__()
        
        # 多尺度卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """返回多尺度特征"""
        feat1 = self.conv1(x)  # 1/2
        feat2 = self.conv2(feat1)  # 1/4
        feat3 = self.conv3(feat2)  # 1/8
        feat4 = self.conv4(feat3)  # 1/16
        
        return [feat1, feat2, feat3, feat4]


class MotionDecoder(nn.Module):
    """运动解码器，从多尺度特征预测运动场"""
    
    def __init__(self):
        super(MotionDecoder, self).__init__()
        
        # 粗到细的运动场预测
        self.flow_pred4 = nn.Conv2d(256, 2, 3, padding=1)
        
        self.upconv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.flow_pred3 = nn.Conv2d(128 + 128 + 2, 2, 3, padding=1)
        
        self.upconv2 = nn.Sequential(
            nn.ConvTranspose2d(128 + 2, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.flow_pred2 = nn.Conv2d(64 + 64 + 2, 2, 3, padding=1)
        
        self.upconv1 = nn.Sequential(
            nn.ConvTranspose2d(64 + 2, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.flow_pred1 = nn.Conv2d(32 + 32 + 2, 2, 3, padding=1)
        
        # 最终上采样到原始分辨率
        self.final_upconv = nn.Sequential(
            nn.ConvTranspose2d(32 + 2, 16, 4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.final_flow = nn.Conv2d(16, 2, 3, padding=1)
    
    def forward(self, features):
        """
        Args:
            features: 多尺度特征列表 [feat1, feat2, feat3, feat4]
        Returns:
            flows: 多尺度运动场列表
            final_flow: 最终的运动场
        """
        feat1, feat2, feat3, feat4 = features
        
        # 最粗尺度的运动场
        flow4 = self.flow_pred4(feat4)
        flows = [flow4]
        
        # 逐级上采样并细化
        x = self.upconv3(feat4)
        flow4_up = F.interpolate(flow4, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, feat3, flow4_up], dim=1)
        flow3 = self.flow_pred3(x)
        flows.append(flow3)
        
        x = self.upconv2(torch.cat([x[:, :128, :, :], flow3], dim=1))
        flow3_up = F.interpolate(flow3, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, feat2, flow3_up], dim=1)
        flow2 = self.flow_pred2(x)
        flows.append(flow2)
        
        x = self.upconv1(torch.cat([x[:, :64, :, :], flow2], dim=1))
        flow2_up = F.interpolate(flow2, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, feat1, flow2_up], dim=1)
        flow1 = self.flow_pred1(x)
        flows.append(flow1)
        
        # 最终上采样
        x = self.final_upconv(torch.cat([x[:, :32, :, :], flow1], dim=1))
        final_flow = self.final_flow(x)
        
        return flows, final_flow


class MotionCompensation(nn.Module):
    """运动补偿模块，使用运动场扭曲图像"""
    
    def __init__(self, size):
        super(MotionCompensation, self).__init__()
        self.size = size
        
        # 创建采样网格
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids)
        grid = grid.unsqueeze(0).float()
        self.register_buffer('grid', grid)
    
    def forward(self, img, flow):
        """使用运动场扭曲图像"""
        new_locs = self.grid + flow
        
        for i in range(len(self.size)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (self.size[i] - 1) - 0.5)
        
        new_locs = new_locs.permute(0, 2, 3, 1)
        new_locs = new_locs[..., [1, 0]]
        
        warped = F.grid_sample(img, new_locs, align_corners=True, mode='bilinear')
        
        return warped


class MotionPyramidNet(nn.Module):
    """
    Motion Pyramid Networks
    
    输入: 两帧CMR图像 (ED和ES)
    输出: 运动场和应变图
    """
    
    def __init__(self, img_size=(256, 256)):
        super(MotionPyramidNet, self).__init__()
        
        self.img_size = img_size
        
        # 编码器和解码器
        self.encoder = MotionEncoder(in_channels=2)
        self.decoder = MotionDecoder()
        
        # 运动补偿
        self.motion_compensation = MotionCompensation(img_size)
    
    def forward(self, frame1, frame2):
        """
        Args:
            frame1: 第一帧 (ED) (B, 1, H, W)
            frame2: 第二帧 (ES) (B, 1, H, W)
        Returns:
            final_flow: 运动场 (B, 2, H, W)
            warped_frame1: 扭曲后的第一帧
            multi_scale_flows: 多尺度运动场列表
        """
        # 拼接两帧
        x = torch.cat([frame1, frame2], dim=1)
        
        # 提取多尺度特征
        features = self.encoder(x)
        
        # 预测多尺度运动场
        multi_scale_flows, final_flow = self.decoder(features)
        
        # 运动补偿
        warped_frame1 = self.motion_compensation(frame1, final_flow)
        
        return final_flow, warped_frame1, multi_scale_flows
    
    def compute_strain(self, flow):
        """
        从运动场计算应变
        
        Args:
            flow: 运动场 (B, 2, H, W)
        Returns:
            strain: 应变图 (B, 1, H, W)
        """
        # 计算位移梯度
        du_dx = flow[:, 0:1, :, 1:] - flow[:, 0:1, :, :-1]
        du_dy = flow[:, 0:1, 1:, :] - flow[:, 0:1, :-1, :]
        dv_dx = flow[:, 1:2, :, 1:] - flow[:, 1:2, :, :-1]
        dv_dy = flow[:, 1:2, 1:, :] - flow[:, 1:2, :-1, :]
        
        # 填充以保持尺寸
        du_dx = F.pad(du_dx, (0, 1, 0, 0))
        du_dy = F.pad(du_dy, (0, 0, 0, 1))
        dv_dx = F.pad(dv_dx, (0, 1, 0, 0))
        dv_dy = F.pad(dv_dy, (0, 0, 0, 1))
        
        # 计算应变（简化版，使用位移梯度的范数）
        strain = torch.sqrt(du_dx**2 + du_dy**2 + dv_dx**2 + dv_dy**2)
        
        return strain


class MotionLoss(nn.Module):
    """运动估计损失函数"""
    
    def __init__(self, photo_weight=1.0, smooth_weight=0.1):
        super(MotionLoss, self).__init__()
        self.photo_weight = photo_weight
        self.smooth_weight = smooth_weight
    
    def photometric_loss(self, warped, target):
        """光度一致性损失"""
        return F.l1_loss(warped, target)
    
    def smoothness_loss(self, flow):
        """运动场平滑度损失"""
        dy = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :])
        dx = torch.abs(flow[:, :, :, 1:] - flow[:, :, :, :-1])
        
        return torch.mean(dx) + torch.mean(dy)
    
    def forward(self, warped, target, flow):
        """
        Args:
            warped: 扭曲后的图像
            target: 目标图像
            flow: 运动场
        """
        photo_loss = self.photometric_loss(warped, target)
        smooth_loss = self.smoothness_loss(flow)
        
        total_loss = self.photo_weight * photo_loss + self.smooth_weight * smooth_loss
        
        return total_loss, photo_loss, smooth_loss


if __name__ == "__main__":
    # 测试代码
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建模型
    model = MotionPyramidNet(img_size=(256, 256)).to(device)
    loss_fn = MotionLoss()
    
    # 创建随机输入
    frame1 = torch.randn(2, 1, 256, 256).to(device)  # ED帧
    frame2 = torch.randn(2, 1, 256, 256).to(device)  # ES帧
    
    # 前向传播
    final_flow, warped, multi_flows = model(frame1, frame2)
    
    # 计算应变
    strain = model.compute_strain(final_flow)
    
    # 计算损失
    total_loss, photo_loss, smooth_loss = loss_fn(warped, frame2, final_flow)
    
    print(f"Frame1 shape: {frame1.shape}")
    print(f"Frame2 shape: {frame2.shape}")
    print(f"Final flow shape: {final_flow.shape}")
    print(f"Warped shape: {warped.shape}")
    print(f"Strain shape: {strain.shape}")
    print(f"Number of multi-scale flows: {len(multi_flows)}")
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Photometric loss: {photo_loss.item():.4f}")
    print(f"Smoothness loss: {smooth_loss.item():.4f}")
