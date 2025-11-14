"""
推理脚本：从CMR图像预测心梗分割
"""

import argparse
import torch
import numpy as np
import nibabel as nib
from pathlib import Path

from models.voxelmorph import VoxelMorph
from models.motion_pyramid import MotionPyramidNet
from models.attention_unet import AttentionUNet


class MIEstimator:
    """心梗估计器：整合三个模块进行端到端推理"""
    
    def __init__(self, checkpoint_path, device='cuda'):
        """
        Args:
            checkpoint_path: 模型权重路径
            device: 计算设备
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 加载模型
        print(f"Loading models from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 初始化三个模块
        self.motion_net = MotionPyramidNet(img_size=(256, 256)).to(self.device)
        self.segmentation_net = AttentionUNet(in_channels=4, out_channels=1).to(self.device)
        
        # 加载权重
        if 'motion_net' in checkpoint:
            self.motion_net.load_state_dict(checkpoint['motion_net'])
        if 'segmentation_net' in checkpoint:
            self.segmentation_net.load_state_dict(checkpoint['segmentation_net'])
        
        # 设置为评估模式
        self.motion_net.eval()
        self.segmentation_net.eval()
        
        print("Models loaded successfully!")
    
    def preprocess(self, cmr_path, myocardium_mask_path=None):
        """
        预处理CMR图像
        
        Args:
            cmr_path: CMR图像路径 (NIfTI格式)
            myocardium_mask_path: 心肌掩模路径（可选）
        Returns:
            cmr_tensor: 预处理后的CMR张量
            myocardium_mask: 心肌掩模
            affine: 仿射矩阵（用于保存结果）
        """
        # 加载CMR图像
        cmr_nii = nib.load(cmr_path)
        cmr_data = cmr_nii.get_fdata()
        affine = cmr_nii.affine
        
        # 假设CMR是4D数据 (H, W, slices, T)
        # 提取第一帧（ED）和中间帧（ES）
        if cmr_data.ndim == 4:
            # 取中间切片
            mid_slice = cmr_data.shape[2] // 2
            ed_frame = cmr_data[:, :, mid_slice, 0]  # 第一帧
            es_frame = cmr_data[:, :, mid_slice, cmr_data.shape[3] // 2]  # 中间帧作为ES
        elif cmr_data.ndim == 3:
            # 如果是3D，取前两帧
            ed_frame = cmr_data[:, :, 0]
            es_frame = cmr_data[:, :, 1] if cmr_data.shape[2] > 1 else ed_frame
        else:
            raise ValueError(f"Unexpected CMR data shape: {cmr_data.shape}")
        
        # 归一化到[0, 1]
        ed_frame = self.normalize(ed_frame)
        es_frame = self.normalize(es_frame)
        
        # 调整大小到256x256
        ed_frame = self.resize(ed_frame, (256, 256))
        es_frame = self.resize(es_frame, (256, 256))
        
        # 转换为张量
        ed_tensor = torch.from_numpy(ed_frame).float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        es_tensor = torch.from_numpy(es_frame).float().unsqueeze(0).unsqueeze(0)
        
        # 加载心肌掩模（如果提供）
        myocardium_mask = None
        if myocardium_mask_path:
            mask_nii = nib.load(myocardium_mask_path)
            mask_data = mask_nii.get_fdata()
            
            if mask_data.ndim == 3:
                mask_data = mask_data[:, :, mask_data.shape[2] // 2]
            
            mask_data = self.resize(mask_data, (256, 256))
            myocardium_mask = torch.from_numpy(mask_data).float().unsqueeze(0).unsqueeze(0)
        
        return ed_tensor, es_tensor, myocardium_mask, affine
    
    def normalize(self, image):
        """归一化图像到[0, 1]"""
        min_val = np.min(image)
        max_val = np.max(image)
        if max_val > min_val:
            return (image - min_val) / (max_val - min_val)
        return image
    
    def resize(self, image, target_size):
        """调整图像大小"""
        from scipy.ndimage import zoom
        
        zoom_factors = [target_size[0] / image.shape[0], target_size[1] / image.shape[1]]
        resized = zoom(image, zoom_factors, order=1)
        
        return resized
    
    @torch.no_grad()
    def predict(self, ed_frame, es_frame, myocardium_mask=None):
        """
        预测心梗分割
        
        Args:
            ed_frame: ED帧张量 (1, 1, H, W)
            es_frame: ES帧张量 (1, 1, H, W)
            myocardium_mask: 心肌掩模（可选）
        Returns:
            mi_segmentation: 心梗分割结果 (H, W)
            motion_field: 运动场 (2, H, W)
            strain: 应变图 (H, W)
        """
        # 移到设备
        ed_frame = ed_frame.to(self.device)
        es_frame = es_frame.to(self.device)
        
        # 1. 估计运动场
        motion_field, warped_ed, _ = self.motion_net(ed_frame, es_frame)
        
        # 2. 计算应变
        strain = self.motion_net.compute_strain(motion_field)
        
        # 3. 准备分割网络输入
        # 输入: CMR图像 + motion_x + motion_y + strain
        seg_input = torch.cat([
            ed_frame,  # CMR图像
            motion_field[:, 0:1, :, :],  # motion_x
            motion_field[:, 1:2, :, :],  # motion_y
            strain  # strain
        ], dim=1)  # (1, 4, H, W)
        
        # 4. 心梗分割
        mi_logits = self.segmentation_net(seg_input)
        mi_prob = torch.sigmoid(mi_logits)
        mi_segmentation = (mi_prob > 0.5).float()
        
        # 5. 如果提供了心肌掩模，只保留心肌区域内的心梗
        if myocardium_mask is not None:
            myocardium_mask = myocardium_mask.to(self.device)
            mi_segmentation = mi_segmentation * myocardium_mask
        
        # 转换为numpy
        mi_segmentation = mi_segmentation.squeeze().cpu().numpy()
        motion_field = motion_field.squeeze().cpu().numpy()
        strain = strain.squeeze().cpu().numpy()
        
        return mi_segmentation, motion_field, strain
    
    def save_result(self, segmentation, output_path, affine):
        """
        保存分割结果为NIfTI格式
        
        Args:
            segmentation: 分割结果 (H, W)
            output_path: 输出路径
            affine: 仿射矩阵
        """
        # 创建NIfTI图像
        nii_img = nib.Nifti1Image(segmentation.astype(np.float32), affine)
        
        # 保存
        nib.save(nii_img, output_path)
        print(f"Segmentation saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='心梗分割推理')
    parser.add_argument('--input', type=str, required=True, help='输入CMR图像路径 (.nii.gz)')
    parser.add_argument('--myocardium_mask', type=str, default=None, help='心肌掩模路径（可选）')
    parser.add_argument('--output', type=str, required=True, help='输出分割结果路径 (.nii.gz)')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型权重路径 (.pth)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='计算设备')
    parser.add_argument('--save_motion', action='store_true', help='是否保存运动场和应变图')
    
    args = parser.parse_args()
    
    # 创建估计器
    estimator = MIEstimator(args.checkpoint, device=args.device)
    
    # 预处理
    print(f"Loading CMR image from {args.input}...")
    ed_frame, es_frame, myocardium_mask, affine = estimator.preprocess(
        args.input, 
        args.myocardium_mask
    )
    
    # 预测
    print("Running inference...")
    mi_segmentation, motion_field, strain = estimator.predict(
        ed_frame, 
        es_frame, 
        myocardium_mask
    )
    
    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    estimator.save_result(mi_segmentation, args.output, affine)
    
    # 保存运动场和应变图（如果需要）
    if args.save_motion:
        motion_path = output_path.parent / f"{output_path.stem}_motion.npy"
        strain_path = output_path.parent / f"{output_path.stem}_strain.npy"
        
        np.save(motion_path, motion_field)
        np.save(strain_path, strain)
        
        print(f"Motion field saved to {motion_path}")
        print(f"Strain map saved to {strain_path}")
    
    # 统计结果
    mi_volume = np.sum(mi_segmentation > 0)
    total_volume = mi_segmentation.size
    mi_percentage = (mi_volume / total_volume) * 100
    
    print(f"\n=== Results ===")
    print(f"MI pixels: {mi_volume}")
    print(f"Total pixels: {total_volume}")
    print(f"MI percentage: {mi_percentage:.2f}%")
    print(f"\nInference completed successfully!")


if __name__ == "__main__":
    main()
