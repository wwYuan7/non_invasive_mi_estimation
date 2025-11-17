"""
自定义数据加载器 - 最终版 (v4)

根据用户最终确认的数据结构进行适配。

### 数据结构 (所有数据都是 case/slice 两级目录):
- **CMR图像**: `images/cmr/<case_id>/<slice_id>.nii.gz` (多帧, T>1)
- **LGE图像**: `images/lge/<case_id>/<slice_id>.nii.gz` (单帧, T=1)
- **CMR心肌掩码**: `labels/cmr/cmr_Myo_mask/<case_id>/<slice_id>.nii.gz` (2D)
- **LGE心肌掩码**: `labels/lge_original/lge_Myo_labels/<case_id>/<slice_id>.nii.gz` (2D)
- **心梗标签**: `labels/lge_original/lge_MI_labels/<case_id>/<slice_id>.nii.gz` (2D)

### 核心逻辑:
1. 以 **切片 (slice)** 为单位构建数据集
2. 在 `__init__` 中遍历所有case和slice，构建 `(case_id, slice_id)` 索引
3. 在 `__getitem__` 中，直接加载对应的切片文件（不需要从3D体积中提取）
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import nibabel as nib
from typing import Dict, List, Tuple, Optional


class CustomMIDatasetFinal(Dataset):
    """
    自定义心梗数据集加载器 (v4 - Final)
    """
    
    def __init__(
        self,
        data_root: str,
        case_list: List[str],
        mode: str = 'segmentation',
        img_size: Tuple[int, int] = (256, 256),
        target_frames: int = 40,
        transform=None
    ):
        self.data_root = Path(data_root)
        self.case_list = case_list
        self.mode = mode
        self.img_size = img_size
        self.target_frames = target_frames  # 目标帧数
        self.transform = transform
        
        # 构建切片索引
        self.slice_index = self._build_slice_index()
        
    def _build_slice_index(self) -> List[Tuple[str, str]]:
        """遍历所有case和slice，只选择完整匹配的slice（5个文件都存在）"""
        index = []
        skipped_count = 0
        
        for case_id in self.case_list:
            cmr_dir = self.data_root / 'images' / 'cmr' / case_id
            if not cmr_dir.exists():
                print(f"Warning: CMR directory not found for {case_id}")
                continue
            
            # 获取所有CMR切片文件并排序
            slice_files = sorted(cmr_dir.glob('*.nii.gz'))
            for slice_file in slice_files:
                slice_id = slice_file.stem.replace('.nii', '')  # 去除.nii.gz后缀
                
                # 检查该slice的所有必需文件是否存在
                required_files = [
                    self.data_root / 'images' / 'cmr' / case_id / f'{slice_id}.nii.gz',
                    self.data_root / 'images' / 'lge' / case_id / f'{slice_id}.nii.gz',
                    self.data_root / 'labels' / 'cmr' / 'cmr_Myo_mask' / case_id / f'{slice_id}.nii.gz',
                    self.data_root / 'labels' / 'lge_original' / 'lge_Myo_labels' / case_id / f'{slice_id}.nii.gz',
                    self.data_root / 'labels' / 'lge_original' / 'lge_MI_labels' / case_id / f'{slice_id}.nii.gz',
                ]
                
                # 只有所有文件都存在时才加入索引
                if all(f.exists() for f in required_files):
                    index.append((case_id, slice_id))
                else:
                    skipped_count += 1
        
        print(f"Found {len(index)} complete slices across {len(self.case_list)} cases.")
        if skipped_count > 0:
            print(f"Skipped {skipped_count} incomplete slices (missing files).")
        return index

    def __len__(self):
        return len(self.slice_index)

    def _load_nifti(self, filepath: Path) -> np.ndarray:
        """加载NIfTI文件"""
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        return nib.load(str(filepath)).get_fdata()

    def _resize(self, img: np.ndarray, is_mask: bool = False) -> np.ndarray:
        """调整图像/掩码大小"""
        from scipy.ndimage import zoom
        if img.shape[-2:] == self.img_size:
            return img
        
        order = 0 if is_mask else 1
        zoom_factors = [1.0] * (img.ndim - 2) + [self.img_size[0] / img.shape[-2], self.img_size[1] / img.shape[-1]]
        return zoom(img, zoom_factors, order=order)

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        """归一化到[0, 1]"""
        img = img.astype(np.float32)
        min_val, max_val = img.min(), img.max()
        return (img - min_val) / (max_val - min_val + 1e-8)
    
    def _temporal_interpolate(self, sequence: np.ndarray, target_frames: int) -> np.ndarray:
        """
        对CMR序列进行时间插值，统一到目标帧数
        
        Args:
            sequence: (T, H, W) 原始序列
            target_frames: 目标帧数
        
        Returns:
            (target_frames, H, W) 插值后的序列
        """
        from scipy.ndimage import zoom
        
        current_frames = sequence.shape[0]
        if current_frames == target_frames:
            return sequence
        
        # 计算时间维度的缩放因子
        zoom_factor = target_frames / current_frames
        
        # 对时间维度进行插值，空间维度保持不变
        zoom_factors = [zoom_factor, 1.0, 1.0]  # (T, H, W)
        
        # 使用线性插值 (order=1)
        interpolated = zoom(sequence, zoom_factors, order=1)
        
        return interpolated

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        case_id, slice_id = self.slice_index[idx]
        
        try:
            # 1. 加载CMR多帧序列 (T, H, W)
            cmr_path = self.data_root / 'images' / 'cmr' / case_id / f'{slice_id}.nii.gz'
            cmr_seq = self._load_nifti(cmr_path)
            cmr_seq = self._resize(cmr_seq)
            cmr_seq = self._normalize(cmr_seq)
            
            # 时间插值到目标帧数
            cmr_seq = self._temporal_interpolate(cmr_seq, self.target_frames)
            
            # 提取ED和ES帧
            cine_ed = cmr_seq[0]                      # ED帧 (第一帧)
            cine_es = cmr_seq[self.target_frames // 2] # ES帧 (中间帧)
            
            # 2. 加载CMR心肌掩模 (可能是多帧的，只取第一帧)
            cmr_mask_path = self.data_root / 'labels' / 'cmr' / 'cmr_Myo_mask' / case_id / f'{slice_id}.nii.gz'
            myocardium_mask = self._load_nifti(cmr_mask_path)
            
            # 如果是多帧的，只取第一帧
            if myocardium_mask.ndim == 3:
                myocardium_mask = myocardium_mask[0]  # 取第一帧
            
            myocardium_mask = myocardium_mask.squeeze()
            myocardium_mask = self._resize(myocardium_mask, is_mask=True)
            myocardium_mask = (myocardium_mask > 0.5).astype(np.float32)

            sample = {
                'case_id': f"{case_id}_{slice_id}",
                'cmr': torch.from_numpy(cmr_seq).float(),  # 完整序列 (T, H, W)
                'cine_ed': torch.from_numpy(cine_ed).unsqueeze(0).float(),
                'cine_es': torch.from_numpy(cine_es).unsqueeze(0).float(),
                'myocardium_mask': torch.from_numpy(myocardium_mask).float(),
            }

            if self.mode in ['registration', 'segmentation']:
                # 3. 加载LGE单帧图像
                lge_path = self.data_root / 'images' / 'lge' / case_id / f'{slice_id}.nii.gz'
                lge_img = self._load_nifti(lge_path).squeeze()  # 确保是2D
                lge_img = self._resize(lge_img)
                lge_img = self._normalize(lge_img)
                sample['lge'] = torch.from_numpy(lge_img).unsqueeze(0).float()

                # 4. 加载LGE心肌掩模 (2D切片)
                lge_myo_path = self.data_root / 'labels' / 'lge_original' / 'lge_Myo_labels' / case_id / f'{slice_id}.nii.gz'
                lge_myo_mask = self._load_nifti(lge_myo_path).squeeze()
                lge_myo_mask = self._resize(lge_myo_mask, is_mask=True)
                sample['lge_myocardium_mask'] = torch.from_numpy((lge_myo_mask > 0.5).astype(np.float32)).float()

                # 5. 加载心梗标签 (2D切片)
                mi_label_path = self.data_root / 'labels' / 'lge_original' / 'lge_MI_labels' / case_id / f'{slice_id}.nii.gz'
                infarct_mask = self._load_nifti(mi_label_path).squeeze()
                infarct_mask = self._resize(infarct_mask, is_mask=True)
                mi_label_tensor = torch.from_numpy((infarct_mask > 0.5).astype(np.float32)).float()
                sample['infarct_mask'] = mi_label_tensor
                sample['mi_label'] = mi_label_tensor  # 别名

            if self.transform:
                sample = self.transform(sample)
            
            return sample

        except Exception as e:
            print(f"Error loading slice {case_id}/{slice_id}: {e}")
            return None


def collate_fn(batch):
    """自定义collate_fn以过滤掉加载失败的样本"""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


def get_custom_dataloader_final(
    data_root: str,
    case_list: List[str],
    mode: str = 'segmentation',
    batch_size: int = 4,
    img_size: Tuple[int, int] = (256, 256),
    target_frames: int = 40,
    shuffle: bool = True,
    num_workers: int = 4,
    transform=None
) -> DataLoader:
    """创建数据加载器"""
    dataset = CustomMIDatasetFinal(
        data_root=data_root,
        case_list=case_list,
        mode=mode,
        img_size=img_size,
        target_frames=target_frames,
        transform=transform
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, 
        num_workers=num_workers, pin_memory=True, collate_fn=collate_fn
    )


if __name__ == '__main__':
    # --- 创建模拟数据集 ---
    print("Creating dummy dataset for testing...")
    dummy_root = Path('dummy_data_v4')
    case_id, slice_id = 'case001', 'slice_01'
    
    # 创建目录结构（所有都是case/slice两级）
    (dummy_root / 'images' / 'cmr' / case_id).mkdir(parents=True, exist_ok=True)
    (dummy_root / 'images' / 'lge' / case_id).mkdir(parents=True, exist_ok=True)
    (dummy_root / 'labels' / 'cmr' / 'cmr_Myo_mask' / case_id).mkdir(parents=True, exist_ok=True)
    (dummy_root / 'labels' / 'lge_original' / 'lge_MI_labels' / case_id).mkdir(parents=True, exist_ok=True)
    (dummy_root / 'labels' / 'lge_original' / 'lge_Myo_labels' / case_id).mkdir(parents=True, exist_ok=True)

    # 创建模拟文件
    cmr_data = np.random.rand(25, 128, 128)  # T=25, H=128, W=128
    lge_data = np.random.rand(128, 128)      # H=128, W=128 (2D)
    mask_data = np.random.randint(0, 2, size=(128, 128)).astype(np.float32)  # 2D mask

    nib.save(nib.Nifti1Image(cmr_data, np.eye(4)), dummy_root / 'images' / 'cmr' / case_id / f'{slice_id}.nii.gz')
    nib.save(nib.Nifti1Image(lge_data, np.eye(4)), dummy_root / 'images' / 'lge' / case_id / f'{slice_id}.nii.gz')
    nib.save(nib.Nifti1Image(mask_data, np.eye(4)), dummy_root / 'labels' / 'cmr' / 'cmr_Myo_mask' / case_id / f'{slice_id}.nii.gz')
    nib.save(nib.Nifti1Image(mask_data, np.eye(4)), dummy_root / 'labels' / 'lge_original' / 'lge_MI_labels' / case_id / f'{slice_id}.nii.gz')
    nib.save(nib.Nifti1Image(mask_data, np.eye(4)), dummy_root / 'labels' / 'lge_original' / 'lge_Myo_labels' / case_id / f'{slice_id}.nii.gz')
    print("Dummy dataset created.")

    # --- 测试数据加载器 ---
    dataloader = get_custom_dataloader_final(
        data_root=str(dummy_root),
        case_list=[case_id],
        mode='segmentation',
        batch_size=1,
        num_workers=0
    )
    
    print("\nTesting dataloader...")
    batch = next(iter(dataloader))
    if batch:
        print("\nBatch loaded successfully!")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  - {key}: {value.shape}")
            else:
                print(f"  - {key}: {value}")
    else:
        print("\nFailed to load batch.")

    # --- 清理模拟数据 ---
    import shutil
    shutil.rmtree(dummy_root)
    print("\nDummy dataset cleaned up.")
