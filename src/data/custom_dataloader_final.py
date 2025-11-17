"""
自定义数据加载器 - 最终版 (v3)

根据用户最终澄清的数据结构进行适配。

### 数据结构:
- **CMR**: `images/cmr/<case_id>/<slice_id>.nii.gz` (多帧, T>1)
- **LGE**: `images/lge/<case_id>/<slice_id>.nii.gz` (单帧, T=1)
- **Masks**: `labels/.../<case_id>.nii.gz` (3D体积, 包含所有切片)

### 核心逻辑:
1.  以 **切片 (slice)** 为单位构建数据集，而不是以病例 (case) 为单位。
2.  在 `__init__` 中遍历所有病例和切片，构建一个包含 `(case_id, slice_id)` 的索引列表。
3.  在 `__getitem__` 中，根据索引加载对应的CMR多帧序列、LGE单帧图像，并从3D Mask体积中提取出当前切片对应的2D掩码。
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
    自定义心梗数据集加载器 (v3 - Final)
    """
    
    def __init__(
        self,
        data_root: str,
        case_list: List[str],
        mode: str = 'segmentation',
        img_size: Tuple[int, int] = (256, 256),
        transform=None
    ):
        self.data_root = Path(data_root)
        self.case_list = case_list
        self.mode = mode
        self.img_size = img_size
        self.transform = transform
        
        # 构建切片索引
        self.slice_index = self._build_slice_index()
        
    def _build_slice_index(self) -> List[Tuple[str, str, int]]:
        """遍历所有病例和切片，构建 (case_id, slice_id, slice_idx) 索引"""
        index = []
        for case_id in self.case_list:
            cmr_dir = self.data_root / 'images' / 'cmr' / case_id
            if not cmr_dir.exists():
                continue
            
            # 获取所有切片文件并排序
            slice_files = sorted(cmr_dir.glob('*.nii.gz'))
            for i, slice_file in enumerate(slice_files):
                slice_id = slice_file.name.replace('.nii.gz', '')  # e.g., 'slice_01'
                index.append((case_id, slice_id, i))
        
        print(f"Found {len(index)} slices across {len(self.case_list)} cases.")
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
        return (img - min_val) / (max_val - min_val) if max_val > min_val else img

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        case_id, slice_id, slice_idx = self.slice_index[idx]
        
        try:
            # 1. 加载CMR多帧序列 (T, H, W)
            cmr_path = self.data_root / 'images' / 'cmr' / case_id / f'{slice_id}.nii.gz'
            cmr_seq = self._load_nifti(cmr_path)
            cmr_seq = self._resize(cmr_seq)
            cmr_seq = self._normalize(cmr_seq)
            
            # 提取ED和ES帧
            T = cmr_seq.shape[0]
            cine_ed = cmr_seq[0]      # (H, W)
            cine_es = cmr_seq[T // 2] # (H, W)
            
            # 2. 加载CMR心肌掩模 (从3D体积中提取2D切片)
            cmr_mask_path = self.data_root / 'labels' / 'cmr' / 'cmr_Myo_mask' / f'{case_id}.nii.gz'
            cmr_mask_3d = self._load_nifti(cmr_mask_path)
            myocardium_mask = cmr_mask_3d[:, :, slice_idx]
            myocardium_mask = self._resize(myocardium_mask, is_mask=True)
            myocardium_mask = (myocardium_mask > 0.5).astype(np.float32)

            sample = {
                'case_id': f"{case_id}_{slice_id}",
                'cine_ed': torch.from_numpy(cine_ed).unsqueeze(0).float(),
                'cine_es': torch.from_numpy(cine_es).unsqueeze(0).float(),
                'myocardium_mask': torch.from_numpy(myocardium_mask).float(),
            }

            if self.mode in ['registration', 'segmentation']:
                # 3. 加载LGE单帧图像
                lge_path = self.data_root / 'images' / 'lge' / case_id / f'{slice_id}.nii.gz'
                lge_img = self._load_nifti(lge_path).squeeze() # 确保是2D
                lge_img = self._resize(lge_img)
                lge_img = self._normalize(lge_img)
                sample['lge'] = torch.from_numpy(lge_img).unsqueeze(0).float()

                # 4. 加载LGE心肌掩模
                lge_myo_path = self.data_root / 'labels' / 'lge' / 'lge_Myo_labels' / f'{case_id}.nii.gz'
                lge_myo_3d = self._load_nifti(lge_myo_path)
                lge_myo_mask = lge_myo_3d[:, :, slice_idx]
                lge_myo_mask = self._resize(lge_myo_mask, is_mask=True)
                sample['lge_myocardium_mask'] = torch.from_numpy((lge_myo_mask > 0.5).astype(np.float32)).float()

                # 5. 加载心梗标签
                mi_label_path = self.data_root / 'labels' / 'lge' / 'lge_MI_labels' / f'{case_id}.nii.gz'
                mi_label_3d = self._load_nifti(mi_label_path)
                infarct_mask = mi_label_3d[:, :, slice_idx]
                infarct_mask = self._resize(infarct_mask, is_mask=True)
                sample['infarct_mask'] = torch.from_numpy((infarct_mask > 0.5).astype(np.float32)).float()

            if self.transform:
                sample = self.transform(sample)
            
            return sample

        except Exception as e:
            print(f"Error loading slice {case_id}/{slice_id}: {e}")
            # 返回一个空的或占位的样本，或者可以跳过
            return None


def collate_fn(batch):
    """自定义collate_fn以过滤掉加载失败的样本 (返回None的)"""
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
    shuffle: bool = True,
    num_workers: int = 4,
    transform=None
) -> DataLoader:
    dataset = CustomMIDatasetFinal(
        data_root=data_root,
        case_list=case_list,
        mode=mode,
        img_size=img_size,
        transform=transform
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, 
        num_workers=num_workers, pin_memory=True, collate_fn=collate_fn
    )


if __name__ == '__main__':
    # --- 创建模拟数据集 ---
    print("Creating dummy dataset for testing...")
    dummy_root = Path('dummy_data_final')
    case_id, slice_id, slice_idx = 'case1', 'slice_01', 0
    
    # 创建目录
    (dummy_root / 'images' / 'cmr' / case_id).mkdir(parents=True, exist_ok=True)
    (dummy_root / 'images' / 'lge' / case_id).mkdir(parents=True, exist_ok=True)
    (dummy_root / 'labels' / 'cmr' / 'cmr_Myo_mask').mkdir(parents=True, exist_ok=True)
    (dummy_root / 'labels' / 'lge' / 'lge_MI_labels').mkdir(parents=True, exist_ok=True)
    (dummy_root / 'labels' / 'lge' / 'lge_Myo_labels').mkdir(parents=True, exist_ok=True)

    # 创建模拟文件
    cmr_data = np.random.rand(5, 128, 128) # T=5, H=128, W=128
    lge_data = np.random.rand(1, 128, 128) # T=1, H=128, W=128
    mask_data = np.random.randint(0, 2, size=(128, 128, 10)).astype(np.float32) # H, W, D=10

    nib.save(nib.Nifti1Image(cmr_data, np.eye(4)), dummy_root / 'images' / 'cmr' / case_id / f'{slice_id}.nii.gz')
    nib.save(nib.Nifti1Image(lge_data, np.eye(4)), dummy_root / 'images' / 'lge' / case_id / f'{slice_id}.nii.gz')
    nib.save(nib.Nifti1Image(mask_data, np.eye(4)), dummy_root / 'labels' / 'cmr' / 'cmr_Myo_mask' / f'{case_id}.nii.gz')
    nib.save(nib.Nifti1Image(mask_data, np.eye(4)), dummy_root / 'labels' / 'lge' / 'lge_MI_labels' / f'{case_id}.nii.gz')
    nib.save(nib.Nifti1Image(mask_data, np.eye(4)), dummy_root / 'labels' / 'lge' / 'lge_Myo_labels' / f'{case_id}.nii.gz')
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
