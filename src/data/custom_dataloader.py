"""
自定义数据加载器 - 适配用户私有数据集格式 (v2)

数据集结构:
data/
├── images/
│   ├── cmr/
│   │   ├── case1/
│   │   │   ├── frame_01.nii.gz
│   │   │   ├── frame_02.nii.gz
│   │   │   └── ...
│   │   ├── case2/
│   │   └── ...
│   └── lge/
│       ├── case1/
│       │   ├── frame_01.nii.gz
│       │   ├── frame_02.nii.gz
│       │   └── ...
│       ├── case2/
│       └── ...
└── labels/
    ├── cmr/
    │   └── cmr_Myo_mask/
    │       ├── case1.nii.gz
    │       ├── case2.nii.gz
    │       └── ...
    └── lge/
        ├── lge_MI_labels/
        │   ├── case1.nii.gz
        │   ├── case2.nii.gz
        │   └── ...
        └── lge_Myo_labels/
            ├── case1.nii.gz
            ├── case2.nii.gz
            └── ...
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import nibabel as nib
from typing import Dict, List, Tuple, Optional
import random


class CustomMIDataset(Dataset):
    """
    自定义心梗数据集加载器 (v2)
    
    - LGE现在被视为与CMR具有相同结构的切片序列
    - 训练时将使用与CMR ED帧对应的LGE切片
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
        
        self._validate_data()
        
    def _validate_data(self):
        """验证数据完整性"""
        missing_files = []
        for case in self.case_list:
            if not (self.data_root / 'images' / 'cmr' / case).exists():
                missing_files.append(f"{case}: CMR directory missing")
            if not (self.data_root / 'images' / 'lge' / case).exists():
                missing_files.append(f"{case}: LGE directory missing")
            if not (self.data_root / 'labels' / 'cmr' / 'cmr_Myo_mask' / f'{case}.nii.gz').exists():
                missing_files.append(f"{case}: CMR Myo mask missing")
            if not (self.data_root / 'labels' / 'lge' / 'lge_MI_labels' / f'{case}.nii.gz').exists():
                missing_files.append(f"{case}: LGE MI label missing")
            if not (self.data_root / 'labels' / 'lge' / 'lge_Myo_labels' / f'{case}.nii.gz').exists():
                missing_files.append(f"{case}: LGE Myo label missing")
        
        if missing_files:
            print(f"Warning: {len(missing_files)} missing files/dirs found:")
            for msg in missing_files[:5]:
                print(f"  - {msg}")
    
    def __len__(self):
        return len(self.case_list)
    
    def _load_nifti(self, filepath: Path) -> np.ndarray:
        """加载NIfTI文件并处理3D->2D"""
        img = nib.load(str(filepath))
        data = img.get_fdata()
        if data.ndim == 3:
            data = data[:, :, data.shape[2] // 2]
        return data

    def _load_sequence(self, case: str, modality: str) -> Tuple[List[Path], np.ndarray]:
        """加载一个序列 (CMR或LGE)"""
        seq_dir = self.data_root / 'images' / modality / case
        frame_files = sorted(seq_dir.glob('*.nii.gz'))
        if not frame_files:
            raise FileNotFoundError(f"No frames found for {case} in {modality} directory")
        
        frames = [self._load_nifti(f) for f in frame_files]
        return frame_files, np.stack(frames, axis=0)

    def _resize(self, img: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """调整图像大小"""
        from scipy.ndimage import zoom
        if img.shape[-2:] == target_size:
            return img
        zoom_factors = [1.0] * (img.ndim - 2) + [target_size[0] / img.shape[-2], target_size[1] / img.shape[-1]]
        return zoom(img, zoom_factors, order=1)

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        """归一化到[0, 1]"""
        img = img.astype(np.float32)
        min_val, max_val = img.min(), img.max()
        return (img - min_val) / (max_val - min_val) if max_val > min_val else img

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        case = self.case_list[idx]
        
        # 加载CMR序列
        cmr_files, cmr_seq = self._load_sequence(case, 'cmr')
        cmr_seq = self._resize(cmr_seq, self.img_size)
        cmr_seq = self._normalize(cmr_seq)
        
        # 提取ED和ES帧 (舒张末期和收缩末期)
        T = cmr_seq.shape[0]
        ed_idx = 0
        es_idx = T // 2
        cine_ed = cmr_seq[ed_idx]
        cine_es = cmr_seq[es_idx]
        
        # 加载CMR心肌掩模
        cmr_mask_file = self.data_root / 'labels' / 'cmr' / 'cmr_Myo_mask' / f'{case}.nii.gz'
        cmr_mask = self._load_nifti(cmr_mask_file)
        cmr_mask = self._resize(cmr_mask, self.img_size)
        cmr_mask = (cmr_mask > 0.5).astype(np.float32)
        
        sample = {
            'case_id': case,
            'cine_ed': torch.from_numpy(cine_ed).unsqueeze(0).float(),
            'cine_es': torch.from_numpy(cine_es).unsqueeze(0).float(),
            'myocardium_mask': torch.from_numpy(cmr_mask).float(),
        }
        
        if self.mode in ['registration', 'segmentation']:
            # 加载LGE序列
            lge_files, lge_seq = self._load_sequence(case, 'lge')
            
            # 找到与CMR ED帧对应的LGE切片
            cmr_ed_filename = cmr_files[ed_idx].name
            lge_ed_file = self.data_root / 'images' / 'lge' / case / cmr_ed_filename
            
            if not lge_ed_file.exists():
                # 如果找不到完全匹配的文件名，则使用第一帧作为后备
                print(f"Warning: LGE slice {cmr_ed_filename} not found for {case}. Using first LGE slice.")
                lge_ed_frame = self._load_nifti(lge_files[0])
            else:
                lge_ed_frame = self._load_nifti(lge_ed_file)
            
            lge_ed_frame = self._resize(lge_ed_frame, self.img_size)
            lge_ed_frame = self._normalize(lge_ed_frame)
            sample['lge'] = torch.from_numpy(lge_ed_frame).unsqueeze(0).float()
            
            # 加载LGE心肌掩模
            lge_myo_file = self.data_root / 'labels' / 'lge' / 'lge_Myo_labels' / f'{case}.nii.gz'
            lge_myo = self._load_nifti(lge_myo_file)
            lge_myo = self._resize(lge_myo, self.img_size)
            sample['lge_myocardium_mask'] = torch.from_numpy((lge_myo > 0.5).astype(np.float32)).float()
            
            # 加载心梗标签
            mi_file = self.data_root / 'labels' / 'lge' / 'lge_MI_labels' / f'{case}.nii.gz'
            mi_label = self._load_nifti(mi_file)
            mi_label = self._resize(mi_label, self.img_size)
            sample['infarct_mask'] = torch.from_numpy((mi_label > 0.5).astype(np.float32)).float()

        if self.transform:
            sample = self.transform(sample)
            
        return sample


def get_custom_dataloader(
    data_root: str,
    case_list: List[str],
    mode: str = 'segmentation',
    batch_size: int = 4,
    img_size: Tuple[int, int] = (256, 256),
    shuffle: bool = True,
    num_workers: int = 4,
    transform=None
) -> DataLoader:
    dataset = CustomMIDataset(
        data_root=data_root,
        case_list=case_list,
        mode=mode,
        img_size=img_size,
        transform=transform
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, 
        num_workers=num_workers, pin_memory=True, drop_last=False
    )


if __name__ == '__main__':
    # 创建一个模拟的数据集结构用于测试
    print("Creating dummy dataset for testing...")
    dummy_root = Path('dummy_data')
    (dummy_root / 'images' / 'cmr' / 'case1').mkdir(parents=True, exist_ok=True)
    (dummy_root / 'images' / 'lge' / 'case1').mkdir(parents=True, exist_ok=True)
    (dummy_root / 'labels' / 'cmr' / 'cmr_Myo_mask').mkdir(parents=True, exist_ok=True)
    (dummy_root / 'labels' / 'lge' / 'lge_MI_labels').mkdir(parents=True, exist_ok=True)
    (dummy_root / 'labels' / 'lge' / 'lge_Myo_labels').mkdir(parents=True, exist_ok=True)

    # 创建模拟文件
    img_data = np.random.rand(128, 128)
    nifti_img = nib.Nifti1Image(img_data, np.eye(4))
    for i in range(5):
        nib.save(nifti_img, dummy_root / 'images' / 'cmr' / 'case1' / f'frame_{i+1:02d}.nii.gz')
        nib.save(nifti_img, dummy_root / 'images' / 'lge' / 'case1' / f'frame_{i+1:02d}.nii.gz')
    nib.save(nifti_img, dummy_root / 'labels' / 'cmr' / 'cmr_Myo_mask' / 'case1.nii.gz')
    nib.save(nifti_img, dummy_root / 'labels' / 'lge' / 'lge_MI_labels' / 'case1.nii.gz')
    nib.save(nifti_img, dummy_root / 'labels' / 'lge' / 'lge_Myo_labels' / 'case1.nii.gz')
    print("Dummy dataset created.")

    # 测试数据加载器
    dataloader = get_custom_dataloader(
        data_root=str(dummy_root),
        case_list=['case1'],
        mode='segmentation',
        batch_size=1,
        num_workers=0
    )
    
    print("\nTesting dataloader...")
    for batch in dataloader:
        print("\nBatch loaded successfully!")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  - {key}: {value.shape}")
        break

    # 清理模拟数据
    import shutil
    shutil.rmtree(dummy_root)
    print("\nDummy dataset cleaned up.")
