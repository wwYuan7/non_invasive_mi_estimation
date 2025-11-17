# 数据集结构说明

本项目使用的数据集结构如下：

## 完整目录结构

```
data_root/
├── images/
│   ├── cmr/
│   │   ├── case001/
│   │   │   ├── slice_01.nii.gz  # CMR多帧序列 (T, H, W)
│   │   │   ├── slice_02.nii.gz
│   │   │   └── ...
│   │   ├── case002/
│   │   └── ...
│   └── lge/
│       ├── case001/
│       │   ├── slice_01.nii.gz  # LGE单帧图像 (1, H, W)
│       │   ├── slice_02.nii.gz
│       │   └── ...
│       ├── case002/
│       └── ...
└── labels/
    ├── cmr/
    │   └── cmr_Myo_mask/
    │       ├── case001.nii.gz  # CMR心肌掩码 (H, W, D)
    │       ├── case002.nii.gz
    │       └── ...
    └── lge_original/
        ├── lge_MI_labels/
        │   ├── case001.nii.gz  # 心梗标签 (H, W, D)
        │   ├── case002.nii.gz
        │   └── ...
        └── lge_Myo_labels/
            ├── case001.nii.gz  # LGE心肌掩码 (H, W, D)
            ├── case002.nii.gz
            └── ...
```

## 数据说明

### 图像数据

#### CMR序列 (`images/cmr/<case_id>/<slice_id>.nii.gz`)
- **格式**: NIfTI (.nii.gz)
- **维度**: (T, H, W)
  - T: 时间帧数（通常20-40帧）
  - H, W: 图像高度和宽度
- **内容**: 心脏电影磁共振成像序列，记录一个心动周期内的心脏运动

#### LGE图像 (`images/lge/<case_id>/<slice_id>.nii.gz`)
- **格式**: NIfTI (.nii.gz)
- **维度**: (1, H, W) 或 (H, W)
- **内容**: 延迟增强磁共振图像，用于显示心肌梗死区域

### 标签数据

#### CMR心肌掩码 (`labels/cmr/cmr_Myo_mask/<case_id>.nii.gz`)
- **格式**: NIfTI (.nii.gz)
- **维度**: (H, W, D)
  - D: 切片数量（深度）
- **内容**: 3D心肌分割掩码，每个切片对应一个2D掩码
- **值**: 0=背景, 1=心肌

#### LGE心肌掩码 (`labels/lge_original/lge_Myo_labels/<case_id>.nii.gz`)
- **格式**: NIfTI (.nii.gz)
- **维度**: (H, W, D)
- **内容**: 3D心肌分割掩码（LGE图像上的）
- **值**: 0=背景, 1=心肌

#### 心梗标签 (`labels/lge_original/lge_MI_labels/<case_id>.nii.gz`)
- **格式**: NIfTI (.nii.gz)
- **维度**: (H, W, D)
- **内容**: 3D心肌梗死区域标签
- **值**: 0=正常, 1=心梗

## 关键特点

1. **多case结构**: 数据集包含多个病例（case001, case002, ...）

2. **切片级组织**: 
   - CMR和LGE图像按切片组织，每个切片一个文件
   - 切片命名需要在CMR和LGE之间对应（如slice_01.nii.gz）

3. **3D标签**: 
   - 所有标签文件都是3D体积
   - 每个case一个标签文件
   - 包含该case所有切片的标注

4. **时间维度**:
   - CMR: 多帧序列（T>1），记录心脏运动
   - LGE: 单帧图像（T=1），静态对比增强图像

## 数据加载逻辑

项目的数据加载器 (`custom_dataloader_final.py`) 采用以下逻辑：

1. **以切片为单位**: 构建 `(case_id, slice_id, slice_idx)` 索引
2. **动态提取**: 从3D标签体积中动态提取对应切片的2D掩码
3. **配对加载**: 确保CMR和LGE的同名切片被正确配对

## 验证数据集

使用以下命令验证您的数据集结构：

```bash
python src/data/prepare_custom_dataset.py --data_root /path/to/your/data
```

这将检查：
- 目录结构是否完整
- 每个case的文件是否齐全
- 文件格式是否正确
- 数据维度是否合理
