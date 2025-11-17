# 数据集结构说明 (v2)

本项目使用的数据集结构如下。**所有数据（图像和标签）都采用 `case/slice` 两级目录结构。**

## 完整目录结构

```
data_root/
├── images/
│   ├── cmr/
│   │   ├── case001/
│   │   │   ├── slice_01.nii.gz  # CMR多帧序列 (T, H, W)
│   │   │   ├── slice_02.nii.gz
│   │   │   ├── slice_03.nii.gz
│   │   │   └── ...
│   │   ├── case002/
│   │   │   ├── slice_01.nii.gz
│   │   │   └── ...
│   │   └── ...
│   └── lge/
│       ├── case001/
│       │   ├── slice_01.nii.gz  # LGE单帧图像 (H, W) 或 (1, H, W)
│       │   ├── slice_02.nii.gz
│       │   └── ...
│       ├── case002/
│       │   └── ...
│       └── ...
└── labels/
    ├── cmr/
    │   └── cmr_Myo_mask/
    │       ├── case001/
    │       │   ├── slice_01.nii.gz  # CMR心肌掩码 (H, W)
    │       │   ├── slice_02.nii.gz
    │       │   └── ...
    │       ├── case002/
    │       │   └── ...
    │       └── ...
    └── lge_original/
        ├── lge_MI_labels/
        │   ├── case001/
        │   │   ├── slice_01.nii.gz  # 心梗标签 (H, W)
        │   │   ├── slice_02.nii.gz
        │   │   └── ...
        │   ├── case002/
        │   │   └── ...
        │   └── ...
        └── lge_Myo_labels/
            ├── case001/
            │   ├── slice_01.nii.gz  # LGE心肌掩码 (H, W)
            │   ├── slice_02.nii.gz
            │   └── ...
            ├── case002/
            │   └── ...
            └── ...
```

## 关键特点

### 1. **统一的两级目录结构**
- **所有数据**（图像和标签）都采用 `<case_id>/<slice_id>.nii.gz` 的组织方式
- 不存在单个case的3D体积文件，每个切片都是独立的文件

### 2. **切片级配对**
- CMR、LGE图像和所有标签的切片命名必须一致
- 例如：`case001/slice_01.nii.gz` 在所有目录中都应存在
- 这确保了数据加载时的正确配对

### 3. **多case支持**
- 数据集包含多个病例（case001, case002, case003, ...）
- 每个case可以有不同数量的切片
- 典型的切片数量：10-20个

## 数据说明

### 图像数据

#### CMR序列 (`images/cmr/<case_id>/<slice_id>.nii.gz`)
- **格式**: NIfTI (.nii.gz)
- **维度**: (T, H, W)
  - T: 时间帧数（通常20-40帧）
  - H, W: 图像高度和宽度（通常256x256或512x512）
- **内容**: 心脏电影磁共振成像序列，记录一个心动周期内的心脏运动
- **数据类型**: float32 或 float64

#### LGE图像 (`images/lge/<case_id>/<slice_id>.nii.gz`)
- **格式**: NIfTI (.nii.gz)
- **维度**: (H, W) 或 (1, H, W)
- **内容**: 延迟增强磁共振图像，用于显示心肌梗死区域
- **数据类型**: float32 或 float64

### 标签数据

#### CMR心肌掩码 (`labels/cmr/cmr_Myo_mask/<case_id>/<slice_id>.nii.gz`)
- **格式**: NIfTI (.nii.gz)
- **维度**: (H, W)
- **内容**: 2D心肌分割掩码
- **值**: 0=背景, 1=心肌
- **数据类型**: uint8 或 float32

#### LGE心肌掩码 (`labels/lge_original/lge_Myo_labels/<case_id>/<slice_id>.nii.gz`)
- **格式**: NIfTI (.nii.gz)
- **维度**: (H, W)
- **内容**: 2D心肌分割掩码（LGE图像上的）
- **值**: 0=背景, 1=心肌
- **数据类型**: uint8 或 float32

#### 心梗标签 (`labels/lge_original/lge_MI_labels/<case_id>/<slice_id>.nii.gz`)
- **格式**: NIfTI (.nii.gz)
- **维度**: (H, W)
- **内容**: 2D心肌梗死区域标签
- **值**: 0=正常, 1=心梗
- **数据类型**: uint8 或 float32

## 数据加载逻辑

项目的数据加载器 (`custom_dataloader_final.py`) 采用以下逻辑：

1. **构建切片索引**: 遍历所有case目录，为每个切片创建 `(case_id, slice_id)` 索引
2. **直接加载**: 根据索引直接加载对应的切片文件（不需要从3D体积中提取）
3. **配对验证**: 确保CMR、LGE和所有标签的同名切片都存在

### 示例代码

```python
from src.data.custom_dataloader_final import get_custom_dataloader_final

# 创建数据加载器
dataloader = get_custom_dataloader_final(
    data_root='/path/to/your/data',
    case_list=['case001', 'case002', 'case003'],
    mode='segmentation',
    batch_size=4,
    img_size=(256, 256),
    shuffle=True,
    num_workers=4
)

# 迭代数据
for batch in dataloader:
    cine_ed = batch['cine_ed']           # (B, 1, H, W)
    cine_es = batch['cine_es']           # (B, 1, H, W)
    lge = batch['lge']                   # (B, 1, H, W)
    myocardium_mask = batch['myocardium_mask']  # (B, H, W)
    infarct_mask = batch['infarct_mask']        # (B, H, W)
    # ... 训练代码
```

## 验证数据集

使用以下命令验证您的数据集结构：

```bash
python src/data/prepare_custom_dataset.py --data_root /path/to/your/data
```

验证脚本会检查：
- ✓ 目录结构是否完整
- ✓ 每个case的切片数量
- ✓ 每个切片的配套文件是否齐全（CMR、LGE、3种标签）
- ✓ 文件格式和维度是否正确

### 验证输出示例

```
==============================================================
DATASET VALIDATION
==============================================================
Data root: /path/to/your/data

Checking directory structure...
  ✓ Found: images/cmr
  ✓ Found: images/lge
  ✓ Found: labels/cmr/cmr_Myo_mask
  ✓ Found: labels/lge_original/lge_MI_labels
  ✓ Found: labels/lge_original/lge_Myo_labels

Found 50 cases
Validating each case...

[1/50] Validating case001...
  ✓ Valid (12 slices)
[2/50] Validating case002...
  ✓ Valid (15 slices)
...

==============================================================
VALIDATION SUMMARY
==============================================================
Total cases:   50
Valid cases:   50 (100.0%)
Invalid cases: 0 (0.0%)

Slice statistics:
  Total slices: 650
  Min per case: 8
  Max per case: 18
  Mean per case: 13.0
  Median per case: 12.0
==============================================================
```

## 常见问题

### Q1: 切片命名有什么要求？
A: 切片命名必须在CMR、LGE和所有标签目录中保持一致。推荐使用 `slice_01.nii.gz`, `slice_02.nii.gz` 等格式。

### Q2: 每个case必须有相同数量的切片吗？
A: 不需要。每个case可以有不同数量的切片，数据加载器会自动处理。

### Q3: 标签文件的值必须是0和1吗？
A: 推荐使用0（背景）和1（前景），但数据加载器会自动进行二值化处理（>0.5为1）。

### Q4: 如果某个切片缺少标签文件会怎样？
A: 验证脚本会报错，该case会被标记为invalid。训练时会跳过加载失败的样本。

## 数据预处理建议

1. **归一化**: 数据加载器会自动将图像归一化到[0, 1]
2. **尺寸调整**: 可以通过 `img_size` 参数指定目标尺寸
3. **数据增强**: 可以通过 `transform` 参数传入自定义的数据增强函数

## 相关文件

- `src/data/custom_dataloader_final.py` - 数据加载器实现
- `src/data/prepare_custom_dataset.py` - 数据验证脚本
- `src/data/split_dataset.py` - 数据集划分脚本
