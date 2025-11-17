# 模型权重说明

## 权重文件获取

由于GitHub对大文件的限制（单文件100MB，总推送限制），预训练的模型权重文件未直接包含在仓库中。

### 方式1：使用您的私有数据集训练（强烈推荐）

使用您的私有数据集训练模型，获得最适合您数据的权重：

```bash
# 1. 验证和准备数据集
python src/data/prepare_custom_dataset.py --data_root /path/to/your/data
python src/data/split_dataset.py --data_root /path/to/your/data

# 2. 训练各模块
DATA_ROOT="/path/to/your/data"
SPLITS_FILE="data/splits/dataset_splits.json"

python src/train_motion.py --data_root $DATA_ROOT --splits_file $SPLITS_FILE --epochs 100
python src/train_registration.py --data_root $DATA_ROOT --splits_file $SPLITS_FILE --epochs 100
python src/train_segmentation.py --data_root $DATA_ROOT --splits_file $SPLITS_FILE --epochs 100 \
    --motion_checkpoint checkpoints/motion_estimation/best_motion_model.pth
```

### 方式2：生成初始化权重

如果您只需要初始化权重用于快速开始：

```bash
python create_init_weights.py
```

## 演示训练结果

以下是使用合成数据进行的演示训练结果（仅供参考）：

### 运动估计模块
- **训练轮数**: 20 epochs
- **最终训练损失**: 0.0395
- **最佳验证损失**: 0.0388 (Epoch 19)
- **光度损失**: 0.0387
- **平滑度损失**: 0.0008
- **模型大小**: ~14MB
- **参数量**: 1,137,638

### 配准模块
- **训练轮数**: 20 epochs
- **最终训练损失**: -0.1473
- **最佳验证损失**: -0.1471
- **最佳Dice系数**: 1.0000 (完美配准)
- **模型大小**: ~23MB
- **参数量**: 1,956,674

### 分割模块
- **训练轮数**: 4 epochs (部分训练)
- **当前训练损失**: 0.4705
- **当前训练Dice**: 0.39
- **最佳验证Dice**: 0.4108 (Epoch 3)
- **模型大小**: ~400MB
- **参数量**: 34,879,149

**重要提示**: 以上结果是在随机生成的合成数据上获得的，仅用于演示训练流程。在真实的CMR/LGE数据集上，您需要重新训练以获得有意义的性能。

## 权重文件结构

训练完成后，您将获得以下权重文件：

```
checkpoints/
├── motion_estimation/
│   ├── best_motion_model.pth           # 最佳验证性能
│   ├── motion_epoch_*.pth              # 特定epoch的检查点
│   └── init_motion_model.pth           # 初始化权重
├── registration/
│   ├── best_registration_model.pth     # 最佳验证性能
│   ├── registration_epoch_*.pth        # 特定epoch的检查点
│   └── init_registration_model.pth     # 初始化权重
└── segmentation/
    ├── best_segmentation_model.pth     # 最佳验证性能
    ├── segmentation_epoch_*.pth        # 特定epoch的检查点
    └── init_segmentation_model.pth     # 初始化权重
```

## 使用预训练权重

### 在推理中使用

```bash
python src/inference.py \
    --input_cmr /path/to/your/data/images/cmr/<case_id>/<slice_id>.nii.gz \
    --input_mask /path/to/your/data/labels/cmr/cmr_Myo_mask/<case_id>.nii.gz \
    --slice_index <slice_index> \
    --output /path/to/output/mi_segmentation.nii.gz \
    --checkpoint checkpoints/segmentation/best_segmentation_model.pth
```

### 在代码中加载

```python
import torch
from src.models.attention_unet import AttentionUNet

# 加载分割模型
model = AttentionUNet(in_channels=4, out_channels=1)
checkpoint = torch.load('checkpoints/segmentation/best_segmentation_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

## 模型参数统计

| 模块 | 参数量 | 初始化权重 | 训练后权重 |
|------|--------|-----------|-----------|
| 运动估计 | 1,137,638 | ~5MB | ~14MB |
| 配准 | 1,956,674 | ~8MB | ~23MB |
| 分割 | 34,879,149 | ~134MB | ~400MB |
| **总计** | **37,973,461** | **~147MB** | **~437MB** |

## 数据集要求

您的私有数据集应遵循以下结构：

```
data/
├── images/
│   ├── cmr/<case_id>/<slice_id>.nii.gz    # 多帧CMR序列 (T, H, W)
│   └── lge/<case_id>/<slice_id>.nii.gz    # 单帧LGE图像 (1, H, W)
└── labels/
    ├── cmr/cmr_Myo_mask/<case_id>.nii.gz  # 3D心肌掩码
    └── lge/
        ├── lge_MI_labels/<case_id>.nii.gz # 3D心梗标签
        └── lge_Myo_labels/<case_id>.nii.gz # 3D LGE心肌掩码
```

## 训练时间估算

在单个GPU（如NVIDIA RTX 3090）上，使用真实数据训练：

- 运动估计：2-3小时（100 epochs）
- 配准：2-3小时（100 epochs）
- 分割：4-5小时（100 epochs）

总计约8-11小时。

## 常见问题

**Q: 为什么权重文件没有直接包含在仓库中？**

A: GitHub对单个文件有100MB的限制，且对推送总大小也有限制。我们的分割模型权重约400MB，超过了这个限制。

**Q: 演示训练的权重可以直接使用吗？**

A: 不建议。演示训练使用的是随机生成的合成数据，权重不具有实际意义。您需要使用真实的CMR/LGE数据重新训练。

**Q: 可以使用CPU训练吗？**

A: 可以，但速度会非常慢（约10-20倍）。强烈建议使用GPU训练。

**Q: 如何获取已训练好的权重？**

A: 由于数据隐私和文件大小限制，我们无法直接提供预训练权重。请使用您自己的数据集训练模型。
