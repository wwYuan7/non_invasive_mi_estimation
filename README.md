# 无创心梗估计 (Non-Invasive Myocardial Infarction Estimation)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

深度学习框架，用于从心脏磁共振电影成像（CMR）直接预测心肌梗死（MI）分割，无需延迟钆增强（LGE）图像。

## 项目概述

本项目旨在开发一个端到端的深度学习系统，能够从常规的CMR cine序列直接分割心肌梗死区域。传统方法依赖于LGE图像作为金标准，但LGE与CMR之间存在配准困难。本方案通过以下创新点解决这一问题：

1.  **CMR-LGE配准模块**: 使用基于VoxelMorph的无监督可变形配准网络，将LGE上的心梗标签准确映射到CMR空间。
2.  **心脏运动场估计模块**: 采用Motion Pyramid Networks提取心肌运动信息，识别功能异常区域。
3.  **心梗分割模块**: 融合CMR图像和运动特征，实现准确的心梗分割。

## 数据准备 (您的私有数据集)

本项目已根据您的私有数据集格式进行了定制。请确保您的数据遵循以下结构：

```
data/
├── images/
│   ├── cmr/
│   │   └── <case_id>/         # 病例文件夹
│   │       └── <slice_id>.nii.gz  # 多帧CMR序列 (T, H, W)
│   └── lge/
│       └── <case_id>/         # 病例文件夹
│           └── <slice_id>.nii.gz  # 单帧LGE图像 (1, H, W)
└── labels/
    ├── cmr/
    │   └── cmr_Myo_mask/
    │       └── <case_id>.nii.gz   # 3D心肌掩码 (H, W, D)
    └── lge/
        ├── lge_MI_labels/
        │   └── <case_id>.nii.gz   # 3D心梗标签 (H, W, D)
        └── lge_Myo_labels/
            └── <case_id>.nii.gz   # 3D LGE心肌掩码 (H, W, D)
```

### 使用流程

1.  **验证数据集**: 在开始训练前，强烈建议运行验证脚本检查数据完整性。

    ```bash
    python src/data/prepare_custom_dataset.py --data_root /path/to/your/data
    ```

2.  **分割数据集**: 将数据集划分为训练、验证和测试集。

    ```bash
    python src/data/split_dataset.py --data_root /path/to/your/data
    ```
    这将在 `data/splits` 目录下生成 `dataset_splits.json` 文件。

## 使用方法

### 模型训练

项目提供了三个独立的训练脚本，用于训练各个模块。请在训练命令中指定数据集根目录和分割文件。

```bash
DATA_ROOT="/path/to/your/data"
SPLITS_FILE="data/splits/dataset_splits.json"

# 1. 训练运动估计模块
python src/train_motion.py --data_root $DATA_ROOT --splits_file $SPLITS_FILE --epochs 100

# 2. 训练配准模块
python src/train_registration.py --data_root $DATA_ROOT --splits_file $SPLITS_FILE --epochs 100

# 3. 训练分割模块
python src/train_segmentation.py --data_root $DATA_ROOT --splits_file $SPLITS_FILE --epochs 100 \
    --motion_checkpoint checkpoints/motion_estimation/best_motion_model.pth
```

### 推理

```bash
python src/inference.py \
    --input_cmr /path/to/your/data/images/cmr/<case_id>/<slice_id>.nii.gz \
    --input_mask /path/to/your/data/labels/cmr/cmr_Myo_mask/<case_id>.nii.gz \
    --slice_index <slice_index_in_3d_volume> \
    --output /path/to/output/mi_segmentation.nii.gz \
    --checkpoint checkpoints/segmentation/best_segmentation_model.pth
```

## 项目结构

```
non_invasive_mi_estimation/
├── README.md
├── requirements.txt
├── src/
│   ├── models/                 # 模型定义
│   ├── data/                   # 数据处理
│   │   ├── custom_dataloader_final.py # 最终版数据加载器
│   │   ├── prepare_custom_dataset.py  # 数据验证脚本
│   │   └── split_dataset.py           # 数据分割脚本
│   ├── utils/                  # 工具函数
│   ├── train_motion.py         # 运动估计训练脚本
│   ├── train_registration.py  # 配准训练脚本
│   ├── train_segmentation.py  # 分割训练脚本
│   └── inference.py           # 推理脚本
├── checkpoints/               # 模型权重
└── ...
```

## 许可证

本项目采用MIT许可证。详见 [LICENSE](LICENSE) 文件。
