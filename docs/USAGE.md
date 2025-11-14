# 项目使用说明

本文档提供“无创心梗估计”项目的详细使用说明，包括环境设置、数据准备、模型训练、推理和评估的全过程。

## 1. 环境设置

### 1.1. 硬件要求

- **GPU**: 推荐使用NVIDIA GPU，显存 >= 12GB，支持CUDA 11.0+
- **CPU**: 多核CPU
- **内存**: >= 32GB

### 1.2. 软件环境

我们推荐使用`conda`来管理Python环境，以确保依赖的隔离和一致性。

```bash
# 步骤1: 克隆GitHub仓库
git clone https://github.com/wwYuan7/non_invasive_mi_estimation.git
cd non_invasive_mi_estimation

# 步骤2: 创建并激活conda环境
conda create -n mi_estimation python=3.8 -y
conda activate mi_estimation

# 步骤3: 安装依赖项
# requirements.txt 文件包含了所有必要的Python库
pip install -r requirements.txt
```

**注意**: `voxelmorph`库可能需要从其官方GitHub仓库安装以获取最新版本，如果`pip`安装失败，请尝试以下命令：

```bash
pip install git+https://github.com/voxelmorph/voxelmorph.git
```

## 2. 数据准备

### 2.1. 数据集下载

本项目支持多个公开数据集。请按照以下说明下载并组织数据。

- **MS-CMR 2019 (主要数据集)**
  1. 访问 [MS-CMR官网](https://zmiclab.github.io/zxh/0/mscmrseg19/) 注册并申请数据。
  2. 下载后，将数据解压到 `data/raw/mscmr/` 目录下。

- **EMIDEC (辅助数据集)**
  1. 访问 [EMIDEC官网](https://emidec.com/) 注册并下载数据。
  2. 解压到 `data/raw/emidec/` 目录下。

- **ACDC (辅助数据集)**
  1. 访问 [ACDC官网](https://www.creatis.insa-lyon.fr/Challenge/acdc/) 下载数据。
  2. 解压到 `data/raw/acdc/` 目录下。

### 2.2. 数据预处理

原始数据需要经过预处理才能用于模型训练。预处理步骤包括：图像重采样、归一化、切片提取等。

```bash
# 预处理MS-CMR数据集
python src/data/preprocess_mscmr.py --input data/raw/mscmr --output data/processed/mscmr

# (可选) 预处理EMIDEC数据集
python src/data/preprocess_emidec.py --input data/raw/emidec --output data/processed/emidec
```

预处理脚本会将数据转换为统一的格式（如NIfTI或HDF5），并保存在 `data/processed/` 目录下。

## 3. 模型训练

训练过程被设计为多阶段，可以逐步训练各个模块，也可以进行端到端联合训练。所有训练参数都在 `configs/` 目录下的YAML文件中配置。

### 3.1. 配置文件说明

以 `configs/end2end.yaml` 为例，主要配置项包括：

- `data`: 数据集路径、批大小、图像尺寸等。
- `model`: 各个网络模块的结构参数和预训练权重路径。
- `training`: 优化器、学习率、训练周期、损失函数权重等。
- `evaluation`: 评估指标和可视化选项。
- `logging`: 日志目录、TensorBoard/WandB配置。

### 3.2. 训练命令

#### 阶段一：预训练运动估计模块 (MotionNet)

此阶段使用CMR cine序列（如ACDC数据集）训练运动估计网络。

```bash
python src/train_motion.py --config configs/motion_estimation.yaml
```

#### 阶段二：预训练配准模块 (VoxelMorph)

此阶段使用未配准的CMR和LGE图像对（如MS-CMR数据集）训练配准网络。

```bash
python src/train_registration.py --config configs/registration.yaml
```

#### 阶段三：训练心梗分割模块 (AttentionUNet)

此阶段使用配准后的LGE标签和提取的运动特征训练分割网络。

```bash
python src/train_segmentation.py --config configs/segmentation.yaml
```

#### 阶段四：端到端联合训练

此阶段将所有模块连接起来，进行端到端的微调。

```bash
python src/train_end2end.py --config configs/end2end.yaml
```

训练过程中，模型权重会保存在 `checkpoints/` 目录下，训练日志会保存在 `logs/` 目录下。

## 4. 推理

训练完成后，可以使用 `inference.py` 脚本对新的CMR图像进行心梗分割预测。

### 4.1. 推理命令

```bash
python src/inference.py \
    --input /path/to/your/cmr_image.nii.gz \
    --myocardium_mask /path/to/your/myocardium_mask.nii.gz \
    --output /path/to/save/mi_segmentation.nii.gz \
    --checkpoint checkpoints/best_model_end2end.pth \
    --device cuda
```

### 4.2. 参数说明

- `--input`: 输入的CMR图像路径（NIfTI格式，4D或3D）。
- `--myocardium_mask`: (可选) 输入的CMR第一帧的心肌掩模，用于约束分割范围。
- `--output`: 输出的心梗分割结果路径。
- `--checkpoint`: 训练好的模型权重文件路径。
- `--device`: 使用`cuda`或`cpu`。
- `--save_motion`: (可选) 是否保存中间结果，如运动场和应变图。

## 5. 评估

使用 `evaluate.py` 脚本可以评估模型在测试集上的性能。

### 5.1. 评估命令

```bash
python src/evaluate.py \
    --predictions /path/to/your/predictions/ \
    --ground_truth /path/to/ground_truth/ \
    --metrics dice hausdorff asd sensitivity specificity
```

### 5.2. 参数说明

- `--predictions`: 模型预测结果所在的目录。
- `--ground_truth`: 真实标签所在的目录。
- `--metrics`: 需要计算的评估指标列表，支持 `dice`, `hausdorff`, `asd`, `sensitivity`, `specificity`。

评估结果将以表格形式打印在控制台，并可以保存为CSV文件。

## 6. 项目结构详解

```
non_invasive_mi_estimation/
├── README.md                   # 项目简介
├── docs/USAGE.md               # 详细使用说明
├── requirements.txt            # Python依赖
├── configs/                    # 配置文件目录
├── src/                        # 源代码目录
│   ├── models/                 # 模型定义 (VoxelMorph, MotionNet, UNet)
│   ├── data/                   # 数据加载和预处理
│   ├── utils/                  # 工具函数 (损失、指标、可视化)
│   ├── train_*.py              # 各阶段训练脚本
│   ├── inference.py            # 推理脚本
│   └── evaluate.py             # 评估脚本
├── data/                       # 数据目录 (git忽略)
│   ├── raw/                    # 原始数据
│   └── processed/              # 预处理后的数据
├── checkpoints/                # 模型权重 (git忽略)
└── logs/                       # 训练日志 (git忽略)
```
