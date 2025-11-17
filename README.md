# 无创心梗估计 (Non-Invasive Myocardial Infarction Estimation)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

深度学习框架，用于从心脏磁共振电影成像（CMR）直接预测心肌梗死（MI）分割，无需延迟钆增强（LGE）图像。

## 项目概述

本项目旨在开发一个端到端的深度学习系统，能够从常规的CMR cine序列直接分割心肌梗死区域。传统方法依赖于LGE图像作为金标准，但LGE与CMR之间存在配准困难（位移、形变、时间维度不匹配）。本方案通过以下创新点解决这一问题：

1. **CMR-LGE配准模块**: 使用基于VoxelMorph的无监督可变形配准网络，将LGE上的心梗标签准确映射到CMR空间
2. **心脏运动场估计模块**: 采用Motion Pyramid Networks提取心肌运动信息，识别功能异常区域
3. **心梗分割模块**: 融合CMR图像和运动特征，实现准确的心梗分割

## 主要特性

- ✅ 端到端的CMR到心梗分割流程
- ✅ 无监督的多模态配准
- ✅ 基于运动场的功能评估
- ✅ 支持MS-CMR、EMIDEC、ACDC等公开数据集
- ✅ 模块化设计，易于扩展
- ✅ 完整的训练和推理脚本

## 系统架构

```
CMR序列 ──┬──> 运动场估计模块 ──┐
          │                      ├──> 心梗分割模块 ──> 心梗分割结果
          └──> CMR-LGE配准模块 ──┘
                    ↑
               LGE图像 (训练阶段)
```

## 安装

### 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.0+ (推荐用于GPU加速)

### 安装步骤

```bash
# 克隆仓库
git clone https://github.com/wwYuan7/non_invasive_mi_estimation.git
cd non_invasive_mi_estimation

# 创建虚拟环境
conda create -n mi_estimation python=3.8
conda activate mi_estimation

# 安装依赖
pip install -r requirements.txt
```

## 数据准备

本项目支持以下公开数据集：

### MS-CMR 2019 (推荐)

MS-CMR数据集包含配对的CMR cine和LGE图像，是本项目的首选数据集。

1. 访问 [MS-CMR官网](https://zmiclab.github.io/zxh/0/mscmrseg19/)
2. 注册并签署数据使用协议
3. 下载数据并放置在 `data/raw/mscmr/` 目录下
4. 运行预处理脚本：

```bash
python src/data/preprocess_mscmr.py --input data/raw/mscmr --output data/processed/mscmr
```

### EMIDEC (辅助数据集)

EMIDEC数据集包含LGE图像和详细的心梗标注，可用于预训练心梗分割模块。

1. 访问 [EMIDEC官网](https://emidec.com/)
2. 注册并同意CC BY-NC-SA 4.0许可
3. 下载数据并放置在 `data/raw/emidec/` 目录下

### ACDC (辅助数据集)

ACDC数据集包含CMR cine序列和心室分割标注，可用于预训练运动估计和分割模块。

1. 访问 [ACDC官网](https://www.creatis.insa-lyon.fr/Challenge/acdc/)
2. 下载数据并放置在 `data/raw/acdc/` 目录下

## 使用方法

### 模型训练与权重

#### 权重文件

由于模型权重文件较大，未直接包含在Git仓库中。请参考 `checkpoints/README.md` 文件了解如何生成或训练权重。

#### 训练脚本

项目提供了三个独立的训练脚本，用于训练各个模块：

- `src/train_motion.py`: 训练运动估计模块
- `src/train_registration.py`: 训练配准模块
- `src/train_segmentation.py`: 训练分割模块

**训练命令示例**:

```bash
# 1. 训练运动估计模块
python src/train_motion.py --epochs 100

# 2. 训练配准模块
python src/train_registration.py --epochs 100

# 3. 训练分割模块（依赖于预训练的运动模型）
python src/train_segmentation.py --epochs 100 --motion_checkpoint checkpoints/motion_estimation/best_motion_model.pth
```

更多训练参数请查看各训练脚本的帮助信息 (`--help`)。

### 推理

```bash
python src/inference.py \
    --input path/to/cmr/image.nii.gz \
    --myocardium_mask path/to/myocardium/mask.nii.gz \
    --output path/to/output/mi_segmentation.nii.gz \
    --checkpoint checkpoints/segmentation/best_segmentation_model.pth
```

### 评估

```bash
python src/evaluate.py \
    --predictions path/to/predictions/ \
    --ground_truth path/to/ground_truth/ \
    --metrics dice hausdorff asd
```

## 模型架构

### 1. CMR-LGE配准模块

基于VoxelMorph的无监督可变形配准网络：

- **编码器**: 5层卷积，提取多尺度特征
- **解码器**: 5层反卷积，生成变形场
- **损失函数**: NCC相似性损失 + 平滑度正则化

### 2. 心脏运动场估计模块

基于Motion Pyramid Networks的运动场估计：

- **金字塔结构**: 3个尺度的运动场预测
- **师生训练**: 循环知识蒸馏提升性能
- **输出**: 像素级位移矢量场和应变图

### 3. 心梗分割模块

基于Attention U-Net的分割网络：

- **输入通道**: CMR图像 + 运动场 + 应变图
- **注意力机制**: 空间和通道注意力
- **损失函数**: Dice Loss + Focal Loss

## 配置文件

所有训练和推理参数都在 `configs/` 目录下的YAML文件中配置。主要配置项包括：

- 数据路径和预处理参数
- 网络架构参数
- 训练超参数（学习率、批大小、epoch数等）
- 损失函数权重
- 评估指标

## 项目结构

```
non_invasive_mi_estimation/
├── README.md                   # 项目说明
├── requirements.txt            # Python依赖
├── setup.py                    # 安装脚本
├── configs/                    # 配置文件
│   ├── motion_estimation.yaml
│   ├── registration.yaml
│   ├── segmentation.yaml
│   └── end2end.yaml
├── src/                        # 源代码
│   ├── models/                 # 模型定义
│   │   ├── voxelmorph.py      # VoxelMorph配准网络
│   │   ├── motion_pyramid.py  # 运动场估计网络
│   │   ├── unet.py            # U-Net分割网络
│   │   └── attention_unet.py  # Attention U-Net
│   ├── data/                   # 数据处理
│   │   ├── dataset.py         # 数据集类
│   │   ├── preprocess_mscmr.py
│   │   ├── preprocess_emidec.py
│   │   └── augmentation.py
│   ├── utils/                  # 工具函数
│   │   ├── losses.py          # 损失函数
│   │   ├── metrics.py         # 评估指标
│   │   ├── visualization.py   # 可视化
│   │   └── logger.py          # 日志记录
│   ├── train_motion.py         # 运动估计训练脚本
│   ├── train_registration.py  # 配准训练脚本
│   ├── train_segmentation.py  # 分割训练脚本
│   ├── train_end2end.py       # 端到端训练脚本
│   ├── inference.py           # 推理脚本
│   └── evaluate.py            # 评估脚本
├── data/                       # 数据目录
│   ├── raw/                   # 原始数据
│   └── processed/             # 预处理后的数据
├── checkpoints/               # 模型权重
├── notebooks/                 # Jupyter notebooks
└── docs/                      # 文档
```

## 性能指标

在MS-CMR测试集上的性能（预期）：

| 指标 | 数值 |
|------|------|
| Dice系数 | > 0.75 |
| Hausdorff距离 (mm) | < 10 |
| 平均表面距离 (mm) | < 3 |

## 引用

如果本项目对您的研究有帮助，请引用以下论文：

```bibtex
@article{zhuang2019multivariate,
  title={Multivariate mixture model for myocardial segmentation combining multi-source images},
  author={Zhuang, Xiahai},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume={41},
  number={12},
  pages={2933--2946},
  year={2019}
}

@article{balakrishnan2019voxelmorph,
  title={VoxelMorph: a learning framework for deformable medical image registration},
  author={Balakrishnan, Guha and Zhao, Amy and Sabuncu, Mert R and Guttag, John and Dalca, Adrian V},
  journal={IEEE transactions on medical imaging},
  volume={38},
  number={8},
  pages={1788--1800},
  year={2019}
}

@article{yu2020motion,
  title={Motion pyramid networks for accurate and efficient cardiac motion estimation},
  author={Yu, Hanchao and Chen, Xiao and Shi, Humphrey and Chen, Terrence and Huang, Thomas S and Sun, Shanhui},
  journal={arXiv preprint arXiv:2006.15710},
  year={2020}
}
```

## 许可证

本项目采用MIT许可证。详见 [LICENSE](LICENSE) 文件。

## 致谢

- MS-CMR Challenge组织者提供的数据集
- VoxelMorph团队的开源配准框架
- PyTorch社区的支持

## 联系方式

如有问题或建议，请通过以下方式联系：

- 提交GitHub Issue
- 发送邮件至项目维护者

## 更新日志

### v1.0.0 (2025-11-15)

- 初始版本发布
- 实现CMR-LGE配准模块
- 实现心脏运动场估计模块
- 实现心梗分割模块
- 支持MS-CMR、EMIDEC、ACDC数据集
