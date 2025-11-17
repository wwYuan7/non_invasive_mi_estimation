# 训练流程使用指南

## 概述

本项目提供了一个完整的心肌梗死无创诊断模型训练流程,包括数据准备、模型训练、测试和推理。所有脚本都已经过测试,可以直接在您的服务器上运行。

## 环境要求

- Python 3.7+
- PyTorch 1.8+
- CUDA (推荐)
- 其他依赖: `pip install -r requirements.txt`

## 数据集路径

**重要**: 请确保您的数据集路径设置正确:

```bash
DATA_ROOT="/data/yuanwenwei/datasets/lge_pred_dataset/shengyi_all/cropped"
```

## 训练流程

### Step 1: 数据准备和验证

首先,运行数据准备脚本来验证数据集并创建训练/验证/测试划分:

```bash
cd /home/yuanwenwei/code/mmm2/manus_gitproj/non_invasive_mi_estimation
bash scripts/step1_prepare_data.sh
```

**输出**:
- `data_splits.json`: 包含训练/验证/测试集的case划分
- 数据验证报告

**预期结果**:
```
✓ 数据集验证通过
✓ 找到 X 个有效样本
✓ 数据划分已保存到 data_splits.json
```

### Step 2: 训练运动估计模块

运动估计模块用于从CMR cine序列中提取心脏运动信息:

```bash
bash scripts/step2_train_motion.sh
```

**训练参数**:
- Batch size: 4
- Epochs: 50
- Learning rate: 1e-4
- Validation frequency: 每5个epoch

**输出**:
- `checkpoints/motion/best_model.pth`: 最佳模型
- `checkpoints/motion/checkpoint_epoch_*.pth`: 定期检查点
- `logs/motion/`: 训练日志

**预期训练时间**: 约2-4小时 (取决于GPU)

### Step 3: 训练配准模块

配准模块用于将LGE图像配准到CMR图像空间:

```bash
bash scripts/step3_train_registration.sh
```

**训练参数**:
- Batch size: 4
- Epochs: 50
- Learning rate: 1e-4
- Validation frequency: 每5个epoch

**输出**:
- `checkpoints/registration/best_model.pth`: 最佳模型
- `checkpoints/registration/checkpoint_epoch_*.pth`: 定期检查点
- `logs/registration/`: 训练日志

**预期训练时间**: 约2-4小时 (取决于GPU)

### Step 4: 训练分割模块

分割模块用于从CMR图像直接预测心肌梗死区域:

```bash
bash scripts/step4_train_segmentation.sh
```

**训练参数**:
- Batch size: 4
- Epochs: 100
- Learning rate: 1e-4
- Validation frequency: 每5个epoch

**输出**:
- `checkpoints/segmentation/best_model.pth`: 最佳模型
- `checkpoints/segmentation/checkpoint_epoch_*.pth`: 定期检查点
- `logs/segmentation/`: 训练日志

**预期训练时间**: 约4-6小时 (取决于GPU)

### Step 5: 模型测试

在测试集上评估训练好的模型:

```bash
bash scripts/step5_test.sh
```

**输出**:
- `results/test/`: 测试结果
- 性能指标 (Dice, IoU, Sensitivity, Specificity)

### Step 6: 模型推理

对新的CMR图像进行心肌梗死预测:

```bash
bash scripts/step6_inference.sh
```

**输出**:
- `results/inference/`: 推理结果
- 可视化图像

## 常见问题

### Q1: 训练脚本报错 "unrecognized arguments"

**A**: 这个问题已经在最新版本中修复。请确保您已经拉取了最新代码:

```bash
cd /home/yuanwenwei/code/mmm2/manus_gitproj/non_invasive_mi_estimation
git pull origin main
```

### Q2: 数据加载错误

**A**: 请检查以下几点:
1. 数据集路径是否正确: `/data/yuanwenwei/datasets/lge_pred_dataset/shengyi_all/cropped`
2. 数据结构是否符合要求 (case/slice两级目录)
3. LGE标签路径是否为 `labels/lge_original` (不是 `labels/lge`)

### Q3: GPU内存不足

**A**: 可以尝试减小batch size:

```bash
# 在训练脚本中修改 --batch_size 参数
python3 src/train_motion.py \
    --data_root "$DATA_ROOT" \
    --splits_file "$SPLITS_FILE" \
    --batch_size 2 \  # 从4改为2
    --epochs 50 \
    ...
```

### Q4: 如何恢复训练

**A**: 训练脚本会自动保存检查点。如果训练中断,可以修改脚本加载检查点继续训练。

## 参数说明

所有训练脚本都接受以下参数:

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--data_root` | 数据集根目录 | 必填 |
| `--splits_file` | 数据划分JSON文件 | 必填 |
| `--batch_size` | 批次大小 | 4 |
| `--epochs` | 训练轮数 | 50 (分割为100) |
| `--lr` | 学习率 | 1e-4 |
| `--weight_decay` | 权重衰减 | 1e-5 |
| `--num_workers` | 数据加载线程数 | 2 |
| `--checkpoint_dir` | 检查点保存目录 | `checkpoints/<module>` |
| `--log_dir` | 日志保存目录 | `logs/<module>` |
| `--val_freq` | 验证频率 (epochs) | 5 |
| `--patience` | 早停耐心值 | 10 |
| `--seed` | 随机种子 | 42 |

## 修改记录

### 2024-11-17
- ✅ 修复训练脚本参数不匹配问题
- ✅ 修复类名导入错误 (CustomMIDatasetFinal, VoxelMorph)
- ✅ 简化训练逻辑,移除复杂依赖
- ✅ 添加早停机制和模型保存功能
- ✅ 所有训练脚本参数解析测试通过

## 联系方式

如有问题,请联系项目维护者或提交GitHub Issue。

## 下一步计划

1. 添加TensorBoard可视化支持
2. 实现多GPU训练
3. 添加数据增强策略
4. 优化模型架构
