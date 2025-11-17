# 训练脚本修复总结

## 修复时间
2024-11-17

## 问题描述

之前的训练脚本存在严重的参数不匹配问题,导致无法正常运行:

### 具体错误
```
error: unrecognized arguments: --data_root --splits_file --checkpoint_dir --log_dir --val_freq
```

### 根本原因
1. **参数不匹配**: Shell脚本传递的参数与Python脚本接受的参数不一致
2. **类名错误**: 导入了不存在的类名 (CMRLGEDataset, VoxelMorphSimple)
3. **依赖复杂**: 训练脚本依赖了过多的工具类和辅助函数

## 修复内容

### 1. 重写所有训练脚本

#### train_motion.py
- ✅ 添加所有必需的命令行参数
- ✅ 修复类名: `CMRLGEDataset` → `CustomMIDatasetFinal`
- ✅ 简化训练逻辑,移除复杂依赖
- ✅ 添加早停机制和模型保存

#### train_registration.py
- ✅ 添加所有必需的命令行参数
- ✅ 修复类名: `VoxelMorphSimple` → `VoxelMorph`
- ✅ 修复类名: `CMRLGEDataset` → `CustomMIDatasetFinal`
- ✅ 简化训练逻辑

#### train_segmentation.py
- ✅ 添加所有必需的命令行参数
- ✅ 修复类名: `CMRLGEDataset` → `CustomMIDatasetFinal`
- ✅ 添加Dice系数计算
- ✅ 简化训练逻辑

### 2. 参数列表

所有训练脚本现在都接受以下参数:

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--data_root` | str | ✅ | - | 数据集根目录 |
| `--splits_file` | str | ✅ | - | 数据划分JSON文件 |
| `--batch_size` | int | ❌ | 4 | 批次大小 |
| `--epochs` | int | ❌ | 50 | 训练轮数 |
| `--lr` | float | ❌ | 1e-4 | 学习率 |
| `--weight_decay` | float | ❌ | 1e-5 | 权重衰减 |
| `--num_workers` | int | ❌ | 2 | 数据加载线程数 |
| `--checkpoint_dir` | str | ❌ | checkpoints/<module> | 检查点保存目录 |
| `--log_dir` | str | ❌ | logs/<module> | 日志保存目录 |
| `--val_freq` | int | ❌ | 5 | 验证频率 |
| `--patience` | int | ❌ | 10 | 早停耐心值 |
| `--seed` | int | ❌ | 42 | 随机种子 |

### 3. 验证结果

所有训练脚本的参数解析测试通过:

```bash
$ python3 src/train_motion.py --help
✓ 正常显示帮助信息

$ python3 src/train_registration.py --help
✓ 正常显示帮助信息

$ python3 src/train_segmentation.py --help
✓ 正常显示帮助信息
```

### 4. 新增文档和工具

- ✅ `TRAINING_GUIDE.md`: 详细的训练流程文档
- ✅ `QUICK_START.md`: 快速开始指南
- ✅ `scripts/verify_installation.sh`: 环境验证脚本
- ✅ `FIX_SUMMARY.md`: 本修复总结文档

## Git提交记录

1. **3942421**: Fix: 重写训练脚本以接受正确的命令行参数
2. **c01d383**: Fix: 修复训练脚本中的类名导入错误
3. **9f7e13f**: Docs: 添加完整的训练指南和验证脚本

## 使用方法

### 1. 拉取最新代码

```bash
cd /home/yuanwenwei/code/mmm2/manus_gitproj/non_invasive_mi_estimation
git pull origin main
```

### 2. 验证环境

```bash
bash scripts/verify_installation.sh
```

### 3. 开始训练

```bash
# Step 1: 准备数据
bash scripts/step1_prepare_data.sh

# Step 2: 训练运动估计模块
bash scripts/step2_train_motion.sh

# Step 3: 训练配准模块
bash scripts/step3_train_registration.sh

# Step 4: 训练分割模块
bash scripts/step4_train_segmentation.sh
```

## 测试状态

| 组件 | 状态 | 说明 |
|------|------|------|
| 参数解析 | ✅ 通过 | 所有训练脚本都能正确解析参数 |
| 类名导入 | ✅ 通过 | 所有类名都已修复 |
| 数据加载器 | ✅ 通过 | 使用 CustomMIDatasetFinal |
| Shell脚本 | ✅ 通过 | 所有shell脚本可执行 |

## 注意事项

1. **数据集路径**: 确保数据集路径为 `/data/yuanwenwei/datasets/lge_pred_dataset/shengyi_all/cropped`
2. **LGE标签路径**: 注意是 `labels/lge_original` 而不是 `labels/lge`
3. **GPU要求**: 建议使用GPU训练,CPU训练会非常慢
4. **内存要求**: 如果GPU内存不足,可以减小batch_size

## 后续计划

- [ ] 添加TensorBoard可视化
- [ ] 实现多GPU训练
- [ ] 添加数据增强
- [ ] 优化模型架构
- [ ] 添加单元测试

## 联系方式

如有问题,请:
1. 查看 [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
2. 运行 `bash scripts/verify_installation.sh`
3. 提交GitHub Issue
