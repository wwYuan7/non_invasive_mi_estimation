# 断点续训使用指南

## 🎯 功能说明

新的 `run_all.sh` 脚本支持三种训练模式:

### 1. 从头开始训练
- 清除所有已训练的模型和进度
- 从步骤1开始重新训练
- **警告**: 会删除所有checkpoint!

### 2. 断点续训
- 从上次中断的地方继续训练
- 保留已完成的步骤
- 适合训练中断后恢复

### 3. 智能跳过
- 自动检测已完成的模型
- 跳过已训练的步骤
- 只训练缺失的模型

---

## 🚀 使用方法

### 启动训练

```bash
cd /home/yuanwenwei/code/mmm2/manus_gitproj/non_invasive_mi_estimation
bash scripts/run_all.sh
```

### 交互式选择

脚本会显示:

```
========================================
非侵入性心梗估计 - 完整训练流程
========================================

检测到上次训练进度: step2_motion_trained

检查已训练的模型:
  ✓ 运动估计模型
  ✗ 配准模型
  ✗ 分割模型

请选择训练模式:
  1) 从头开始训练 (清除所有进度和模型)
  2) 断点续训 (从上次中断处继续)
  3) 智能跳过 (自动跳过已完成的步骤)
  4) 退出

请输入选项 [1-4]:
```

---

## 📊 进度跟踪

### 进度文件

训练进度保存在: `.training_progress`

可能的进度状态:
- `step0_not_started` - 未开始
- `step1_data_prepared` - 数据已准备
- `step2_motion_trained` - 运动估计已训练
- `step3_registration_trained` - 配准已训练
- `step4_segmentation_trained` - 分割已训练
- `all_completed` - 全部完成

### 查看当前进度

```bash
cat .training_progress
```

### 手动重置进度

```bash
echo "step0_not_started" > .training_progress
```

---

## 🔄 使用场景

### 场景1: 首次训练

```bash
bash scripts/run_all.sh
# 选择: 1 (从头开始训练)
```

### 场景2: 训练中断后继续

```bash
bash scripts/run_all.sh
# 选择: 2 (断点续训)
```

脚本会自动从上次中断的步骤继续。

### 场景3: 部分模型已训练

例如，您已经手动训练了运动估计模型:

```bash
bash scripts/run_all.sh
# 选择: 3 (智能跳过)
```

脚本会:
- ✓ 跳过运动估计 (已存在)
- ✗ 训练配准模块
- ✗ 训练分割模块

### 场景4: 重新训练某个模块

如果想重新训练配准模块:

```bash
# 删除配准模型
rm -rf checkpoints/registration

# 运行脚本
bash scripts/run_all.sh
# 选择: 3 (智能跳过)
```

---

## 📁 模型文件位置

训练好的模型保存在:

```
checkpoints/
├── motion/
│   └── best_model.pth          # 运动估计模型
├── registration/
│   └── best_model.pth          # 配准模型
└── segmentation/
    └── best_model.pth          # 分割模型
```

---

## ⚠️ 注意事项

### 1. 选择"从头开始训练"时

会删除:
- `checkpoints/motion/`
- `checkpoints/registration/`
- `checkpoints/segmentation/`
- `logs/motion/`
- `logs/registration/`
- `logs/segmentation/`
- `.training_progress`

**请确认后再继续!**

### 2. 断点续训的限制

- 只能从完整步骤的边界继续
- 不能从某个epoch中间继续
- 如果想从epoch中间继续，需要修改Python训练脚本

### 3. 智能跳过的逻辑

脚本通过检查 `best_model.pth` 文件判断模型是否已训练:

```bash
checkpoints/motion/best_model.pth        # 存在 → 跳过
checkpoints/registration/best_model.pth  # 不存在 → 训练
checkpoints/segmentation/best_model.pth  # 不存在 → 训练
```

---

## 🛠️ 高级用法

### 查看训练状态

```bash
# 检查进度
cat .training_progress

# 检查模型文件
ls -lh checkpoints/*/best_model.pth

# 检查日志
tail -f logs/motion/train.log
```

### 清除特定模块

```bash
# 只清除运动估计模块
rm -rf checkpoints/motion logs/motion

# 只清除配准模块
rm -rf checkpoints/registration logs/registration

# 只清除分割模块
rm -rf checkpoints/segmentation logs/segmentation
```

### 完全重置

```bash
# 清除所有训练结果
rm -rf checkpoints logs .training_progress data/splits
```

---

## 📝 示例流程

### 完整训练流程

```bash
# 1. 首次训练
bash scripts/run_all.sh
# 选择: 1 (从头开始)

# 训练进行中...
# 假设在步骤2完成后中断 (Ctrl+C)

# 2. 继续训练
bash scripts/run_all.sh
# 选择: 2 (断点续训)
# 会从步骤3开始

# 训练继续...
# 假设步骤3也完成了

# 3. 最后完成步骤4
bash scripts/run_all.sh
# 选择: 2 (断点续训)
# 只运行步骤4

# 全部完成!
```

---

## 🔍 故障排除

### 问题1: 进度文件损坏

```bash
# 手动设置进度
echo "step2_motion_trained" > .training_progress
```

### 问题2: 模型文件存在但损坏

```bash
# 删除损坏的模型
rm checkpoints/motion/best_model.pth

# 使用智能跳过重新训练
bash scripts/run_all.sh
# 选择: 3
```

### 问题3: 想从某个步骤重新开始

```bash
# 设置进度到前一步
echo "step1_data_prepared" > .training_progress

# 删除对应的模型
rm -rf checkpoints/motion

# 断点续训
bash scripts/run_all.sh
# 选择: 2
```

---

## 📚 相关文档

- **QUICK_START.md** - 快速开始指南
- **TRAINING_GUIDE.md** - 详细训练指南
- **TROUBLESHOOTING.md** - 故障排除
- **PARAMETERS.md** - 参数说明

---

## 🎉 总结

新的 `run_all.sh` 脚本提供了灵活的训练控制:

- ✅ 支持断点续训
- ✅ 自动检测进度
- ✅ 智能跳过已完成步骤
- ✅ 交互式选择
- ✅ 清晰的状态显示

**建议**: 首次训练使用"从头开始"，后续使用"断点续训"或"智能跳过"。
