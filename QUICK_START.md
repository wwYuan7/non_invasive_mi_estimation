# 快速开始指南

## 最新更新 (2024-11-17)

✅ **所有训练脚本参数不匹配问题已修复!**

现在您可以直接运行训练脚本,无需担心参数错误。

## 快速开始

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

## 修复内容

### 问题
之前的训练脚本存在参数不匹配问题:
- Shell脚本传递: `--data_root`, `--splits_file`, `--checkpoint_dir`, `--log_dir`, `--val_freq`
- Python脚本不接受这些参数,导致 "unrecognized arguments" 错误

### 解决方案
1. **重写所有训练脚本** (train_motion.py, train_registration.py, train_segmentation.py)
   - 添加所有必需的命令行参数
   - 使用正确的数据加载器类 (CustomMIDatasetFinal)
   - 使用正确的模型类 (VoxelMorph, MotionPyramidNet, AttentionUNet)
   
2. **简化训练逻辑**
   - 移除复杂的依赖和工具类
   - 直接在训练脚本中实现核心功能
   - 添加早停机制和模型保存

3. **参数验证**
   - 所有训练脚本都能正确解析参数
   - 与shell脚本完全兼容

## 数据集要求

确保您的数据集路径正确:
```
/data/yuanwenwei/datasets/lge_pred_dataset/shengyi_all/cropped/
├── images/
│   ├── cmr/
│   │   └── case001/
│   │       ├── slice_01.nii.gz
│   │       └── ...
│   └── lge/
│       └── case001/
│           ├── slice_01.nii.gz
│           └── ...
└── labels/
    ├── cmr/
    │   └── cmr_Myo_mask/
    │       └── case001/...
    └── lge_original/  # 注意: 是 lge_original 不是 lge
        ├── lge_MI_labels/
        │   └── case001/...
        └── lge_Myo_labels/
            └── case001/...
```

## 详细文档

更多详细信息请参阅:
- [完整训练指南](TRAINING_GUIDE.md)
- [项目README](README.md)

## 常见问题

### Q: 如何检查训练是否正常运行?

A: 训练开始后,您应该看到类似以下的输出:
```
Using device: cuda
Train cases: XX
Val cases: XX
Train samples: XXX
Val samples: XXX
Model parameters: X,XXX,XXX

Epoch 1/50
--------------------------------------------------
Epoch 1: 100%|████████| XX/XX [XX:XX<00:00, X.XXit/s, loss=X.XXXX]
Train Loss: X.XXXX
```

### Q: 训练需要多长时间?

A: 
- 运动估计模块: 约2-4小时
- 配准模块: 约2-4小时  
- 分割模块: 约4-6小时

(具体时间取决于GPU性能和数据集大小)

### Q: 如何监控训练进度?

A: 检查点和日志会保存在:
- `checkpoints/<module>/`: 模型检查点
- `logs/<module>/`: 训练日志

## 技术支持

如有问题,请:
1. 检查 [TRAINING_GUIDE.md](TRAINING_GUIDE.md) 中的常见问题
2. 运行 `bash scripts/verify_installation.sh` 验证环境
3. 提交GitHub Issue
