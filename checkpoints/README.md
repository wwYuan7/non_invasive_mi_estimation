# 模型权重说明

## 权重文件生成

由于模型权重文件较大（总计约170MB），未直接包含在Git仓库中。您可以通过以下两种方式获取模型权重：

### 方式1：生成初始化权重（推荐用于开发测试）

在项目根目录运行以下命令生成随机初始化的权重：

```bash
python create_init_weights.py
```

这将在`checkpoints/`目录下生成三个模块的初始权重：
- `motion_estimation/init_motion_model.pth` - 运动估计模块（约5MB）
- `registration/init_registration_model.pth` - 配准模块（约8MB）
- `segmentation/init_segmentation_model.pth` - 分割模块（约134MB）

**注意**：这些是随机初始化的权重，需要在真实数据上训练才能获得良好性能。

### 方式2：训练模型获取权重

使用真实的CMR/LGE数据训练模型以获得最佳性能：

#### 步骤1：准备数据

将您的数据组织成以下格式：
```
data/
├── train/
│   ├── cmr/          # CMR序列（NIfTI格式）
│   ├── lge/          # LGE图像（NIfTI格式）
│   └── masks/        # 心肌和心梗掩模
├── val/
└── test/
```

#### 步骤2：训练各模块

```bash
# 1. 训练运动估计模块（约2-3小时，GPU）
python src/train_motion.py \
    --epochs 100 \
    --batch_size 8 \
    --lr 1e-4 \
    --save_dir checkpoints

# 2. 训练配准模块（约2-3小时，GPU）
python src/train_registration.py \
    --epochs 100 \
    --batch_size 8 \
    --lr 1e-4 \
    --save_dir checkpoints

# 3. 训练分割模块（约4-5小时，GPU）
python src/train_segmentation.py \
    --epochs 100 \
    --batch_size 4 \
    --lr 1e-4 \
    --motion_checkpoint checkpoints/motion_estimation/best_motion_model.pth \
    --save_dir checkpoints
```

训练完成后，最佳权重将保存为：
- `motion_estimation/best_motion_model.pth`
- `registration/best_registration_model.pth`
- `segmentation/best_segmentation_model.pth`

## 权重文件结构

每个权重文件（.pth）包含以下内容：

```python
{
    'epoch': int,                    # 训练轮数
    'model_state_dict': OrderedDict, # 模型参数
    'optimizer_state_dict': OrderedDict,  # 优化器状态（如果保存）
    'best_metric': float,            # 最佳指标（loss或dice）
    'note': str                      # 备注信息
}
```

## 使用预训练权重

在推理时加载权重：

```python
import torch
from src.models.motion_pyramid import MotionPyramidNet

# 加载运动估计模型
model = MotionPyramidNet(img_size=(256, 256))
checkpoint = torch.load('checkpoints/motion_estimation/best_motion_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

或使用提供的推理脚本：

```bash
python src/inference.py \
    --input path/to/cmr.nii.gz \
    --myocardium_mask path/to/mask.nii.gz \
    --output path/to/output.nii.gz \
    --checkpoint checkpoints/segmentation/best_segmentation_model.pth
```

## 模型参数统计

| 模块 | 参数量 | 文件大小 |
|------|--------|----------|
| 运动估计 | 1,137,638 | ~5MB |
| 配准 | 1,956,674 | ~8MB |
| 分割 | 34,879,149 | ~134MB |
| **总计** | **37,973,461** | **~147MB** |

## 推荐数据集

训练模型时推荐使用以下公开数据集：

1. **MS-CMR 2019** - 多序列CMR数据集
   - 包含45例配对的CMR和LGE数据
   - 官网：https://zmiclab.github.io/zxh/0/mscmrseg19/

2. **EMIDEC** - 心梗数据集
   - 包含150例DE-MRI数据
   - 官网：https://emidec.com/

3. **ACDC** - 自动心脏诊断挑战
   - 大量cine-MRI数据
   - 官网：https://www.creatis.insa-lyon.fr/Challenge/acdc/

## 常见问题

**Q: 为什么不直接提供预训练权重？**

A: 由于以下原因：
1. 文件较大，不适合直接放在Git仓库
2. 需要使用特定的医疗数据集训练，涉及数据使用协议
3. 不同应用场景可能需要在特定数据上微调

**Q: 初始化权重能直接用于推理吗？**

A: 不建议。初始化权重是随机的，需要在真实数据上训练后才能获得有意义的结果。

**Q: 训练需要多长时间？**

A: 在单个GPU（如NVIDIA RTX 3090）上：
- 运动估计：2-3小时（100 epochs）
- 配准：2-3小时（100 epochs）
- 分割：4-5小时（100 epochs）

总计约8-11小时。使用更强的GPU或多GPU可以加速训练。

**Q: 可以使用CPU训练吗？**

A: 可以，但会非常慢（约10-20倍）。强烈建议使用GPU训练。
