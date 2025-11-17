# 模型接口文档

本文档详细说明所有模型的输入输出接口,确保训练脚本正确调用模型。

---

## MotionPyramidNet (运动估计模块)

**文件**: `src/models/motion_pyramid.py`

### Forward方法

```python
def forward(self, frame1, frame2):
    """
    Args:
        frame1: 第一帧 (ED) (B, 1, H, W)
        frame2: 第二帧 (ES) (B, 1, H, W)
    
    Returns:
        final_flow: 运动场 (B, 2, H, W)
        warped_frame1: 扭曲后的第一帧 (B, 1, H, W)
        multi_scale_flows: 多尺度运动场列表
    """
```

### 训练脚本调用

```python
# train_motion.py
motion_field, warped_frame, multi_scale_flows = model(frame1, frame2)
```

### 注意事项
- 返回**3个值**,必须全部接收
- `final_flow`是最终的运动场
- `warped_frame1`是运动补偿后的第一帧
- `multi_scale_flows`是多尺度运动场列表(用于多尺度损失)

---

## VoxelMorph (配准模块)

**文件**: `src/models/voxelmorph_simple.py`

### Forward方法

```python
def forward(self, moving, fixed):
    """
    Args:
        moving: 浮动图像 (B, 1, H, W)
        fixed: 固定图像 (B, 1, H, W)
    
    Returns:
        warped_moving: 配准后的浮动图像 (B, 1, H, W)
        flow: 变形场 (B, 2, H, W)
    """
```

### 训练脚本调用

```python
# train_registration.py
warped_lge, flow = model(lge, cmr_frame)
```

### 注意事项
- 返回**2个值**
- `warped_moving`是配准后的图像
- `flow`是变形场

---

## AttentionUNet (分割模块)

**文件**: `src/models/attention_unet.py`

### Forward方法

```python
def forward(self, x):
    """
    Args:
        x: 输入图像 (B, C, H, W)
    
    Returns:
        output: 分割输出 (B, 1, H, W) - logits (未经sigmoid)
    """
```

### 训练脚本调用

```python
# train_segmentation.py
pred = model(cmr_frame)  # 返回1个值
pred = pred.squeeze(1)   # (B, H, W)
```

### 注意事项
- 返回**1个值**
- 输出是logits,需要经过sigmoid才能得到概率
- 使用BCEWithLogitsLoss时不需要手动sigmoid

---

## 数据加载器接口

**文件**: `src/data/custom_dataloader_final.py`

### 返回的字典键

```python
sample = {
    'case_id': str,                    # 样本ID
    'cmr': Tensor,                     # 完整CMR序列 (T, H, W)
    'cine_ed': Tensor,                 # ED帧 (1, H, W)
    'cine_es': Tensor,                 # ES帧 (1, H, W)
    'myocardium_mask': Tensor,         # 心肌掩模 (H, W)
    'lge': Tensor,                     # LGE图像 (1, H, W) [仅mode='registration'或'segmentation']
    'lge_myocardium_mask': Tensor,     # LGE心肌掩模 (H, W) [仅mode='registration'或'segmentation']
    'infarct_mask': Tensor,            # 心梗标签 (H, W) [仅mode='registration'或'segmentation']
    'mi_label': Tensor,                # 心梗标签别名 (H, W) [仅mode='registration'或'segmentation']
}
```

### 训练脚本使用

```python
# train_motion.py
cmr = batch['cmr']  # (B, T, H, W)

# train_registration.py
cmr = batch['cmr']  # (B, T, H, W)
lge = batch['lge']  # (B, 1, H, W)

# train_segmentation.py
cmr = batch['cmr']      # (B, T, H, W)
mi_label = batch['mi_label']  # (B, H, W)
```

---

## 常见错误

### 1. ValueError: too many values to unpack

**错误代码**:
```python
motion_field, _ = model(frame1, frame2)  # ❌ 错误: MotionPyramidNet返回3个值
```

**正确代码**:
```python
motion_field, warped_frame, multi_scale_flows = model(frame1, frame2)  # ✅ 正确
```

### 2. KeyError: 'cmr'

**错误原因**: 数据加载器没有返回'cmr'键

**解决方案**: 确保使用最新版本的`custom_dataloader_final.py`,它已经添加了'cmr'键

### 3. KeyError: 'mi_label'

**错误原因**: 数据加载器没有返回'mi_label'键

**解决方案**: 确保使用最新版本的`custom_dataloader_final.py`,它已经添加了'mi_label'作为'infarct_mask'的别名

---

## 测试代码

### 测试MotionPyramidNet

```python
import torch
from src.models.motion_pyramid import MotionPyramidNet

model = MotionPyramidNet(img_size=(256, 256))
frame1 = torch.randn(2, 1, 256, 256)
frame2 = torch.randn(2, 1, 256, 256)

final_flow, warped_frame, multi_scale_flows = model(frame1, frame2)
print(f"final_flow shape: {final_flow.shape}")  # (2, 2, 256, 256)
print(f"warped_frame shape: {warped_frame.shape}")  # (2, 1, 256, 256)
print(f"Number of scales: {len(multi_scale_flows)}")
```

### 测试VoxelMorph

```python
import torch
from src.models.voxelmorph_simple import VoxelMorph

model = VoxelMorph(img_size=(256, 256))
moving = torch.randn(2, 1, 256, 256)
fixed = torch.randn(2, 1, 256, 256)

warped_moving, flow = model(moving, fixed)
print(f"warped_moving shape: {warped_moving.shape}")  # (2, 1, 256, 256)
print(f"flow shape: {flow.shape}")  # (2, 2, 256, 256)
```

### 测试AttentionUNet

```python
import torch
from src.models.attention_unet import AttentionUNet

model = AttentionUNet(in_channels=1, out_channels=1)
x = torch.randn(2, 1, 256, 256)

output = model(x)
print(f"output shape: {output.shape}")  # (2, 1, 256, 256)
```

---

## 更新日期

**最后更新**: 2024-11-17

**修复内容**:
- ✅ 修复MotionPyramidNet返回值解包错误
- ✅ 添加完整的模型接口文档
- ✅ 提供测试代码示例
