# 故障排除指南

本文档记录了训练过程中遇到的所有问题及其解决方案。

---

## 已修复的问题

### 1. ❌ `unrecognized arguments: --save_freq 10`

**错误类型**: 参数不匹配

**原因**: Shell脚本传递了Python脚本不支持的参数

**解决方案**:
- 移除 `--save_freq` (自动每10个epoch保存)
- 移除 `--lambda_smooth`, `--motion_checkpoint`, `--registration_checkpoint`, `--alpha_dice`

**相关文档**: `PARAMETERS.md`

---

### 2. ❌ `KeyError: 'cmr'`

**错误类型**: 数据加载器键名不匹配

**原因**: 数据加载器没有返回训练脚本期望的键

**解决方案**:
- 修改 `custom_dataloader_final.py`
- 添加 `'cmr'` 键返回完整CMR序列 (T, H, W)
- 添加 `'mi_label'` 键作为 `'infarct_mask'` 的别名

**相关文档**: `MODEL_INTERFACES.md`

---

### 3. ❌ `ValueError: too many values to unpack (expected 2)`

**错误类型**: 模型返回值解包错误

**原因**: MotionPyramidNet返回3个值,但训练脚本只接收2个

**错误代码**:
```python
motion_field, _ = model(frame1, frame2)  # ❌ 错误
```

**正确代码**:
```python
motion_field, warped_frame, multi_scale_flows = model(frame1, frame2)  # ✅ 正确
```

**相关文档**: `MODEL_INTERFACES.md`

---

### 4. ❌ `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`

**错误类型**: 损失函数没有梯度

**原因**: 损失函数没有使用模型的输出,导致计算图断裂

**错误代码** (train_motion.py):
```python
motion_field, warped_frame, _ = model(frame1, frame2)
loss = nn.functional.l1_loss(frame1, frame2)  # ❌ 没有使用模型输出!
loss.backward()  # 报错: 没有梯度
```

**正确代码**:
```python
motion_field, warped_frame, _ = model(frame1, frame2)
photometric_loss = nn.functional.l1_loss(warped_frame, frame2)  # ✅ 使用模型输出
smoothness_loss = compute_smoothness_loss(motion_field)
loss = photometric_loss + 0.1 * smoothness_loss
loss.backward()  # 正常工作
```

**关键点**:
- 损失函数**必须**使用模型的输出
- 否则PyTorch无法构建计算图
- 反向传播会失败

---

### 5. ⚠️ CMR序列长度不固定 (25帧或40帧)

**问题**: 不同病例的CMR序列长度不同

**解决方案**: 自动检测T维度

**代码**:
```python
# ❌ 错误: 硬编码帧索引
frame1 = cmr[:, 0:1, :, :]
frame2 = cmr[:, 1:2, :, :]  # 假设ES帧是第2帧

# ✅ 正确: 自动检测T维度
T = cmr.shape[1]
frame1 = cmr[:, 0:1, :, :]      # ED帧 (第一帧)
frame2 = cmr[:, T//2:T//2+1, :, :]  # ES帧 (中间帧)
```

**说明**:
- ED (End-Diastolic) 帧通常是第一帧
- ES (End-Systolic) 帧通常在中间位置
- 使用 `T//2` 自动适应不同长度的序列

---

## 损失函数设计

### train_motion.py (运动估计)

```python
# 1. 光度损失 (Photometric Loss)
photometric_loss = nn.functional.l1_loss(warped_frame, frame2)

# 2. 平滑正则化 (Smoothness Regularization)
smoothness_loss = compute_smoothness_loss(motion_field)

# 3. 总损失
loss = photometric_loss + 0.1 * smoothness_loss
```

**目标**: 让warped_frame尽可能接近frame2,同时保持运动场平滑

### train_registration.py (配准)

```python
# 1. 图像相似度损失
similarity_loss = nn.functional.l1_loss(warped_lge, cmr_frame)

# 2. 平滑正则化
smoothness_loss = compute_smoothness_loss(flow)

# 3. 总损失
loss = similarity_loss + 1.0 * smoothness_loss
```

**目标**: 将LGE配准到CMR,同时保持变形场平滑

### train_segmentation.py (分割)

```python
# 使用BCEWithLogitsLoss
criterion = nn.BCEWithLogitsLoss()
loss = criterion(pred, mi_label)

# 评估使用Dice系数
dice = dice_coefficient(torch.sigmoid(pred), mi_label)
```

**目标**: 准确分割心梗区域

---

## 平滑损失函数

```python
def compute_smoothness_loss(flow):
    """计算运动场/变形场的平滑损失"""
    # 计算水平和垂直方向的梯度
    dx = flow[:, :, :, 1:] - flow[:, :, :, :-1]
    dy = flow[:, :, 1:, :] - flow[:, :, :-1, :]
    
    # L1范数
    return torch.mean(torch.abs(dx)) + torch.mean(torch.abs(dy))
```

**作用**:
- 防止运动场/变形场过于复杂
- 鼓励相邻像素的运动/变形相似
- 提高配准的稳定性

---

## 常见问题

### Q1: 训练时loss为NaN

**可能原因**:
1. 学习率过大
2. 梯度爆炸
3. 数据归一化问题

**解决方案**:
```bash
# 降低学习率
--lr 1e-5  # 从1e-4降到1e-5

# 添加梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Q2: GPU内存不足

**解决方案**:
```bash
# 减小batch size
--batch_size 2  # 从4降到2

# 减小图像尺寸 (需要修改代码)
img_size = (128, 128)  # 从(256, 256)降到(128, 128)
```

### Q3: 训练速度慢

**解决方案**:
```bash
# 增加num_workers
--num_workers 8  # 从4增加到8

# 使用混合精度训练 (需要修改代码)
from torch.cuda.amp import autocast, GradScaler
```

### Q4: 数据加载失败

**检查清单**:
- [ ] 数据集路径是否正确
- [ ] LGE标签路径是否为 `labels/lge_original`
- [ ] 所有必需的文件是否存在
- [ ] 文件权限是否正确

---

## 调试技巧

### 1. 打印张量形状

```python
print(f"cmr shape: {cmr.shape}")
print(f"frame1 shape: {frame1.shape}")
print(f"motion_field shape: {motion_field.shape}")
```

### 2. 检查梯度

```python
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm={param.grad.norm()}")
```

### 3. 可视化中间结果

```python
import matplotlib.pyplot as plt

# 保存warped_frame
warped_np = warped_frame[0, 0].cpu().numpy()
plt.imsave('warped_frame.png', warped_np, cmap='gray')
```

---

## 性能优化

### 数据加载优化

```python
# 使用pin_memory加速GPU传输
DataLoader(..., pin_memory=True)

# 使用persistent_workers减少worker重启开销
DataLoader(..., persistent_workers=True)
```

### 训练优化

```python
# 使用torch.compile (PyTorch 2.0+)
model = torch.compile(model)

# 使用混合精度训练
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
```

---

## 更新日期

**最后更新**: 2024-11-17

**修复内容**:
- ✅ 修复所有参数不匹配问题
- ✅ 修复数据加载器键名问题
- ✅ 修复模型返回值解包问题
- ✅ 修复损失函数梯度问题
- ✅ 添加CMR序列长度自动检测
- ✅ 添加平滑正则化损失
