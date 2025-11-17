# 训练脚本参数对照表

本文档列出了所有训练脚本支持的参数,确保shell脚本和Python脚本参数完全一致。

---

## train_motion.py (运动估计模块)

### 支持的参数

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--data_root` | str | ✅ | - | 数据集根目录 |
| `--splits_file` | str | ✅ | - | 数据划分JSON文件 |
| `--batch_size` | int | ❌ | 4 | 批次大小 |
| `--epochs` | int | ❌ | 50 | 训练轮数 |
| `--lr` | float | ❌ | 1e-4 | 学习率 |
| `--weight_decay` | float | ❌ | 1e-5 | 权重衰减 |
| `--num_workers` | int | ❌ | 2 | 数据加载线程数 |
| `--checkpoint_dir` | str | ❌ | checkpoints/motion | 检查点保存目录 |
| `--log_dir` | str | ❌ | logs/motion | 日志保存目录 |
| `--val_freq` | int | ❌ | 5 | 验证频率(epochs) |
| `--patience` | int | ❌ | 10 | 早停耐心值 |
| `--seed` | int | ❌ | 42 | 随机种子 |

### Shell脚本传递的参数 (step2_train_motion.sh)

```bash
python3 src/train_motion.py \
    --data_root "$DATA_ROOT" \
    --splits_file "$SPLITS_FILE" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --num_workers $NUM_WORKERS \
    --checkpoint_dir checkpoints/motion \
    --log_dir logs/motion \
    --val_freq 5
```

### 自动行为
- 每10个epoch自动保存检查点
- 验证时如果性能提升会自动保存最佳模型
- 早停机制自动触发(patience=10)

---

## train_registration.py (配准模块)

### 支持的参数

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--data_root` | str | ✅ | - | 数据集根目录 |
| `--splits_file` | str | ✅ | - | 数据划分JSON文件 |
| `--batch_size` | int | ❌ | 4 | 批次大小 |
| `--epochs` | int | ❌ | 50 | 训练轮数 |
| `--lr` | float | ❌ | 1e-4 | 学习率 |
| `--weight_decay` | float | ❌ | 1e-5 | 权重衰减 |
| `--num_workers` | int | ❌ | 2 | 数据加载线程数 |
| `--checkpoint_dir` | str | ❌ | checkpoints/registration | 检查点保存目录 |
| `--log_dir` | str | ❌ | logs/registration | 日志保存目录 |
| `--val_freq` | int | ❌ | 5 | 验证频率(epochs) |
| `--patience` | int | ❌ | 10 | 早停耐心值 |
| `--seed` | int | ❌ | 42 | 随机种子 |

### Shell脚本传递的参数 (step3_train_registration.sh)

```bash
python3 src/train_registration.py \
    --data_root "$DATA_ROOT" \
    --splits_file "$SPLITS_FILE" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --num_workers $NUM_WORKERS \
    --checkpoint_dir checkpoints/registration \
    --log_dir logs/registration \
    --val_freq 5
```

### 自动行为
- 每10个epoch自动保存检查点
- 验证时如果性能提升会自动保存最佳模型
- 早停机制自动触发(patience=10)
- 平滑损失权重内置在代码中

---

## train_segmentation.py (分割模块)

### 支持的参数

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--data_root` | str | ✅ | - | 数据集根目录 |
| `--splits_file` | str | ✅ | - | 数据划分JSON文件 |
| `--batch_size` | int | ❌ | 4 | 批次大小 |
| `--epochs` | int | ❌ | 50 | 训练轮数 |
| `--lr` | float | ❌ | 1e-4 | 学习率 |
| `--weight_decay` | float | ❌ | 1e-5 | 权重衰减 |
| `--num_workers` | int | ❌ | 2 | 数据加载线程数 |
| `--checkpoint_dir` | str | ❌ | checkpoints/segmentation | 检查点保存目录 |
| `--log_dir` | str | ❌ | logs/segmentation | 日志保存目录 |
| `--val_freq` | int | ❌ | 5 | 验证频率(epochs) |
| `--patience` | int | ❌ | 10 | 早停耐心值 |
| `--seed` | int | ❌ | 42 | 随机种子 |

### Shell脚本传递的参数 (step4_train_segmentation.sh)

```bash
python3 src/train_segmentation.py \
    --data_root "$DATA_ROOT" \
    --splits_file "$SPLITS_FILE" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --num_workers $NUM_WORKERS \
    --checkpoint_dir checkpoints/segmentation \
    --log_dir logs/segmentation \
    --val_freq 5
```

### 自动行为
- 每10个epoch自动保存检查点
- 验证时如果Dice提升会自动保存最佳模型
- 早停机制自动触发(patience=10)
- 使用BCEWithLogitsLoss作为损失函数
- Dice系数用于评估

---

## 已移除的参数

以下参数在之前的版本中存在,但现在已经移除或内置:

### ❌ `--save_freq`
- **原因**: 自动每10个epoch保存,无需手动指定
- **替代**: 内置在训练脚本中

### ❌ `--lambda_smooth`
- **原因**: 平滑损失权重已内置在配准模块中
- **替代**: 使用默认值或修改源代码

### ❌ `--motion_checkpoint`
- **原因**: 当前简化版本的分割模块不需要预训练的运动估计模型
- **替代**: 如需使用,需要修改train_segmentation.py

### ❌ `--registration_checkpoint`
- **原因**: 当前简化版本的分割模块不需要预训练的配准模型
- **替代**: 如需使用,需要修改train_segmentation.py

### ❌ `--alpha_dice`
- **原因**: Dice损失权重已内置在分割模块中
- **替代**: 使用默认值或修改源代码

---

## 如何添加新参数

如果您需要添加新的参数:

1. **修改Python训练脚本**
   ```python
   parser.add_argument('--new_param', type=int, default=10, help='新参数说明')
   ```

2. **修改对应的shell脚本**
   ```bash
   NEW_PARAM=10
   python3 src/train_xxx.py \
       --new_param $NEW_PARAM \
       ...
   ```

3. **更新本文档**

---

## 验证参数

运行以下命令验证参数是否正确:

```bash
# 查看train_motion.py支持的参数
python3 src/train_motion.py --help

# 查看train_registration.py支持的参数
python3 src/train_registration.py --help

# 查看train_segmentation.py支持的参数
python3 src/train_segmentation.py --help
```

---

## 更新日期

**最后更新**: 2024-11-17

**修复内容**:
- ✅ 移除了所有不支持的参数
- ✅ 确保shell脚本和Python脚本参数完全一致
- ✅ 所有训练脚本测试通过
