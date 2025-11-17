# 全流程运行脚本使用说明

本文档详细说明了如何使用`scripts/`目录下的脚本来运行整个无创心梗估计项目的流程，从数据准备到模型训练、测试和推理。

---

## 1. 环境准备

在运行任何脚本之前，请确保您已完成以下准备工作：

### 1.1. 克隆项目

将项目克隆到您的服务器指定位置：

```bash
git clone https://github.com/wwYuan7/non_invasive_mi_estimation.git /home/yuanwenwei/code/mmm2/manus_gitproj/non_invasive_mi_estimation
```

### 1.2. 准备数据集

确保您的数据集位于以下路径：

```
/data/yuanwenwei/datasets/lge_pred_dataset/shengyi_all/cropped/
```

并且数据集结构符合要求：

```
cropped/
├── images/
│   ├── cmr/<case_id>/<slice_id>.nii.gz
│   └── lge/<case_id>/<slice_id>.nii.gz
└── labels/
    ├── cmr/cmr_Myo_mask/<case_id>.nii.gz
    └── lge/
        ├── lge_MI_labels/<case_id>.nii.gz
        └── lge_Myo_labels/<case_id>.nii.gz
```

### 1.3. 安装依赖

进入项目根目录并安装所有Python依赖：

```bash
cd /home/yuanwenwei/code/mmm2/manus_gitproj/non_invasive_mi_estimation
pip install -r requirements.txt
```

### 1.4. 检查脚本路径

所有脚本中的项目路径和数据路径已根据您的要求硬编码。如果您的路径有变动，请手动修改`scripts/`目录下的所有`.sh`文件中的`PROJECT_ROOT`和`DATA_ROOT`变量。

---

## 2. 运行方式

我们提供了两种运行方式：**一键全流程运行**和**分步运行**。

### 2.1. 一键全流程运行（推荐）

如果您想自动执行从数据准备到模型训练和测试的所有步骤，可以直接运行`run_all.sh`脚本。

**警告**：此过程将非常耗时（可能需要数天），请确保有足够的计算资源和时间。

```bash
cd /home/yuanwenwei/code/mmm2/manus_gitproj/non_invasive_mi_estimation

# 运行一键脚本
bash scripts/run_all.sh
```

脚本会提示您确认，输入`yes`后将开始执行。所有日志将保存在`logs/full_run_YYYYMMDD_HHMMSS/`目录下。

### 2.2. 分步运行（灵活，推荐用于调试）

您可以按照顺序手动执行每个步骤的脚本。这允许您在每个步骤之间检查结果，或者在中断后从特定步骤继续。

#### **步骤1：数据准备**

此脚本将验证您的数据集结构，并生成训练、验证、测试集的划分文件。

```bash
bash scripts/step1_prepare_data.sh
```

- **输出**：`data/splits/dataset_splits.json`

#### **步骤2：训练运动估计模块**

训练用于提取心脏运动场的Motion Pyramid Networks。

```bash
bash scripts/step2_train_motion.sh
```

- **输出**：`checkpoints/motion/best_model.pth`

#### **步骤3：训练配准模块**

训练用于CMR-LGE配准的VoxelMorph网络。

```bash
bash scripts/step3_train_registration.sh
```

- **输出**：`checkpoints/registration/best_model.pth`

#### **步骤4：训练分割模块**

训练用于最终心梗分割的Attention U-Net。这是最耗时的一步。

```bash
bash scripts/step4_train_segmentation.sh
```

- **输出**：`checkpoints/segmentation/best_model.pth`

#### **步骤5：测试模型**

在测试集上评估训练好的模型，并计算性能指标。

```bash
bash scripts/step5_test.sh
```

- **输出**：
  - `results/test_results/`：保存每个测试病例的预测结果（.nii.gz）
  - `results/test_results/test_results.json`：包含所有性能指标的JSON文件

#### **步骤6：单例推理**

对单个病例进行推理，并保存中间结果和最终分割结果。

```bash
# 示例：对case001进行推理
bash scripts/step6_inference.sh case001
```

- **输出**：`results/inference/case001/`
  - `mi_prediction.nii.gz`：最终心梗分割结果
  - `motion_field.nii.gz`：运动场
  - `strain_map.nii.gz`：应变图
  - `deformation_field.nii.gz`：配准形变场

---

## 3. 日志与结果

- **训练日志**：所有训练过程的详细输出都将保存在`logs/`目录下，每个模块一个子目录。
- **TensorBoard日志**：训练过程中的损失和指标变化会保存在`logs/`目录下的TensorBoard文件中。您可以使用以下命令查看：
  ```bash
  tensorboard --logdir logs
  ```
- **模型权重**：所有训练好的最佳模型权重都保存在`checkpoints/`目录下。
- **测试与推理结果**：所有测试和推理的输出都保存在`results/`目录下。

---

## 4. 常见问题

- **脚本权限问题**：如果遇到`Permission denied`错误，请确保所有脚本都有执行权限：
  ```bash
  chmod +x scripts/*.sh
  ```

- **路径错误**：如果遇到`No such file or directory`错误，请检查脚本中的`PROJECT_ROOT`和`DATA_ROOT`路径是否正确。

- **训练中断**：如果训练意外中断，您可以从失败的步骤重新运行对应的脚本。脚本会自动加载已有的最佳权重（如果存在）并尝试继续训练。

- **GPU内存不足**：如果遇到`CUDA out of memory`错误，请尝试减小训练脚本中的`BATCH_SIZE`参数。

---

祝您使用愉快！如有任何问题，欢迎随时提出！
