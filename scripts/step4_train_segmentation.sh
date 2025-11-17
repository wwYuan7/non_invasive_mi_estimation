#!/bin/bash
# ============================================================================
# 步骤4：训练分割模块
# 功能：训练Attention U-Net进行心梗分割
# ============================================================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}步骤4：训练分割模块${NC}"
echo -e "${GREEN}========================================${NC}"

# 项目路径和数据集路径
PROJECT_ROOT="/home/yuanwenwei/code/mmm2/manus_gitproj/non_invasive_mi_estimation"
DATA_ROOT="/data/yuanwenwei/datasets/lge_pred_dataset/shengyi_all/cropped"

cd "$PROJECT_ROOT"

# 检查数据集划分文件
SPLITS_FILE="data/splits/dataset_splits.json"
if [ ! -f "$SPLITS_FILE" ]; then
    echo -e "${RED}错误: 数据集划分文件不存在！${NC}"
    echo -e "${YELLOW}请先运行: ./scripts/step1_prepare_data.sh${NC}"
    exit 1
fi

# 检查运动估计模块权重
MOTION_CHECKPOINT="checkpoints/motion/best_model.pth"
if [ ! -f "$MOTION_CHECKPOINT" ]; then
    echo -e "${RED}错误: 运动估计模块权重不存在！${NC}"
    echo -e "${YELLOW}请先运行: ./scripts/step2_train_motion.sh${NC}"
    exit 1
fi

# 检查配准模块权重
REG_CHECKPOINT="checkpoints/registration/best_model.pth"
if [ ! -f "$REG_CHECKPOINT" ]; then
    echo -e "${RED}错误: 配准模块权重不存在！${NC}"
    echo -e "${YELLOW}请先运行: ./scripts/step3_train_registration.sh${NC}"
    exit 1
fi

# 创建checkpoints目录
mkdir -p checkpoints/segmentation

# 训练参数
EPOCHS=200
BATCH_SIZE=8
LEARNING_RATE=1e-4
NUM_WORKERS=4
ALPHA_DICE=0.7

echo -e "${YELLOW}训练参数:${NC}"
echo -e "  Epochs: ${EPOCHS}"
echo -e "  Batch Size: ${BATCH_SIZE}"
echo -e "  Learning Rate: ${LEARNING_RATE}"
echo -e "  Num Workers: ${NUM_WORKERS}"
echo -e "  Alpha Dice: ${ALPHA_DICE}"
echo -e "  Motion Checkpoint: ${MOTION_CHECKPOINT}"
echo -e "  Registration Checkpoint: ${REG_CHECKPOINT}"

echo -e "\n${GREEN}开始训练分割模块...${NC}"
echo -e "${YELLOW}提示: 这是最大的模型，训练时间最长${NC}"
echo -e "${YELLOW}可以使用 Ctrl+C 中断训练${NC}\n"

# 训练命令
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

if [ $? -ne 0 ]; then
    echo -e "${RED}训练失败！${NC}"
    exit 1
fi

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}步骤4完成！${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "${YELLOW}训练好的模型保存在: checkpoints/segmentation/${NC}"
echo -e "${YELLOW}下一步: 运行 ./scripts/step5_test.sh 进行测试${NC}"
