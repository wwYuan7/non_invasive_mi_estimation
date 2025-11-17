#!/bin/bash
# ============================================================================
# 步骤3：训练配准模块
# 功能：训练VoxelMorph进行CMR-LGE配准
# ============================================================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}步骤3：训练配准模块${NC}"
echo -e "${GREEN}========================================${NC}"

# 项目路径和数据集路径
PROJECT_ROOT="/home/yuanwenwei/code/mmm2/manus_gitproj/non_invasive_mi_estimation"
DATA_ROOT="/data/yuanwenwei/datasets/lge_pred_dataset/shengyi_all/cropped"

cd "$PROJECT_ROOT"

# 检查数据集划分文件是否存在
SPLITS_FILE="data/splits/dataset_splits.json"
if [ ! -f "$SPLITS_FILE" ]; then
    echo -e "${RED}错误: 数据集划分文件不存在！${NC}"
    echo -e "${YELLOW}请先运行: ./scripts/step1_prepare_data.sh${NC}"
    exit 1
fi

# 创建checkpoints目录
mkdir -p checkpoints/registration

# 训练参数
EPOCHS=150
BATCH_SIZE=8
LEARNING_RATE=1e-4
NUM_WORKERS=4
LAMBDA_SMOOTH=1.0

echo -e "${YELLOW}训练参数:${NC}"
echo -e "  Epochs: ${EPOCHS}"
echo -e "  Batch Size: ${BATCH_SIZE}"
echo -e "  Learning Rate: ${LEARNING_RATE}"
echo -e "  Num Workers: ${NUM_WORKERS}"
echo -e "  Lambda Smooth: ${LAMBDA_SMOOTH}"

echo -e "\n${GREEN}开始训练配准模块...${NC}"
echo -e "${YELLOW}提示: 训练过程可能需要数小时，请耐心等待${NC}"
echo -e "${YELLOW}可以使用 Ctrl+C 中断训练${NC}\n"

# 训练命令
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

if [ $? -ne 0 ]; then
    echo -e "${RED}训练失败！${NC}"
    exit 1
fi

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}步骤3完成！${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "${YELLOW}训练好的模型保存在: checkpoints/registration/${NC}"
echo -e "${YELLOW}下一步: 运行 ./scripts/step4_train_segmentation.sh${NC}"
