#!/bin/bash
# ============================================================================
# 步骤6：单例推理
# 功能：对单个病例进行心梗分割预测
# 使用方法: ./scripts/step6_inference.sh <case_id>
# 示例: ./scripts/step6_inference.sh case001
# ============================================================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}步骤6：单例推理${NC}"
echo -e "${GREEN}========================================${NC}"

# 检查参数
if [ $# -eq 0 ]; then
    echo -e "${RED}错误: 请提供病例ID${NC}"
    echo -e "${YELLOW}使用方法: ./scripts/step6_inference.sh <case_id>${NC}"
    echo -e "${YELLOW}示例: ./scripts/step6_inference.sh case001${NC}"
    exit 1
fi

CASE_ID=$1

# 项目路径和数据集路径
PROJECT_ROOT="/home/yuanwenwei/code/mmm2/manus_gitproj/non_invasive_mi_estimation"
DATA_ROOT="/data/yuanwenwei/datasets/lge_pred_dataset/shengyi_all/cropped"

cd "$PROJECT_ROOT"

# 检查病例是否存在
CMR_DIR="${DATA_ROOT}/images/cmr/${CASE_ID}"
LGE_DIR="${DATA_ROOT}/images/lge/${CASE_ID}"

if [ ! -d "$CMR_DIR" ]; then
    echo -e "${RED}错误: CMR数据不存在: ${CMR_DIR}${NC}"
    exit 1
fi

if [ ! -d "$LGE_DIR" ]; then
    echo -e "${RED}错误: LGE数据不存在: ${LGE_DIR}${NC}"
    exit 1
fi

# 检查模型权重
MOTION_CHECKPOINT="checkpoints/motion/best_model.pth"
REG_CHECKPOINT="checkpoints/registration/best_model.pth"
SEG_CHECKPOINT="checkpoints/segmentation/best_model.pth"

if [ ! -f "$MOTION_CHECKPOINT" ] || [ ! -f "$REG_CHECKPOINT" ] || [ ! -f "$SEG_CHECKPOINT" ]; then
    echo -e "${RED}错误: 模型权重文件不完整！${NC}"
    echo -e "${YELLOW}请确保已完成所有训练步骤${NC}"
    exit 1
fi

# 创建输出目录
OUTPUT_DIR="results/inference/${CASE_ID}"
mkdir -p "$OUTPUT_DIR"

echo -e "${YELLOW}推理参数:${NC}"
echo -e "  病例ID: ${CASE_ID}"
echo -e "  CMR路径: ${CMR_DIR}"
echo -e "  LGE路径: ${LGE_DIR}"
echo -e "  输出目录: ${OUTPUT_DIR}"

echo -e "\n${GREEN}开始推理...${NC}\n"

# 运行推理
python3 src/inference.py \
    --case_id "$CASE_ID" \
    --data_root "$DATA_ROOT" \
    --motion_checkpoint "$MOTION_CHECKPOINT" \
    --registration_checkpoint "$REG_CHECKPOINT" \
    --segmentation_checkpoint "$SEG_CHECKPOINT" \
    --output_dir "$OUTPUT_DIR" \
    --save_intermediate

if [ $? -ne 0 ]; then
    echo -e "${RED}推理失败！${NC}"
    exit 1
fi

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}步骤6完成！${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "${YELLOW}推理结果保存在: ${OUTPUT_DIR}/${NC}"
echo -e "${YELLOW}包含以下文件:${NC}"
echo -e "  - mi_prediction.nii.gz (心梗分割结果)"
echo -e "  - motion_field.nii.gz (运动场)"
echo -e "  - strain_map.nii.gz (应变图)"
echo -e "  - deformation_field.nii.gz (形变场)"
