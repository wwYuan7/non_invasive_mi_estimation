#!/bin/bash
# ============================================================================
# 完整训练流程 - 支持断点续训
# ============================================================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 项目路径
PROJECT_ROOT="/home/yuanwenwei/code/mmm2/manus_gitproj/non_invasive_mi_estimation"
cd "$PROJECT_ROOT"

# 进度文件
PROGRESS_FILE="$PROJECT_ROOT/.training_progress"

# 初始化进度文件
if [ ! -f "$PROGRESS_FILE" ]; then
    echo "step0_not_started" > "$PROGRESS_FILE"
fi

# 读取当前进度
CURRENT_STEP=$(cat "$PROGRESS_FILE")

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}非侵入性心梗估计 - 完整训练流程${NC}"
echo -e "${GREEN}支持断点续训${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# 显示当前进度
echo -e "${BLUE}当前训练进度: ${CURRENT_STEP}${NC}"
echo ""

# 检查模型是否已训练完成
check_model_exists() {
    local model_path=$1
    local model_name=$2
    
    if [ -f "$model_path" ]; then
        echo -e "${GREEN}✓ ${model_name} 已存在${NC}"
        return 0
    else
        echo -e "${YELLOW}✗ ${model_name} 不存在${NC}"
        return 1
    fi
}

# 更新进度
update_progress() {
    echo "$1" > "$PROGRESS_FILE"
    echo -e "${GREEN}进度已更新: $1${NC}"
}

# ============================================================================
# 步骤1: 准备数据
# ============================================================================

if [[ "$CURRENT_STEP" == "step0_not_started" ]]; then
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}步骤1: 准备数据${NC}"
    echo -e "${YELLOW}========================================${NC}"
    
    bash scripts/step1_prepare_data.sh
    
    if [ $? -eq 0 ]; then
        update_progress "step1_data_prepared"
        echo -e "${GREEN}步骤1完成！${NC}\n"
    else
        echo -e "${RED}步骤1失败！${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✓ 步骤1: 数据已准备${NC}\n"
fi

# ============================================================================
# 步骤2: 训练运动估计模块
# ============================================================================

MOTION_MODEL="checkpoints/motion/best_model.pth"

if [[ "$CURRENT_STEP" == "step1_data_prepared" ]]; then
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}步骤2: 训练运动估计模块${NC}"
    echo -e "${YELLOW}========================================${NC}"
    
    bash scripts/step2_train_motion.sh
    
    if [ $? -eq 0 ] && [ -f "$MOTION_MODEL" ]; then
        update_progress "step2_motion_trained"
        echo -e "${GREEN}步骤2完成！${NC}\n"
    else
        echo -e "${RED}步骤2失败！${NC}"
        exit 1
    fi
elif check_model_exists "$MOTION_MODEL" "运动估计模型"; then
    echo -e "${GREEN}✓ 步骤2: 运动估计模块已训练${NC}\n"
    if [[ "$CURRENT_STEP" == "step1_data_prepared" ]]; then
        update_progress "step2_motion_trained"
    fi
else
    echo -e "${YELLOW}需要重新训练运动估计模块${NC}"
    bash scripts/step2_train_motion.sh
    if [ $? -eq 0 ]; then
        update_progress "step2_motion_trained"
    else
        exit 1
    fi
fi

# ============================================================================
# 步骤3: 训练配准模块
# ============================================================================

REG_MODEL="checkpoints/registration/best_model.pth"

if [[ "$CURRENT_STEP" == "step2_motion_trained" ]]; then
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}步骤3: 训练配准模块${NC}"
    echo -e "${YELLOW}========================================${NC}"
    
    bash scripts/step3_train_registration.sh
    
    if [ $? -eq 0 ] && [ -f "$REG_MODEL" ]; then
        update_progress "step3_registration_trained"
        echo -e "${GREEN}步骤3完成！${NC}\n"
    else
        echo -e "${RED}步骤3失败！${NC}"
        exit 1
    fi
elif check_model_exists "$REG_MODEL" "配准模型"; then
    echo -e "${GREEN}✓ 步骤3: 配准模块已训练${NC}\n"
    if [[ "$CURRENT_STEP" == "step2_motion_trained" ]]; then
        update_progress "step3_registration_trained"
    fi
else
    echo -e "${YELLOW}需要重新训练配准模块${NC}"
    bash scripts/step3_train_registration.sh
    if [ $? -eq 0 ]; then
        update_progress "step3_registration_trained"
    else
        exit 1
    fi
fi

# ============================================================================
# 步骤4: 训练分割模块
# ============================================================================

SEG_MODEL="checkpoints/segmentation/best_model.pth"

if [[ "$CURRENT_STEP" == "step3_registration_trained" ]]; then
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}步骤4: 训练分割模块${NC}"
    echo -e "${YELLOW}========================================${NC}"
    
    bash scripts/step4_train_segmentation.sh
    
    if [ $? -eq 0 ] && [ -f "$SEG_MODEL" ]; then
        update_progress "step4_segmentation_trained"
        echo -e "${GREEN}步骤4完成！${NC}\n"
    else
        echo -e "${RED}步骤4失败！${NC}"
        exit 1
    fi
elif check_model_exists "$SEG_MODEL" "分割模型"; then
    echo -e "${GREEN}✓ 步骤4: 分割模块已训练${NC}\n"
    if [[ "$CURRENT_STEP" == "step3_registration_trained" ]]; then
        update_progress "step4_segmentation_trained"
    fi
else
    echo -e "${YELLOW}需要重新训练分割模块${NC}"
    bash scripts/step4_train_segmentation.sh
    if [ $? -eq 0 ]; then
        update_progress "step4_segmentation_trained"
    else
        exit 1
    fi
fi

# ============================================================================
# 完成
# ============================================================================

if [[ "$CURRENT_STEP" == "step4_segmentation_trained" ]] || [ -f "$SEG_MODEL" ]; then
    update_progress "all_completed"
    
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}所有训练步骤已完成！${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "${YELLOW}训练好的模型:${NC}"
    echo -e "  - 运动估计: ${MOTION_MODEL}"
    echo -e "  - 配准: ${REG_MODEL}"
    echo -e "  - 分割: ${SEG_MODEL}"
    echo ""
    echo -e "${YELLOW}下一步: 运行测试${NC}"
    echo -e "  bash scripts/step5_test.sh"
fi
