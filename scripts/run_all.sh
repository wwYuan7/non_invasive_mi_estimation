#!/bin/bash
# ============================================================================
# 完整训练流程 - 支持断点续训和重新训练
# ============================================================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# 项目路径
PROJECT_ROOT="/home/yuanwenwei/code/mmm2/manus_gitproj/non_invasive_mi_estimation"
cd "$PROJECT_ROOT"

# 进度文件
PROGRESS_FILE="$PROJECT_ROOT/.training_progress"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}非侵入性心梗估计 - 完整训练流程${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# 检查是否存在进度文件
if [ -f "$PROGRESS_FILE" ]; then
    CURRENT_STEP=$(cat "$PROGRESS_FILE")
    echo -e "${BLUE}检测到上次训练进度: ${CURRENT_STEP}${NC}"
    echo ""
else
    CURRENT_STEP="step0_not_started"
fi

# 检查已完成的模型
echo -e "${CYAN}检查已训练的模型:${NC}"
MOTION_MODEL="checkpoints/motion/best_model.pth"
REG_MODEL="checkpoints/registration/best_model.pth"
SEG_MODEL="checkpoints/segmentation/best_model.pth"

[ -f "$MOTION_MODEL" ] && echo -e "${GREEN}  ✓ 运动估计模型${NC}" || echo -e "${YELLOW}  ✗ 运动估计模型${NC}"
[ -f "$REG_MODEL" ] && echo -e "${GREEN}  ✓ 配准模型${NC}" || echo -e "${YELLOW}  ✗ 配准模型${NC}"
[ -f "$SEG_MODEL" ] && echo -e "${GREEN}  ✓ 分割模型${NC}" || echo -e "${YELLOW}  ✗ 分割模型${NC}"
echo ""

# 交互式选择训练模式
echo -e "${CYAN}请选择训练模式:${NC}"
echo -e "  ${YELLOW}1)${NC} 从头开始训练 (清除所有进度和模型)"
echo -e "  ${YELLOW}2)${NC} 断点续训 (从上次中断处继续)"
echo -e "  ${YELLOW}3)${NC} 智能跳过 (自动跳过已完成的步骤)"
echo -e "  ${YELLOW}4)${NC} 退出"
echo ""
read -p "请输入选项 [1-4]: " MODE

case $MODE in
    1)
        echo -e "${YELLOW}选择: 从头开始训练${NC}"
        echo -e "${RED}警告: 这将删除所有已训练的模型和进度！${NC}"
        read -p "确认继续? [y/N]: " CONFIRM
        if [[ "$CONFIRM" =~ ^[Yy]$ ]]; then
            rm -rf checkpoints/motion checkpoints/registration checkpoints/segmentation
            rm -rf logs/motion logs/registration logs/segmentation
            echo "step0_not_started" > "$PROGRESS_FILE"
            CURRENT_STEP="step0_not_started"
            echo -e "${GREEN}已清除所有进度，将从头开始训练${NC}"
        else
            echo -e "${YELLOW}已取消${NC}"
            exit 0
        fi
        ;;
    2)
        echo -e "${YELLOW}选择: 断点续训${NC}"
        echo -e "${GREEN}将从 ${CURRENT_STEP} 继续训练${NC}"
        ;;
    3)
        echo -e "${YELLOW}选择: 智能跳过${NC}"
        echo -e "${GREEN}将自动检测并跳过已完成的步骤${NC}"
        SMART_SKIP=true
        ;;
    4)
        echo -e "${YELLOW}退出${NC}"
        exit 0
        ;;
    *)
        echo -e "${RED}无效选项${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}开始训练...${NC}"
echo ""

# 更新进度函数
update_progress() {
    echo "$1" > "$PROGRESS_FILE"
    echo -e "${GREEN}✓ 进度已更新: $1${NC}"
}

# 检查模型是否存在
check_model_exists() {
    local model_path=$1
    if [ -f "$model_path" ]; then
        return 0
    else
        return 1
    fi
}

# ============================================================================
# 步骤1: 准备数据
# ============================================================================

if [[ "$CURRENT_STEP" == "step0_not_started" ]] || [[ "$SMART_SKIP" == true && ! -f "data/splits/dataset_splits.json" ]]; then
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}步骤1: 准备数据${NC}"
    echo -e "${YELLOW}========================================${NC}"
    
    bash scripts/step1_prepare_data.sh
    
    if [ $? -eq 0 ]; then
        update_progress "step1_data_prepared"
        CURRENT_STEP="step1_data_prepared"
        echo -e "${GREEN}步骤1完成！${NC}\n"
    else
        echo -e "${RED}步骤1失败！${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✓ 跳过步骤1: 数据已准备${NC}\n"
    if [[ "$CURRENT_STEP" == "step0_not_started" ]]; then
        CURRENT_STEP="step1_data_prepared"
    fi
fi

# ============================================================================
# 步骤2: 训练运动估计模块
# ============================================================================

SHOULD_TRAIN_MOTION=false

if [[ "$CURRENT_STEP" == "step1_data_prepared" ]]; then
    SHOULD_TRAIN_MOTION=true
elif [[ "$SMART_SKIP" == true ]]; then
    if ! check_model_exists "$MOTION_MODEL"; then
        SHOULD_TRAIN_MOTION=true
    fi
fi

if [ "$SHOULD_TRAIN_MOTION" = true ]; then
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}步骤2: 训练运动估计模块${NC}"
    echo -e "${YELLOW}========================================${NC}"
    
    bash scripts/step2_train_motion.sh
    
    if [ $? -eq 0 ] && check_model_exists "$MOTION_MODEL"; then
        update_progress "step2_motion_trained"
        CURRENT_STEP="step2_motion_trained"
        echo -e "${GREEN}步骤2完成！${NC}\n"
    else
        echo -e "${RED}步骤2失败！${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✓ 跳过步骤2: 运动估计模块已训练${NC}\n"
    if [[ "$CURRENT_STEP" == "step1_data_prepared" ]]; then
        CURRENT_STEP="step2_motion_trained"
    fi
fi

# ============================================================================
# 步骤3: 训练配准模块
# ============================================================================

SHOULD_TRAIN_REG=false

if [[ "$CURRENT_STEP" == "step2_motion_trained" ]]; then
    SHOULD_TRAIN_REG=true
elif [[ "$SMART_SKIP" == true ]]; then
    if ! check_model_exists "$REG_MODEL"; then
        SHOULD_TRAIN_REG=true
    fi
fi

if [ "$SHOULD_TRAIN_REG" = true ]; then
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}步骤3: 训练配准模块${NC}"
    echo -e "${YELLOW}========================================${NC}"
    
    bash scripts/step3_train_registration.sh
    
    if [ $? -eq 0 ] && check_model_exists "$REG_MODEL"; then
        update_progress "step3_registration_trained"
        CURRENT_STEP="step3_registration_trained"
        echo -e "${GREEN}步骤3完成！${NC}\n"
    else
        echo -e "${RED}步骤3失败！${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✓ 跳过步骤3: 配准模块已训练${NC}\n"
    if [[ "$CURRENT_STEP" == "step2_motion_trained" ]]; then
        CURRENT_STEP="step3_registration_trained"
    fi
fi

# ============================================================================
# 步骤4: 训练分割模块
# ============================================================================

SHOULD_TRAIN_SEG=false

if [[ "$CURRENT_STEP" == "step3_registration_trained" ]]; then
    SHOULD_TRAIN_SEG=true
elif [[ "$SMART_SKIP" == true ]]; then
    if ! check_model_exists "$SEG_MODEL"; then
        SHOULD_TRAIN_SEG=true
    fi
fi

if [ "$SHOULD_TRAIN_SEG" = true ]; then
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}步骤4: 训练分割模块${NC}"
    echo -e "${YELLOW}========================================${NC}"
    
    bash scripts/step4_train_segmentation.sh
    
    if [ $? -eq 0 ] && check_model_exists "$SEG_MODEL"; then
        update_progress "step4_segmentation_trained"
        CURRENT_STEP="step4_segmentation_trained"
        echo -e "${GREEN}步骤4完成！${NC}\n"
    else
        echo -e "${RED}步骤4失败！${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✓ 跳过步骤4: 分割模块已训练${NC}\n"
    if [[ "$CURRENT_STEP" == "step3_registration_trained" ]]; then
        CURRENT_STEP="step4_segmentation_trained"
    fi
fi

# ============================================================================
# 完成
# ============================================================================

if [[ "$CURRENT_STEP" == "step4_segmentation_trained" ]]; then
    update_progress "all_completed"
    
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}🎉 所有训练步骤已完成！${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "${CYAN}训练好的模型:${NC}"
    echo -e "  ${GREEN}✓${NC} 运动估计: ${MOTION_MODEL}"
    echo -e "  ${GREEN}✓${NC} 配准: ${REG_MODEL}"
    echo -e "  ${GREEN}✓${NC} 分割: ${SEG_MODEL}"
    echo ""
    echo -e "${YELLOW}下一步: 运行测试${NC}"
    echo -e "  bash scripts/step5_test.sh"
    echo ""
fi
