#!/bin/bash
# ============================================================================
# 一键运行全流程脚本
# 功能：自动执行从数据准备到模型训练的所有步骤
# 警告：这将需要很长时间（可能数天），请确保有足够的计算资源
# ============================================================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}无创心梗估计 - 全流程自动训练${NC}"
echo -e "${BLUE}============================================${NC}"

# 项目路径
PROJECT_ROOT="/home/yuanwenwei/code/mmm2/manus_gitproj/non_invasive_mi_estimation"
cd "$PROJECT_ROOT"

# 确认开始
echo -e "${YELLOW}警告: 此脚本将执行完整的训练流程，可能需要数天时间！${NC}"
echo -e "${YELLOW}请确保:${NC}"
echo -e "  1. 数据集路径正确"
echo -e "  2. 有足够的磁盘空间（至少50GB）"
echo -e "  3. 有GPU可用（强烈推荐）"
echo -e "  4. 可以长时间运行不中断"
echo ""
read -p "确认开始？(yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo -e "${RED}已取消${NC}"
    exit 0
fi

# 记录开始时间
START_TIME=$(date +%s)
echo -e "\n${GREEN}开始时间: $(date)${NC}\n"

# 创建日志目录
LOG_DIR="logs/full_run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo -e "${BLUE}日志将保存到: ${LOG_DIR}${NC}\n"

# 步骤1: 数据准备
echo -e "${GREEN}[1/5] 执行步骤1: 数据准备${NC}"
bash scripts/step1_prepare_data.sh 2>&1 | tee "${LOG_DIR}/step1_prepare_data.log"
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo -e "${RED}步骤1失败！请查看日志: ${LOG_DIR}/step1_prepare_data.log${NC}"
    exit 1
fi
echo -e "${GREEN}✓ 步骤1完成${NC}\n"

# 步骤2: 训练运动估计模块
echo -e "${GREEN}[2/5] 执行步骤2: 训练运动估计模块${NC}"
echo -e "${YELLOW}预计时间: 数小时${NC}"
bash scripts/step2_train_motion.sh 2>&1 | tee "${LOG_DIR}/step2_train_motion.log"
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo -e "${RED}步骤2失败！请查看日志: ${LOG_DIR}/step2_train_motion.log${NC}"
    exit 1
fi
echo -e "${GREEN}✓ 步骤2完成${NC}\n"

# 步骤3: 训练配准模块
echo -e "${GREEN}[3/5] 执行步骤3: 训练配准模块${NC}"
echo -e "${YELLOW}预计时间: 数小时${NC}"
bash scripts/step3_train_registration.sh 2>&1 | tee "${LOG_DIR}/step3_train_registration.log"
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo -e "${RED}步骤3失败！请查看日志: ${LOG_DIR}/step3_train_registration.log${NC}"
    exit 1
fi
echo -e "${GREEN}✓ 步骤3完成${NC}\n"

# 步骤4: 训练分割模块
echo -e "${GREEN}[4/5] 执行步骤4: 训练分割模块${NC}"
echo -e "${YELLOW}预计时间: 数小时到1天${NC}"
bash scripts/step4_train_segmentation.sh 2>&1 | tee "${LOG_DIR}/step4_train_segmentation.log"
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo -e "${RED}步骤4失败！请查看日志: ${LOG_DIR}/step4_train_segmentation.log${NC}"
    exit 1
fi
echo -e "${GREEN}✓ 步骤4完成${NC}\n"

# 步骤5: 测试
echo -e "${GREEN}[5/5] 执行步骤5: 测试模型${NC}"
bash scripts/step5_test.sh 2>&1 | tee "${LOG_DIR}/step5_test.log"
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo -e "${RED}步骤5失败！请查看日志: ${LOG_DIR}/step5_test.log${NC}"
    exit 1
fi
echo -e "${GREEN}✓ 步骤5完成${NC}\n"

# 计算总用时
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}全流程完成！${NC}"
echo -e "${BLUE}============================================${NC}"
echo -e "${GREEN}结束时间: $(date)${NC}"
echo -e "${GREEN}总用时: ${HOURS}小时${MINUTES}分钟${NC}"
echo -e "${YELLOW}所有日志保存在: ${LOG_DIR}/${NC}"
echo -e "${YELLOW}测试结果保存在: results/test_results/${NC}"
echo ""
echo -e "${YELLOW}下一步:${NC}"
echo -e "  - 查看测试结果: cat results/test_results/test_results.json"
echo -e "  - 进行单例推理: ./scripts/step6_inference.sh <case_id>"
