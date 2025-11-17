#!/bin/bash
# ============================================================================
# 步骤1：数据验证和准备
# 功能：验证数据集结构，生成训练/验证/测试集划分
# ============================================================================

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}步骤1：数据验证和准备${NC}"
echo -e "${GREEN}========================================${NC}"

# 项目路径和数据集路径
PROJECT_ROOT="/home/yuanwenwei/code/mmm2/manus_gitproj/non_invasive_mi_estimation"
DATA_ROOT="/data/yuanwenwei/datasets/lge_pred_dataset/shengyi_all/cropped"

echo -e "${YELLOW}项目路径: ${PROJECT_ROOT}${NC}"
echo -e "${YELLOW}数据集路径: ${DATA_ROOT}${NC}"

# 检查路径是否存在
if [ ! -d "$PROJECT_ROOT" ]; then
    echo -e "${RED}错误: 项目路径不存在！${NC}"
    exit 1
fi

if [ ! -d "$DATA_ROOT" ]; then
    echo -e "${RED}错误: 数据集路径不存在！${NC}"
    exit 1
fi

cd "$PROJECT_ROOT"

# 1. 验证数据集结构
echo -e "\n${GREEN}[1/3] 验证数据集结构...${NC}"
python3 src/data/prepare_custom_dataset.py \
    --data_root "$DATA_ROOT"

if [ $? -ne 0 ]; then
    echo -e "${RED}数据集结构验证失败！请检查数据集格式。${NC}"
    exit 1
fi

echo -e "${GREEN}✓ 数据集结构验证通过${NC}"

# 2. 生成数据集划分
echo -e "\n${GREEN}[2/3] 生成训练/验证/测试集划分...${NC}"
python3 src/data/split_dataset.py \
    --data_root "$DATA_ROOT" \
    --splits_file data/splits/dataset_splits.json \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15 \
    --seed 42

if [ $? -ne 0 ]; then
    echo -e "${RED}数据集划分失败！${NC}"
    exit 1
fi

echo -e "${GREEN}✓ 数据集划分完成${NC}"

# 3. 显示数据集统计信息
echo -e "\n${GREEN}[3/3] 数据集统计信息：${NC}"
python3 << EOF
import json
import os

splits_file = "${PROJECT_ROOT}/data/splits/dataset_splits.json"
if os.path.exists(splits_file):
    with open(splits_file, 'r') as f:
        splits = json.load(f)
    
    print(f"训练集: {len(splits['train'])} 例")
    print(f"验证集: {len(splits['val'])} 例")
    print(f"测试集: {len(splits['test'])} 例")
    print(f"总计: {len(splits['train']) + len(splits['val']) + len(splits['test'])} 例")
else:
    print("错误: 划分文件不存在！")
EOF

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}步骤1完成！${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "${YELLOW}下一步: 运行 ./scripts/step2_train_motion.sh${NC}"
