#!/bin/bash

echo "=========================================="
echo "验证训练环境和脚本"
echo "=========================================="
echo ""

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 检查Python
echo "1. 检查Python环境..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo -e "${GREEN}✓${NC} Python已安装: $PYTHON_VERSION"
else
    echo -e "${RED}✗${NC} Python未安装"
    exit 1
fi

# 检查PyTorch
echo ""
echo "2. 检查PyTorch..."
python3 -c "import torch; print('PyTorch版本:', torch.__version__); print('CUDA可用:', torch.cuda.is_available())" 2>/dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} PyTorch已安装"
else
    echo -e "${RED}✗${NC} PyTorch未安装或版本不兼容"
fi

# 检查必要的包
echo ""
echo "3. 检查必要的Python包..."
PACKAGES=("numpy" "nibabel" "tqdm")
for pkg in "${PACKAGES[@]}"; do
    python3 -c "import $pkg" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $pkg 已安装"
    else
        echo -e "${RED}✗${NC} $pkg 未安装"
    fi
done

# 检查训练脚本
echo ""
echo "4. 检查训练脚本参数解析..."
SCRIPTS=("train_motion.py" "train_registration.py" "train_segmentation.py")
for script in "${SCRIPTS[@]}"; do
    python3 src/$script --help > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $script 参数解析正常"
    else
        echo -e "${RED}✗${NC} $script 参数解析失败"
    fi
done

# 检查数据集路径
echo ""
echo "5. 检查数据集路径..."
DATA_ROOT="/data/yuanwenwei/datasets/lge_pred_dataset/shengyi_all/cropped"
if [ -d "$DATA_ROOT" ]; then
    echo -e "${GREEN}✓${NC} 数据集路径存在: $DATA_ROOT"
    
    # 检查关键子目录
    SUBDIRS=("images/cmr" "images/lge" "labels/cmr/cmr_Myo_mask" "labels/lge_original/lge_MI_labels")
    for subdir in "${SUBDIRS[@]}"; do
        if [ -d "$DATA_ROOT/$subdir" ]; then
            echo -e "${GREEN}  ✓${NC} $subdir"
        else
            echo -e "${RED}  ✗${NC} $subdir 不存在"
        fi
    done
else
    echo -e "${RED}✗${NC} 数据集路径不存在: $DATA_ROOT"
    echo "  请确保数据集已正确挂载"
fi

# 检查shell脚本
echo ""
echo "6. 检查训练shell脚本..."
SHELL_SCRIPTS=("step1_prepare_data.sh" "step2_train_motion.sh" "step3_train_registration.sh" "step4_train_segmentation.sh")
for script in "${SHELL_SCRIPTS[@]}"; do
    if [ -f "scripts/$script" ] && [ -x "scripts/$script" ]; then
        echo -e "${GREEN}✓${NC} $script 存在且可执行"
    else
        echo -e "${RED}✗${NC} $script 不存在或不可执行"
    fi
done

echo ""
echo "=========================================="
echo "验证完成!"
echo "=========================================="
echo ""
echo "如果所有检查都通过,您可以开始训练:"
echo "  1. bash scripts/step1_prepare_data.sh"
echo "  2. bash scripts/step2_train_motion.sh"
echo "  3. bash scripts/step3_train_registration.sh"
echo "  4. bash scripts/step4_train_segmentation.sh"
