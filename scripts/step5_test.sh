#!/bin/bash
# ============================================================================
# 步骤5：在测试集上评估模型
# 功能：评估训练好的模型在测试集上的性能
# ============================================================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}步骤5：测试模型${NC}"
echo -e "${GREEN}========================================${NC}"

# 项目路径和数据集路径
PROJECT_ROOT="/home/yuanwenwei/code/mmm2/manus_gitproj/non_invasive_mi_estimation"
DATA_ROOT="/data/yuanwenwei/datasets/lge_pred_dataset/shengyi_all/cropped"

cd "$PROJECT_ROOT"

# 检查必要文件
SPLITS_FILE="data/splits/dataset_splits.json"
MOTION_CHECKPOINT="checkpoints/motion/best_model.pth"
REG_CHECKPOINT="checkpoints/registration/best_model.pth"
SEG_CHECKPOINT="checkpoints/segmentation/best_model.pth"

if [ ! -f "$SPLITS_FILE" ]; then
    echo -e "${RED}错误: 数据集划分文件不存在！${NC}"
    exit 1
fi

if [ ! -f "$MOTION_CHECKPOINT" ]; then
    echo -e "${RED}错误: 运动估计模块权重不存在！${NC}"
    exit 1
fi

if [ ! -f "$REG_CHECKPOINT" ]; then
    echo -e "${RED}错误: 配准模块权重不存在！${NC}"
    exit 1
fi

if [ ! -f "$SEG_CHECKPOINT" ]; then
    echo -e "${RED}错误: 分割模块权重不存在！${NC}"
    exit 1
fi

# 创建输出目录
OUTPUT_DIR="results/test_results"
mkdir -p "$OUTPUT_DIR"

echo -e "${YELLOW}测试参数:${NC}"
echo -e "  Motion Checkpoint: ${MOTION_CHECKPOINT}"
echo -e "  Registration Checkpoint: ${REG_CHECKPOINT}"
echo -e "  Segmentation Checkpoint: ${SEG_CHECKPOINT}"
echo -e "  Output Directory: ${OUTPUT_DIR}"

echo -e "\n${GREEN}开始在测试集上评估模型...${NC}\n"

# 创建测试脚本
cat > src/test.py << 'PYTHON_SCRIPT'
import torch
import numpy as np
import argparse
import json
import os
from tqdm import tqdm
import nibabel as nib

from models.motion_pyramid import MotionPyramidNet
from models.voxelmorph_simple import VoxelMorph
from models.attention_unet import AttentionUNet
from data.custom_dataloader_final import MIDataset
from utils.train_utils import dice_coefficient, hausdorff_distance_95, average_surface_distance

def test(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据集划分
    with open(args.splits_file, 'r') as f:
        splits = json.load(f)
    test_cases = splits['test']
    print(f"测试集样本数: {len(test_cases)}")
    
    # 创建数据集
    test_dataset = MIDataset(
        data_root=args.data_root,
        case_ids=test_cases,
        mode='test'
    )
    
    # 加载模型
    print("\n加载模型...")
    motion_model = MotionPyramidNet(in_channels=2).to(device)
    motion_model.load_state_dict(torch.load(args.motion_checkpoint, map_location=device))
    motion_model.eval()
    
    reg_model = VoxelMorph(in_channels=2).to(device)
    reg_model.load_state_dict(torch.load(args.registration_checkpoint, map_location=device))
    reg_model.eval()
    
    seg_model = AttentionUNet(in_channels=4, out_channels=1).to(device)
    seg_model.load_state_dict(torch.load(args.segmentation_checkpoint, map_location=device))
    seg_model.eval()
    
    print("模型加载完成！")
    
    # 评估指标
    dice_scores = []
    hd95_scores = []
    asd_scores = []
    
    print("\n开始测试...")
    with torch.no_grad():
        for idx in tqdm(range(len(test_dataset)), desc="测试进度"):
            data = test_dataset[idx]
            
            # 准备输入
            cmr_ed = data['cmr_ed'].unsqueeze(0).to(device)
            cmr_es = data['cmr_es'].unsqueeze(0).to(device)
            lge = data['lge'].unsqueeze(0).to(device)
            myo_mask = data['cmr_myo_mask'].unsqueeze(0).to(device)
            gt_mi = data['lge_mi_label'].unsqueeze(0).to(device)
            
            # 1. 运动估计
            motion_field, strain_map = motion_model(cmr_ed, cmr_es)
            
            # 2. 配准
            flow = reg_model(cmr_ed, lge)
            
            # 3. 分割
            seg_input = torch.cat([cmr_ed, motion_field, strain_map.unsqueeze(1)], dim=1)
            pred_mi = seg_model(seg_input)
            pred_mi = (pred_mi > 0.5).float()
            
            # 计算指标
            pred_np = pred_mi.cpu().numpy()[0, 0]
            gt_np = gt_mi.cpu().numpy()[0, 0]
            
            dice = dice_coefficient(pred_np, gt_np)
            hd95 = hausdorff_distance_95(pred_np, gt_np)
            asd = average_surface_distance(pred_np, gt_np)
            
            dice_scores.append(dice)
            hd95_scores.append(hd95)
            asd_scores.append(asd)
            
            # 保存预测结果
            case_id = test_cases[idx]
            output_path = os.path.join(args.output_dir, f"{case_id}_pred.nii.gz")
            nii_img = nib.Nifti1Image(pred_np, affine=np.eye(4))
            nib.save(nii_img, output_path)
    
    # 计算平均指标
    mean_dice = np.mean(dice_scores)
    std_dice = np.std(dice_scores)
    mean_hd95 = np.mean(hd95_scores)
    std_hd95 = np.std(hd95_scores)
    mean_asd = np.mean(asd_scores)
    std_asd = np.std(asd_scores)
    
    # 打印结果
    print("\n" + "="*50)
    print("测试结果:")
    print("="*50)
    print(f"Dice系数: {mean_dice:.4f} ± {std_dice:.4f}")
    print(f"Hausdorff距离(95%): {mean_hd95:.2f} ± {std_hd95:.2f} mm")
    print(f"平均表面距离: {mean_asd:.2f} ± {std_asd:.2f} mm")
    print("="*50)
    
    # 保存结果
    results = {
        'dice_mean': float(mean_dice),
        'dice_std': float(std_dice),
        'hd95_mean': float(mean_hd95),
        'hd95_std': float(std_hd95),
        'asd_mean': float(mean_asd),
        'asd_std': float(std_asd),
        'per_case_results': [
            {
                'case_id': test_cases[i],
                'dice': float(dice_scores[i]),
                'hd95': float(hd95_scores[i]),
                'asd': float(asd_scores[i])
            }
            for i in range(len(test_cases))
        ]
    }
    
    results_file = os.path.join(args.output_dir, 'test_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\n结果已保存到: {results_file}")
    print(f"预测结果已保存到: {args.output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--splits_file', type=str, required=True)
    parser.add_argument('--motion_checkpoint', type=str, required=True)
    parser.add_argument('--registration_checkpoint', type=str, required=True)
    parser.add_argument('--segmentation_checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    
    test(args)
PYTHON_SCRIPT

# 运行测试
python3 src/test.py \
    --data_root "$DATA_ROOT" \
    --splits_file "$SPLITS_FILE" \
    --motion_checkpoint "$MOTION_CHECKPOINT" \
    --registration_checkpoint "$REG_CHECKPOINT" \
    --segmentation_checkpoint "$SEG_CHECKPOINT" \
    --output_dir "$OUTPUT_DIR"

if [ $? -ne 0 ]; then
    echo -e "${RED}测试失败！${NC}"
    exit 1
fi

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}步骤5完成！${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "${YELLOW}测试结果保存在: ${OUTPUT_DIR}/${NC}"
echo -e "${YELLOW}下一步: 运行 ./scripts/step6_inference.sh 进行单例推理${NC}"
