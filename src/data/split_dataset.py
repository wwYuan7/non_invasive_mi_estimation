"""
数据集分割脚本

将用户的数据集分割为训练集、验证集和测试集
"""

import os
import json
import random
import argparse
from pathlib import Path
from typing import List, Tuple, Dict


def get_all_cases(data_root: str) -> List[str]:
    """
    获取所有病例名称
    
    Args:
        data_root: 数据根目录
    
    Returns:
        病例名称列表
    """
    cmr_dir = Path(data_root) / 'images' / 'cmr'
    
    if not cmr_dir.exists():
        raise ValueError(f"CMR directory not found: {cmr_dir}")
    
    # 获取所有子目录名称作为病例ID
    cases = [d.name for d in cmr_dir.iterdir() if d.is_dir()]
    cases = sorted(cases)
    
    print(f"Found {len(cases)} cases in {cmr_dir}")
    
    return cases


def split_dataset(
    cases: List[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """
    分割数据集
    
    Args:
        cases: 所有病例列表
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
    
    Returns:
        (train_cases, val_cases, test_cases)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    # 设置随机种子
    random.seed(seed)
    
    # 打乱病例顺序
    cases = cases.copy()
    random.shuffle(cases)
    
    # 计算分割点
    n_total = len(cases)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # 分割
    train_cases = cases[:n_train]
    val_cases = cases[n_train:n_train + n_val]
    test_cases = cases[n_train + n_val:]
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_cases)} cases ({len(train_cases)/n_total*100:.1f}%)")
    print(f"  Val:   {len(val_cases)} cases ({len(val_cases)/n_total*100:.1f}%)")
    print(f"  Test:  {len(test_cases)} cases ({len(test_cases)/n_total*100:.1f}%)")
    
    return train_cases, val_cases, test_cases


def save_splits(
    train_cases: List[str],
    val_cases: List[str],
    test_cases: List[str],
    output_dir: str
):
    """
    保存分割结果
    
    Args:
        train_cases: 训练集病例
        val_cases: 验证集病例
        test_cases: 测试集病例
        output_dir: 输出目录
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存为JSON格式
    splits = {
        'train': train_cases,
        'val': val_cases,
        'test': test_cases,
        'total': len(train_cases) + len(val_cases) + len(test_cases)
    }
    
    json_file = output_dir / 'dataset_splits.json'
    with open(json_file, 'w') as f:
        json.dump(splits, f, indent=2)
    
    print(f"\nSaved splits to: {json_file}")
    
    # 同时保存为文本文件（方便查看）
    for split_name, cases in [('train', train_cases), ('val', val_cases), ('test', test_cases)]:
        txt_file = output_dir / f'{split_name}_cases.txt'
        with open(txt_file, 'w') as f:
            f.write('\n'.join(cases))
        print(f"Saved {split_name} cases to: {txt_file}")


def load_splits(splits_file: str) -> Dict[str, List[str]]:
    """
    加载已保存的分割结果
    
    Args:
        splits_file: 分割文件路径 (JSON)
    
    Returns:
        包含train/val/test的字典
    """
    with open(splits_file, 'r') as f:
        splits = json.load(f)
    
    return splits


def main():
    parser = argparse.ArgumentParser(description='Split dataset into train/val/test')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory of the dataset')
    parser.add_argument('--output_dir', type=str, default='data/splits',
                        help='Output directory for split files')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Training set ratio (default: 0.7)')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Validation set ratio (default: 0.15)')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='Test set ratio (default: 0.15)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Dataset Splitting")
    print("="*60)
    print(f"Data root: {args.data_root}")
    print(f"Output dir: {args.output_dir}")
    print(f"Split ratios: Train={args.train_ratio}, Val={args.val_ratio}, Test={args.test_ratio}")
    print(f"Random seed: {args.seed}")
    print("="*60)
    
    # 获取所有病例
    cases = get_all_cases(args.data_root)
    
    if len(cases) == 0:
        print("Error: No cases found!")
        return
    
    # 分割数据集
    train_cases, val_cases, test_cases = split_dataset(
        cases,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    # 保存分割结果
    save_splits(train_cases, val_cases, test_cases, args.output_dir)
    
    print("\n" + "="*60)
    print("Dataset splitting completed successfully!")
    print("="*60)
    
    # 显示示例
    print("\nExample cases:")
    print(f"  Train: {train_cases[:3]}")
    print(f"  Val:   {val_cases[:3]}")
    print(f"  Test:  {test_cases[:3]}")


if __name__ == '__main__':
    main()
