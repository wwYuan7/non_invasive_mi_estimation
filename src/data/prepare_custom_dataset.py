"""
数据集准备和验证脚本

用于验证用户数据集的完整性和格式正确性
"""

import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import nibabel as nib
import numpy as np


def check_file_exists(filepath: Path) -> bool:
    """检查文件是否存在"""
    return filepath.exists() and filepath.is_file()


def check_nifti_file(filepath: Path) -> Dict:
    """
    检查NIfTI文件并返回基本信息
    
    Returns:
        包含shape, dtype等信息的字典
    """
    try:
        img = nib.load(str(filepath))
        data = img.get_fdata()
        
        return {
            'exists': True,
            'shape': data.shape,
            'dtype': str(data.dtype),
            'min': float(data.min()),
            'max': float(data.max()),
            'error': None
        }
    except Exception as e:
        return {
            'exists': False,
            'error': str(e)
        }


def validate_case(data_root: Path, case_id: str) -> Dict:
    """
    验证单个病例的数据完整性
    
    Args:
        data_root: 数据根目录
        case_id: 病例ID
    
    Returns:
        验证结果字典
    """
    result = {
        'case_id': case_id,
        'valid': True,
        'errors': [],
        'warnings': [],
        'files': {}
    }
    
    # 检查CMR序列目录
    cmr_dir = data_root / 'images' / 'cmr' / case_id
    if not cmr_dir.exists():
        result['valid'] = False
        result['errors'].append(f"CMR directory not found: {cmr_dir}")
    else:
        cmr_files = list(cmr_dir.glob('*.nii.gz'))
        result['files']['cmr_frames'] = len(cmr_files)
        if len(cmr_files) == 0:
            result['valid'] = False
            result['errors'].append("No CMR frames found")
        elif len(cmr_files) < 10:
            result['warnings'].append(f"Only {len(cmr_files)} CMR frames (expected 20-40)")
    
    # 检查LGE图像
    lge_file = data_root / 'images' / 'lge' / f'{case_id}.nii.gz'
    if not check_file_exists(lge_file):
        result['valid'] = False
        result['errors'].append(f"LGE image not found: {lge_file}")
    else:
        lge_info = check_nifti_file(lge_file)
        result['files']['lge'] = lge_info
        if lge_info.get('error'):
            result['valid'] = False
            result['errors'].append(f"LGE file error: {lge_info['error']}")
    
    # 检查CMR心肌掩模
    cmr_mask_file = data_root / 'labels' / 'cmr' / 'cmr_Myo_mask' / f'{case_id}.nii.gz'
    if not check_file_exists(cmr_mask_file):
        result['valid'] = False
        result['errors'].append(f"CMR myocardium mask not found: {cmr_mask_file}")
    else:
        mask_info = check_nifti_file(cmr_mask_file)
        result['files']['cmr_myo_mask'] = mask_info
        if mask_info.get('error'):
            result['valid'] = False
            result['errors'].append(f"CMR mask error: {mask_info['error']}")
    
    # 检查LGE心肌掩模
    lge_myo_file = data_root / 'labels' / 'lge' / 'lge_Myo_labels' / f'{case_id}.nii.gz'
    if not check_file_exists(lge_myo_file):
        result['valid'] = False
        result['errors'].append(f"LGE myocardium mask not found: {lge_myo_file}")
    else:
        mask_info = check_nifti_file(lge_myo_file)
        result['files']['lge_myo_mask'] = mask_info
        if mask_info.get('error'):
            result['valid'] = False
            result['errors'].append(f"LGE myo mask error: {mask_info['error']}")
    
    # 检查心梗标签
    mi_file = data_root / 'labels' / 'lge' / 'lge_MI_labels' / f'{case_id}.nii.gz'
    if not check_file_exists(mi_file):
        result['valid'] = False
        result['errors'].append(f"MI label not found: {mi_file}")
    else:
        mi_info = check_nifti_file(mi_file)
        result['files']['mi_label'] = mi_info
        if mi_info.get('error'):
            result['valid'] = False
            result['errors'].append(f"MI label error: {mi_info['error']}")
    
    return result


def validate_dataset(data_root: str) -> Tuple[List[str], List[Dict]]:
    """
    验证整个数据集
    
    Args:
        data_root: 数据根目录
    
    Returns:
        (valid_cases, validation_results)
    """
    data_root = Path(data_root)
    
    # 检查目录结构
    required_dirs = [
        'images/cmr',
        'images/lge',
        'labels/cmr/cmr_Myo_mask',
        'labels/lge/lge_MI_labels',
        'labels/lge/lge_Myo_labels'
    ]
    
    print("Checking directory structure...")
    for dir_path in required_dirs:
        full_path = data_root / dir_path
        if not full_path.exists():
            print(f"  ❌ Missing: {dir_path}")
            raise ValueError(f"Required directory not found: {full_path}")
        else:
            print(f"  ✓ Found: {dir_path}")
    
    # 获取所有病例
    cmr_dir = data_root / 'images' / 'cmr'
    cases = [d.name for d in cmr_dir.iterdir() if d.is_dir()]
    cases = sorted(cases)
    
    print(f"\nFound {len(cases)} cases")
    print("Validating each case...")
    
    # 验证每个病例
    validation_results = []
    valid_cases = []
    
    for i, case_id in enumerate(cases, 1):
        print(f"\n[{i}/{len(cases)}] Validating {case_id}...")
        result = validate_case(data_root, case_id)
        validation_results.append(result)
        
        if result['valid']:
            valid_cases.append(case_id)
            print(f"  ✓ Valid")
            if result['warnings']:
                for warning in result['warnings']:
                    print(f"    ⚠ {warning}")
        else:
            print(f"  ❌ Invalid")
            for error in result['errors']:
                print(f"    ✗ {error}")
    
    return valid_cases, validation_results


def print_summary(valid_cases: List[str], validation_results: List[Dict]):
    """打印验证摘要"""
    total_cases = len(validation_results)
    valid_count = len(valid_cases)
    invalid_count = total_cases - valid_count
    
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print(f"Total cases:   {total_cases}")
    print(f"Valid cases:   {valid_count} ({valid_count/total_cases*100:.1f}%)")
    print(f"Invalid cases: {invalid_count} ({invalid_count/total_cases*100:.1f}%)")
    
    if invalid_count > 0:
        print("\nInvalid cases:")
        for result in validation_results:
            if not result['valid']:
                print(f"  - {result['case_id']}")
                for error in result['errors'][:2]:  # 只显示前2个错误
                    print(f"      {error}")
    
    # 统计CMR帧数
    frame_counts = [r['files'].get('cmr_frames', 0) for r in validation_results 
                    if 'cmr_frames' in r['files']]
    if frame_counts:
        print(f"\nCMR frames statistics:")
        print(f"  Min:    {min(frame_counts)}")
        print(f"  Max:    {max(frame_counts)}")
        print(f"  Mean:   {np.mean(frame_counts):.1f}")
        print(f"  Median: {np.median(frame_counts):.1f}")
    
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Validate custom dataset')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory of the dataset')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for validation report (JSON)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("DATASET VALIDATION")
    print("="*60)
    print(f"Data root: {args.data_root}\n")
    
    try:
        # 验证数据集
        valid_cases, validation_results = validate_dataset(args.data_root)
        
        # 打印摘要
        print_summary(valid_cases, validation_results)
        
        # 保存报告
        if args.output:
            import json
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            report = {
                'total_cases': len(validation_results),
                'valid_cases': valid_cases,
                'validation_results': validation_results
            }
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"\nValidation report saved to: {output_path}")
        
        # 保存有效病例列表
        valid_cases_file = Path(args.data_root) / 'valid_cases.txt'
        with open(valid_cases_file, 'w') as f:
            f.write('\n'.join(valid_cases))
        print(f"Valid cases list saved to: {valid_cases_file}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
