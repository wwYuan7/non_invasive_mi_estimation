"""
数据集准备和验证脚本 (v2)

用于验证用户数据集的完整性和格式正确性

数据结构 (所有数据都是 case/slice 两级目录):
- CMR图像: images/cmr/<case_id>/<slice_id>.nii.gz
- LGE图像: images/lge/<case_id>/<slice_id>.nii.gz
- CMR心肌掩码: labels/cmr/cmr_Myo_mask/<case_id>/<slice_id>.nii.gz
- LGE心肌掩码: labels/lge_original/lge_Myo_labels/<case_id>/<slice_id>.nii.gz
- 心梗标签: labels/lge_original/lge_MI_labels/<case_id>/<slice_id>.nii.gz
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
        'files': {},
        'slice_count': 0,
        'complete_slice_count': 0,
        'incomplete_slice_count': 0
    }
    
    # 检查CMR序列目录
    cmr_dir = data_root / 'images' / 'cmr' / case_id
    if not cmr_dir.exists():
        result['valid'] = False
        result['errors'].append(f"CMR directory not found: {cmr_dir}")
        return result
    
    # 获取CMR切片列表
    cmr_slices = sorted(list(cmr_dir.glob('*.nii.gz')))
    result['slice_count'] = len(cmr_slices)
    
    if len(cmr_slices) == 0:
        result['valid'] = False
        result['errors'].append("No CMR slices found")
        return result
    elif len(cmr_slices) < 5:
        result['warnings'].append(f"Only {len(cmr_slices)} CMR slices (expected 10-20)")
    
    # 验证每个切片的配套文件
    slice_ids = [s.stem.replace('.nii', '') for s in cmr_slices]
    
    for slice_id in slice_ids:
        # 检查该slice的所有必需文件
        required_files = [
            data_root / 'images' / 'cmr' / case_id / f'{slice_id}.nii.gz',
            data_root / 'images' / 'lge' / case_id / f'{slice_id}.nii.gz',
            data_root / 'labels' / 'cmr' / 'cmr_Myo_mask' / case_id / f'{slice_id}.nii.gz',
            data_root / 'labels' / 'lge_original' / 'lge_Myo_labels' / case_id / f'{slice_id}.nii.gz',
            data_root / 'labels' / 'lge_original' / 'lge_MI_labels' / case_id / f'{slice_id}.nii.gz',
        ]
        
        # 统计完整和不完整的slice
        if all(check_file_exists(f) for f in required_files):
            result['complete_slice_count'] += 1
        else:
            result['incomplete_slice_count'] += 1
    
    # 只要有完整的slice，就认为case有效
    if result['complete_slice_count'] == 0:
        result['valid'] = False
        result['errors'].append(f"No complete slices found (all {result['incomplete_slice_count']} slices have missing files)")
    elif result['incomplete_slice_count'] > 0:
        result['warnings'].append(f"{result['incomplete_slice_count']} incomplete slices will be skipped during training")
    
    # 检查第一个切片的数据格式
    if cmr_slices:
        first_slice = slice_ids[0]
        cmr_info = check_nifti_file(data_root / 'images' / 'cmr' / case_id / f'{first_slice}.nii.gz')
        result['files']['cmr_example'] = cmr_info
        
        if cmr_info.get('error'):
            result['valid'] = False
            result['errors'].append(f"CMR file error: {cmr_info['error']}")
        elif len(cmr_info['shape']) < 3:
            result['warnings'].append(f"CMR shape {cmr_info['shape']} - expected 3D (T, H, W)")
    
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
        'labels/lge_original/lge_MI_labels',
        'labels/lge_original/lge_Myo_labels'
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
            complete = result.get('complete_slice_count', result['slice_count'])
            incomplete = result.get('incomplete_slice_count', 0)
            if incomplete > 0:
                print(f"  ✓ Valid ({complete} complete, {incomplete} incomplete slices)")
            else:
                print(f"  ✓ Valid ({complete} slices)")
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
    print(f"\nNote: Valid cases have at least one complete slice.")
    print(f"      Incomplete slices will be automatically skipped during training.")
    
    if invalid_count > 0:
        print("\nInvalid cases:")
        for result in validation_results:
            if not result['valid']:
                print(f"  - {result['case_id']}")
                for error in result['errors'][:2]:  # 只显示前2个错误
                    print(f"      {error}")
    
    # 统计切片数
    complete_counts = [r.get('complete_slice_count', 0) for r in validation_results]
    incomplete_counts = [r.get('incomplete_slice_count', 0) for r in validation_results]
    total_complete = sum(complete_counts)
    total_incomplete = sum(incomplete_counts)
    total_slices = total_complete + total_incomplete
    
    if total_slices > 0:
        print(f"\nSlice statistics:")
        print(f"  Total slices: {total_slices}")
        print(f"  Complete slices: {total_complete} ({total_complete/total_slices*100:.1f}%)")
        print(f"  Incomplete slices: {total_incomplete} ({total_incomplete/total_slices*100:.1f}%)")
        if complete_counts:
            print(f"  Complete slices per case:")
            print(f"    Min: {min([c for c in complete_counts if c > 0])}")
            print(f"    Max: {max(complete_counts)}")
            print(f"    Mean: {np.mean([c for c in complete_counts if c > 0]):.1f}")
            print(f"    Median: {np.median([c for c in complete_counts if c > 0]):.1f}")
    
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
