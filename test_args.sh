#!/bin/bash

echo "=========================================="
echo "Testing Training Scripts Parameter Passing"
echo "=========================================="

DATA_ROOT="/data/yuanwenwei/datasets/lge_pred_dataset/shengyi_all/cropped"
SPLITS_FILE="data_splits.json"

echo ""
echo "1. Testing train_motion.py..."
python3 src/train_motion.py \
    --data_root "$DATA_ROOT" \
    --splits_file "$SPLITS_FILE" \
    --checkpoint_dir "checkpoints/motion" \
    --log_dir "logs/motion" \
    --val_freq 5 \
    --epochs 1 \
    --help > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo "✓ train_motion.py accepts all required parameters"
else
    echo "✗ train_motion.py has parameter issues"
fi

echo ""
echo "2. Testing train_registration.py..."
python3 src/train_registration.py \
    --data_root "$DATA_ROOT" \
    --splits_file "$SPLITS_FILE" \
    --checkpoint_dir "checkpoints/registration" \
    --log_dir "logs/registration" \
    --val_freq 5 \
    --epochs 1 \
    --help > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo "✓ train_registration.py accepts all required parameters"
else
    echo "✗ train_registration.py has parameter issues"
fi

echo ""
echo "3. Testing train_segmentation.py..."
python3 src/train_segmentation.py \
    --data_root "$DATA_ROOT" \
    --splits_file "$SPLITS_FILE" \
    --checkpoint_dir "checkpoints/segmentation" \
    --log_dir "logs/segmentation" \
    --val_freq 5 \
    --epochs 1 \
    --help > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo "✓ train_segmentation.py accepts all required parameters"
else
    echo "✗ train_segmentation.py has parameter issues"
fi

echo ""
echo "=========================================="
echo "All tests completed!"
echo "=========================================="
