#!/bin/bash
set -e

echo "=== Step 1: Data Preparation ==="
python data/fineweb5b/prepare.py

echo ""
echo "=== Step 2: Training ==="
python train.py config/train_gpt2_64M.py
