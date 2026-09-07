#!/bin/bash
# Launch script for 2x T4 DDP training on Kaggle
# Usage: ./launch_2xt4.sh

# Kaggle sets these automatically for 2 GPUs:
# RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT

torchrun --standalone --nproc_per_node=2 layerWiseTrain.py config/train_gpt2_64m_2xT4.py