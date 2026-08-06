# Config for training a 64M parameter GPT model on OpenWebText
# This is a smaller model than GPT-2 (124M) for faster experimentation
# Total parameters: ~63.52M
#
# Usage:
# python train.py config/train_gpt2_64M.py
#
# Or with DDP (multi-GPU):
# torchrun --standalone --nproc_per_node=4 train.py config/train_gpt2_64M.py
#
# Expected training time:
# - Single GPU: ~10-15 hours for 30K iterations
# - 4x A100: ~3-4 hours for 30K iterations
#
# Expected results after 30K iterations:
# - Train loss: ~4.49
# - Val loss: ~4.65

wandb_log = False
wandb_project = 'owt'
wandb_run_name = 'gpt2-64M'

# Batch configuration
# Effective batch size = batch_size * block_size * gradient_accumulation_steps
# = 1 * 256 * 16 = 4,096 tokens per iteration
batch_size = 1
block_size = 256
gradient_accumulation_steps = 16

# Training duration
# 30K iterations * 4,096 tokens = ~123M tokens total
max_iters = 30000
lr_decay_iters = 30000

# Evaluation settings
eval_interval = 1000
eval_iters = 200
log_interval = 100

# Optimizer settings
weight_decay = 1e-1
learning_rate = 3e-4

# Model architecture
# 64M params: 12 layers, 8 heads, 512 embedding dim
n_layer = 12
n_head = 8
n_embd = 512
dropout = 0.2  # 0.2 for pretraining; 0.0 for finetuning

# System settings
compile = False  # Set to True for PyTorch 2.0 compilation speedup