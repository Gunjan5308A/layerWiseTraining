# Config: GPT-2 (124M) trained for 2 epochs on the 5B-token FineWeb subset.
# Dataset prepared by: python data/fineweb5b/prepare.py
#
# Usage:
#   python train.py config/gpt2_124m.py
#
# Sizing (fits a 4 GB VRAM GPU, e.g. RTX 3050):
#   - model params (fp32)              ~0.50 GB
#   - gradients (fp32)                 ~0.50 GB
#   - AdamW m/v states (fp32)          ~1.00 GB
#   - activations (bf16, b=2, seq 1024)~0.35 GB
#   - CUDA context / misc              ~0.40 GB
#   - total                            ~2.75 GB  < 4 GB
#
# Throughput:
#   tokens/iter = batch_size * block_size * gradient_accumulation_steps
#               = 2 * 1024 * 64 = 131,072
#   train tokens (2 epochs) = 2 * 4.99e9 = 9.98e9
#   max_iters               = 9.98e9 / 131072 = 76,142

# wandb logging
wandb_log = True
wandb_project = 'fineweb5b'
wandb_run_name = 'gpt2-124m'

# data
dataset = 'fineweb5b'
gradient_accumulation_steps = 64  # effective batch 131,072 tokens/iter
batch_size = 2                    # micro-batch, small for 4 GB VRAM
block_size = 1024

# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0

# optimizer
learning_rate = 6e-4
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# schedule: 2 epochs over ~4.99B train tokens
max_iters = 76142
lr_decay_iters = 76142
warmup_iters = 500
min_lr = 6e-5
decay_lr = True

# eval
eval_interval = 1000
eval_iters = 100
log_interval = 1
always_save_checkpoint = True

# system
compile = False  # torch.compile can OOM the 4 GB GPU during compilation
dtype = 'bfloat16'  # falls back to float16 w/ GradScaler if unsupported