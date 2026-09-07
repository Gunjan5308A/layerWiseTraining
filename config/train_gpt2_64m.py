# Config: GPT-2 64M — 4GB VRAM, 1.5 epochs over 600M tokens (layer-wise)
# Model: 12 layers, 8 heads, 576 embed dim ≈ 65M params
# Effective batch: 1 * 1024 * 32 = 32,768 tokens/step
# Total tokens seen: 32,768 * 27,466 ≈ 900M (1.5 epochs × 600M)
# Total steps: 27,466 | Cycles: 229 (120 steps/cycle = 12 layers × 10 steps)

# data
dataset = 'fineweb600m'

# throughput (4GB VRAM safe)
block_size = 1024
batch_size = 1
gradient_accumulation_steps = 32  # 32,768 tokens/iter

# model (64M params: 12L, 8H, 512D)
n_layer = 12
n_head = 8
n_embd = 512
dropout = 0.1
bias = False

# optimizer (tuned for layer-wise: ~2 layers active per step = ~11M active params)
# Base LR 4e-4 (vs 3e-4 for 124M) — smaller model, fewer total steps, slightly higher LR
# freeze_lr_mult=0.1 → prev layer gets 4e-5
learning_rate = 4e-4
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# lr schedule (1.5 epochs = ~27,466 steps)
max_iters = 27466
lr_decay_iters = 27466
warmup_iters = 500
min_lr = 4e-5
decay_lr = True

# layer-wise training (loss spike reduction)
layer_examples = 10
freeze_lr_mult = 0.1
lr_ramp_steps = 8
lr_ramp_start = 0.1
prev_ramp_steps = 3
grad_clip_warmup = 0.5
grad_clip_warmup_steps = 3
grad_ema_decay = 0.9
loss_stop_thresh = 1e-4
save_interval = 500

# eval & logging
eval_interval = 500
eval_iters = 200
log_interval = 1
always_save_checkpoint = False

# performance
compile = False
dtype = 'float16'