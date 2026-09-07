# Config: GPT-2 64M — 2x T4 (32GB VRAM), 1.5 epochs over 600M tokens (layer-wise)
# Model: 12 layers, 8 heads, 512 embed dim = 63.5M params
# 2x T4 DDP: batch_size=8 per GPU, grad_accum=4
# Effective batch: 8 * 1024 * 4 * 2 = 65,536 tokens/step (2x 4GB config)
# Total tokens seen: 65,536 * 13,733 ≈ 900M (1.5 epochs × 600M)
# Total steps: 13,733 | Cycles: 114 (120 steps/cycle)

# data
dataset = 'fineweb600m'

# throughput (2x T4 16GB each = 32GB total)
block_size = 1024
batch_size = 8              # per GPU
gradient_accumulation_steps = 4  # 65,536 tokens/iter total

# model (64M params: 12L, 8H, 512D)
n_layer = 12
n_head = 8
n_embd = 512
dropout = 0.1
bias = False

# optimizer (scaled for 2x batch size: 4e-4 * sqrt(2) ≈ 5.6e-4, use 6e-4)
# Layer-wise: ~2 layers active = ~11M params, so effective LR per param similar
learning_rate = 6e-4
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# lr schedule (1.5 epochs = ~13,733 steps)
max_iters = 13733
lr_decay_iters = 13733
warmup_iters = 500
min_lr = 6e-5
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

# performance (T4 supports bfloat16, torch.compile helps)
compile = True
dtype = 'bfloat16'