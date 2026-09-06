# Config: GPT-2 (124M) — 4GB VRAM, 1.5 epochs over 1.5B tokens (layer-wise)
# Effective batch: 1 * 1024 * 32 = 32,768 tokens/step
# Total tokens: 32,768 * 68,665 ≈ 2.25B (1.5 epochs × 1.5B)
# Total steps: 68,665 | Cycles: 572 (120 steps/cycle)

# data
dataset = 'fineweb5b'

# throughput (4GB VRAM safe)
block_size = 1024
batch_size = 1
gradient_accumulation_steps = 32  # 32,768 tokens/iter

# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.2

# optimizer (tuned for layer-wise: ~20% params active per step)
learning_rate = 3e-4
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# lr schedule (1.5 epochs = ~68K steps)
max_iters = 68665
lr_decay_iters = 68665
warmup_iters = 1000
min_lr = 3e-5
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
