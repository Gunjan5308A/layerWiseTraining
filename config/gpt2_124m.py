# Config: GPT-2 (124M) — 15GB VRAM (Colab T4 GPU), optimized layer-wise training
# Effective batch: 4 * 1024 * 32 = 131,072 tokens/step

# data
dataset = 'fineweb5b'

# throughput
block_size = 1024
batch_size = 4
gradient_accumulation_steps = 32

# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0

# optimizer
learning_rate = 6e-4
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# lr schedule
max_iters = 20000
lr_decay_iters = 20000
warmup_iters = 500
min_lr = 6e-5
decay_lr = True

# layer-wise training parameters
layer_examples = 20
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
always_save_checkpoint = True