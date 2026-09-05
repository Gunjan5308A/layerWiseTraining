# Config: GPT-2 (124M) — 4GB VRAM, optimized layer-wise training
# Effective batch: 1 * 1024 * 128 = 131,072 tokens/step (stabilizes noisy gradients)
# Total steps: 20,000 | Cycles: 166 (120 steps/cycle)

# data
dataset = 'fineweb5b'

# throughput (4GB VRAM safe)
block_size = 1024
batch_size = 1
gradient_accumulation_steps = 128  # Increased from 32 to 128 for smoother gradient updates

# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0

# optimizer (tuned for layer-wise: ~20% params active per step)
learning_rate = 6e-4
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# lr schedule (adjusted decay schedule so model settles into local minima faster)
max_iters = 20000
lr_decay_iters = 20000
warmup_iters = 500
min_lr = 6e-5
decay_lr = True

# layer-wise training (loss spike reduction)
layer_examples = 20  # Increased from 10 to 20 steps per layer before rotation
freeze_lr_mult = 0.1
lr_ramp_steps = 8
lr_ramp_start = 0.1
prev_ramp_steps = 3
grad_clip_warmup = 0.5
grad_clip_warmup_steps = 3
grad_ema_decay = 0.9
loss_stop_thresh = 1e-4