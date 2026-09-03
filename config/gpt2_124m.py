# Config: GPT-2 (124M) — 4GB VRAM, 2 epochs over 1.5B tokens
# Effective batch: 1 * 1024 * 32 = 32,768 tokens/step
# Total tokens: 32,768 * 91,553 ≈ 3B (2 epochs × 1.5B)

# data
dataset = 'fineweb5b'

# throughput (4GB VRAM safe)
block_size = 1024
batch_size = 1
gradient_accumulation_steps = 32  # 1 * 1024 * 32 = 32,768 tokens/iter

# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0

# optimizer
learning_rate = 1e-4
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# lr schedule (2 epochs = ~91K steps)
max_iters = 91553
lr_decay_iters = 91553
warmup_iters = 1000
min_lr = 1e-5
decay_lr = True

# layer-wise training
layer_examples = 10
freeze_lr_mult = 0.1
lr_ramp_steps = 5
lr_ramp_start = 0.5
loss_stop_thresh = 1e-4

# eval & logging
eval_interval = 500
eval_iters = 200
log_interval = 1
always_save_checkpoint = True

# performance
compile = False
dtype = 'float16'                 # safer on 4GB than bfloat16