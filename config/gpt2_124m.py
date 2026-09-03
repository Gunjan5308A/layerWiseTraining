# Config: GPT-2 (124M) — 4GB VRAM fit
# Effective batch: 1 * 1024 * 32 = 32,768 tokens/step (~33K)

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
learning_rate = 6e-4
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# lr schedule (~33K tokens/step * 2670 steps = ~88M tokens; fineweb5b has 1.5B)
max_iters = 2670
lr_decay_iters = 2670
warmup_iters = 133
min_lr = 6e-5
decay_lr = True

# eval & logging
eval_interval = 250
eval_iters = 100
log_interval = 5
always_save_checkpoint = True

# performance
compile = False
dtype = 'float16'                 # safer on 4GB than bfloat16