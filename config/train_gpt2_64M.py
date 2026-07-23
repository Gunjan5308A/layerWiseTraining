# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

wandb_log = False
wandb_project = 'owt'
wandb_run_name='gpt2-64M'

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 1
block_size = 256
gradient_accumulation_steps = 16

# this makes total number of tokens be 300B
max_iters = 30000
lr_decay_iters = 30000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 100

# weight decay
weight_decay = 1e-1
learning_rate = 3e-4

n_layer = 12
n_head = 8      # or 8
n_embd = 512
dropout = 0.2

compile = False