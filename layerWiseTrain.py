"""
Layer-wise training script for nanoGPT.

This implements a soft-freezing approach where transformer layers are trained
sequentially rather than simultaneously. The key insight is that instead of
fully freezing layers (which loses AdamW momentum), we use a reduced learning
rate multiplier (freeze_lr_mult) for inactive layers.

Algorithm:
1. LAYER-WISE PHASE: For each layer i in [0, n_layer):
   - Set layer i's LR to full learning rate
   - Set all other layers' LR to lr * freeze_lr_mult (soft freeze)
   - Always-trainable groups (wte, wpe, ln_f) always get full LR
   - Run 'layer_examples' optimizer steps

2. GLOBAL SYNC PHASE: After completing a full cycle through all layers:
   - Set ALL layers to full learning rate
   - Run 'sync_steps' end-to-end steps
   - This re-aligns all layers and allows co-adaptation

Why soft-freezing?
- Fully freezing layers loses AdamW momentum, causing "cold restart" when unfreezing
- Soft-freezing preserves first/second moment estimates in AdamW
- The small LR keeps the optimizer state "warm" for each layer

Single global optimizer:
- All layers share one optimizer (not per-layer optimizers)
- This preserves AdamW momentum across the entire training process
- The LR manipulation is done via param_group['lr'] = base_lr * freeze_lr_mult

Usage (single GPU):
$ python layerWiseTrain.py --batch_size=32 --compile=False

Usage (DDP):
$ torchrun --standalone --nproc_per_node=4 layerWiseTrain.py
"""

import os
import time
import math
import pickle
import inspect
from contextlib import nullcontext
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

loss_data = []
val_loss_data = []

# -----------------------------------------------------------------------------
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'
# wandb logging
wandb_log = False
wandb_project = 'owt'
wandb_run_name = 'gpt2'
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8
batch_size = 12
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False
# adamw optimizer
learning_rate = 6e-4
max_iters = 600000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
# learning rate decay settings
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5
# DDP settings
backend = 'nccl'
# system
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False  # compile is disabled by default for layer-wise (freq param changes trigger recompilation)
# layer-wise training settings
layer_examples = 30  # number of optimizer steps per layer; all layers share the same pre-sampled data
freeze_lr_mult = 0.01   # LR multiplier for frozen layers (soft freeze)
sync_steps = 5          # end-to-end steps after each full cycle through all layers
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)
def get_batch(split):
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

iter_num = 0
best_val_loss = 1e9

meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout)
if init_from == 'scratch':
    print("Initializing a new model from scratch")
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)

if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size
model.to(device)

# peak VRAM tracking
if device_type == 'cuda':
    torch.cuda.reset_peak_memory_stats(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])


# --- Layer groups for layer-wise training ---
def get_layer_groups(gpt):
    """Return (always_trainable_groups, layer_wise_groups).

    always_trainable_groups: wte_lm_head, wpe, ln_f - always train at full LR.
    layer_wise_groups: transformer layers only - trained one at a time.
    """
    always_trainable = [
        ('wte_lm_head', nn.ModuleList([gpt.transformer.wte, gpt.lm_head])),
        ('wpe', gpt.transformer.wpe),
        ('ln_f', gpt.transformer.ln_f),
    ]
    layer_wise = []
    for i, block in enumerate(gpt.transformer.h):
        layer_wise.append((f'layer_{i}', block))
    return always_trainable, layer_wise


def build_optimizer(model, weight_decay, learning_rate, betas, device_type):
    """Create a single optimizer with one param group per layer group."""
    always_trainable_groups, layer_wise_groups = get_layer_groups(model)
    param_groups = []
    for name, module in always_trainable_groups + layer_wise_groups:
        params = [p for p in module.parameters() if p.requires_grad]
        if not params:
            continue
        decay = [p for p in params if p.dim() >= 2]
        nodecay = [p for p in params if p.dim() < 2]
        if decay:
            param_groups.append({'params': decay, 'weight_decay': weight_decay, 'lr': learning_rate, 'layer_name': name})
        if nodecay:
            param_groups.append({'params': nodecay, 'weight_decay': 0.0, 'lr': learning_rate, 'layer_name': name})

    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == 'cuda'
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(param_groups, lr=learning_rate, betas=betas, **extra_args)
    return optimizer


def get_group_param_indices(optimizer):
    """Map layer group name -> list of param_group indices."""
    mapping = {}
    for i, pg in enumerate(optimizer.param_groups):
        name = pg['layer_name']
        if name not in mapping:
            mapping[name] = []
        mapping[name].append(i)
    return mapping


def set_layer_lrs(optimizer, group_param_indices, active_group_name, base_lr, always_trainable_names, prev_group_name=None):
    """Set full LR for active group, prev group, and always-trainable groups. freeze_lr_mult for others."""
    for name, indices in group_param_indices.items():
        for i in indices:
            if name in always_trainable_names or name == active_group_name or name == prev_group_name:
                optimizer.param_groups[i]['lr'] = base_lr
            else:
                optimizer.param_groups[i]['lr'] = base_lr * freeze_lr_mult


def set_all_lrs(optimizer, group_param_indices, base_lr):
    """Set full LR for all layer groups."""
    for name, indices in group_param_indices.items():
        for i in indices:
            optimizer.param_groups[i]['lr'] = base_lr


def freeze_layer_params(model, always_trainable_groups, layer_wise_groups, active_group_name=None, prev_group_name=None):
    """Hard freeze most layers, soft freeze prev layer.

    - Always-trainable (wte, wpe, lm_head, ln_f): requires_grad=True, full LR
    - Active layer: requires_grad=True, full LR
    - Previous layer: requires_grad=True, soft freeze LR (momentum continuity)
    - All others: requires_grad=False (VRAM savings)
    If active_group_name is None (sync phase), all layers are trainable.
    """
    for _, module in layer_wise_groups:
        for p in module.parameters():
            p.requires_grad = False
    for _, module in always_trainable_groups:
        for p in module.parameters():
            p.requires_grad = True
    if active_group_name is not None:
        for name, module in layer_wise_groups:
            if name == active_group_name or name == prev_group_name:
                for p in module.parameters():
                    p.requires_grad = True


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# --- Layer-wise Training Loop ---
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model
running_mfu = -1.0
peak_vram = 0
layerwise_peak_vram = 0
sync_peak_vram = 0

# Build layer groups
always_trainable_groups, layer_wise_groups = get_layer_groups(raw_model)
always_trainable_names = [name for name, _ in always_trainable_groups]

count = iter_num  # total optimizer steps taken (used for LR decay, eval, etc.)

# Build single global optimizer
optimizer = build_optimizer(raw_model, weight_decay, learning_rate, (beta1, beta2), device_type)

if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
    cycle_idx = checkpoint.get('cycle_idx', 0)
    step_in_group = checkpoint.get('step_in_group', 0)
    in_sync_phase = checkpoint.get('in_sync_phase', False)
    sync_step = checkpoint.get('sync_step', 0)
else:
    cycle_idx = 0          # which layer group in current cycle
    step_in_group = 0      # steps within current layer group
    in_sync_phase = False  # are we in the sync phase?
    sync_step = 0          # steps within sync phase

checkpoint = None  # free up memory

# Build param group index mapping
group_param_indices = get_group_param_indices(optimizer)

# Initial freeze: set requires_grad based on starting state
if in_sync_phase:
    freeze_layer_params(raw_model, always_trainable_groups, layer_wise_groups, active_group_name=None)
else:
    prev_idx = (cycle_idx - 1) % len(layer_wise_groups)
    prev_name = layer_wise_groups[prev_idx][0] if len(layer_wise_groups) > 1 else None
    freeze_layer_params(raw_model, always_trainable_groups, layer_wise_groups, layer_wise_groups[cycle_idx][0], prev_name)

while count < max_iters:

    if not in_sync_phase:
        # === LAYER-WISE PHASE ===
        layer_name, layer_module = layer_wise_groups[cycle_idx]
        prev_idx = (cycle_idx - 1) % len(layer_wise_groups)
        prev_name = layer_wise_groups[prev_idx][0] if len(layer_wise_groups) > 1 else None

        # Hard freeze most layers, soft freeze prev layer
        freeze_layer_params(raw_model, always_trainable_groups, layer_wise_groups, layer_name, prev_name)

        # Set LR: active layer + prev layer get full LR, always-trainable get full LR, others get freeze_lr_mult
        lr = get_lr(count) if decay_lr else learning_rate
        set_layer_lrs(optimizer, group_param_indices, layer_name, lr, always_trainable_names, prev_name)

        # evaluate the loss on train/val sets and write checkpoints
        if count % eval_interval == 0 and master_process:
            losses = estimate_loss()
            print(f"step {count}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f} (phase: {layer_name})")
            val_loss_data.append(losses['val'])
            if wandb_log:
                wandb.log({
                    "iter": count,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "lr": lr,
                    "mfu": running_mfu*100,
                    "phase": layer_name,
                })
            if losses['val'] < best_val_loss or always_save_checkpoint:
                best_val_loss = losses['val']
                if count > 0:
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': count,
                        'best_val_loss': best_val_loss,
                        'config': config,
                        'cycle_idx': cycle_idx,
                        'step_in_group': step_in_group,
                        'in_sync_phase': in_sync_phase,
                        'sync_step': sync_step,
                    }
                    print(f"saving checkpoint to {out_dir}")
                    torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
        if count == 0 and eval_only:
            break

        # forward backward update with gradient accumulation
        for micro_step in range(gradient_accumulation_steps):
            X, Y = get_batch('train')
            if ddp:
                model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / gradient_accumulation_steps
            scaler.scale(loss).backward()

        # clip the gradient
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in raw_model.parameters() if p.requires_grad],
                grad_clip
            )

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        count += 1
        step_in_group += 1

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if count % log_interval == 0 and master_process:
            lossf = loss.item() * gradient_accumulation_steps
            if local_iter_num >= 5:
                mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            if device_type == 'cuda':
                peak_vram = max(peak_vram, torch.cuda.max_memory_allocated(device))
                layerwise_peak_vram = max(layerwise_peak_vram, torch.cuda.max_memory_allocated(device))
            print(f"iter {count}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%, phase: {layer_name}")
            loss_data.append(lossf)

        local_iter_num += 1

        # Advance to next layer group or sync phase
        if step_in_group >= layer_examples:
            step_in_group = 0
            cycle_idx += 1
            if cycle_idx >= len(layer_wise_groups):
                # All layers done, enter sync phase
                in_sync_phase = True
                sync_step = 0
                cycle_idx = 0  # reset for next cycle

    else:
        # === GLOBAL SYNC PHASE ===
        # Unfreeze all layers
        freeze_layer_params(raw_model, always_trainable_groups, layer_wise_groups, active_group_name=None)

        # All layers at full LR
        lr = get_lr(count) if decay_lr else learning_rate
        set_all_lrs(optimizer, group_param_indices, lr)

        # evaluate the loss on train/val sets and write checkpoints
        if count % eval_interval == 0 and master_process:
            losses = estimate_loss()
            print(f"step {count}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f} (phase: sync)")
            val_loss_data.append(losses['val'])
            if wandb_log:
                wandb.log({
                    "iter": count,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "lr": lr,
                    "mfu": running_mfu*100,
                    "phase": "sync",
                })
            if losses['val'] < best_val_loss or always_save_checkpoint:
                best_val_loss = losses['val']
                if count > 0:
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': count,
                        'best_val_loss': best_val_loss,
                        'config': config,
                        'cycle_idx': cycle_idx,
                        'step_in_group': step_in_group,
                        'in_sync_phase': in_sync_phase,
                        'sync_step': sync_step,
                    }
                    print(f"saving checkpoint to {out_dir}")
                    torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
        if count == 0 and eval_only:
            break

        # forward backward update with gradient accumulation
        for micro_step in range(gradient_accumulation_steps):
            X, Y = get_batch('train')
            if ddp:
                model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / gradient_accumulation_steps
            scaler.scale(loss).backward()

        # clip the gradient
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in raw_model.parameters() if p.requires_grad],
                grad_clip
            )

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        count += 1
        sync_step += 1
        local_iter_num += 1

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if count % log_interval == 0 and master_process:
            lossf = loss.item() * gradient_accumulation_steps
            if local_iter_num >= 5:
                mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            if device_type == 'cuda':
                peak_vram = max(peak_vram, torch.cuda.max_memory_allocated(device))
                sync_peak_vram = max(sync_peak_vram, torch.cuda.max_memory_allocated(device))
            print(f"iter {count}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%, phase: sync")
            loss_data.append(lossf)

        if sync_step >= sync_steps:
            in_sync_phase = False
            sync_step = 0

    # termination conditions
    if count >= max_iters:
        break

# Save loss plot after training
if master_process:
    train_iters = list(range(len(loss_data)))
    plt.plot(train_iters, loss_data, label='Training Loss', color='blue')
    if val_loss_data:
        val_iters = list(range(len(val_loss_data)))
        plt.plot(val_iters, val_loss_data, label='Validation Loss', color='orange')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss over Iterations (Layer-Wise)')
    plt.legend()
    plt.savefig('loss_plot.png')
    print("Loss plot saved to loss_plot.png")
    if device_type == 'cuda':
        print(f"Peak VRAM: {peak_vram / 1e9:.2f} GB (layer-wise: {layerwise_peak_vram / 1e9:.2f} GB, sync: {sync_peak_vram / 1e9:.2f} GB)")

if ddp:
    destroy_process_group()
