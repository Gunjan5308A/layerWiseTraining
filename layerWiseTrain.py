"""
Layer-wise training script for nanoGPT.

Algorithm:
1. Train one transformer layer at a time:
   - Active layer: requires_grad=True, LR ramps 10%→100% over first 8 steps
   - Prev layer: requires_grad=True, LR ramps 100%→10% over first 3 steps (soft freeze)
   - ALL other layers: requires_grad=False, LR=0 (hard frozen)
   - Always-trainable groups (wte, wpe, ln_f): requires_grad=True, full LR
   - Each layer trains for 'layer_examples' steps (10), then advance to next layer
   - Adam state (m, v) reset to zero when layer becomes active
   - First step after unfreeze skipped (noisy gradient)
   - Gradient norm EMA smooths gradient spikes
   - Dynamic grad_clip: 0.5 for first 3 steps, then 1.0

2. After full cycle through all 12 layers (120 steps total):
   - EMA sync: blend inactive layers toward trained layers (decay=0.99)
   - Reset cycle, repeat on next 120-step chunk of data
   - No global sync phase, no full-model unfreeze. VRAM stays flat.

3. Early stopping: halt when val loss reaches loss_stop_thresh (1e-4).

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
import subprocess
from datetime import datetime
from contextlib import nullcontext
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'
# data
dataset = 'fineweb5b'
dataset_dirs = {
    'openwebtext': 'data/openwebtext',
    'fineweb5b': 'data/fineweb5b',
    'shakespeare': 'data/shakespeare',
    'shakespeare_char': 'data/shakespeare_char',
}
gradient_accumulation_steps = 32
batch_size = 1
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
dtype = 'float16'
compile = False
# layer-wise training settings
layer_examples = 10
freeze_lr_mult = 0.1
lr_ramp_steps = 8
lr_ramp_start = 0.1
prev_ramp_steps = 3
grad_clip_warmup = 0.5
grad_clip_warmup_steps = 3
grad_ema_decay = 0.9
loss_stop_thresh = 1e-4
ema_decay = 0.99
ema_sync_every_cycle = True
save_interval = 1000
# notebook / rclone settings
notebook = False
gdrive_remote = 'gdrive'
gdrive_subdir = 'nanoGPT/checkpoints'
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

# ── Output stats directory ───────────────────────────────────────────────────
STATS_DIR = 'output_stats'
os.makedirs(STATS_DIR, exist_ok=True)
CSV_PATH = os.path.join(STATS_DIR, 'train_log.csv')
PLOT_PATH = os.path.join(STATS_DIR, 'loss_plot.png')
SUMMARY_PATH = os.path.join(STATS_DIR, 'run_summary.txt')

# ── rclone upload helper ─────────────────────────────────────────────────────
def rclone_upload(src, remote, dest_subdir):
    """Upload src file/dir to remote:dest_subdir via rclone. Returns True on success."""
    dest = f"{remote}:{dest_subdir}"
    try:
        cmd = ['rclone', 'copy', src, dest, '--transfers', '4', '--checkers', '8']
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"  >> rclone upload failed: {result.stderr.strip()}")
            return False
        print(f"  >> Uploaded to {dest}")
        return True
    except FileNotFoundError:
        print("  >> rclone not found — skipping upload")
        return False
    except subprocess.TimeoutExpired:
        print("  >> rclone upload timed out")
        return False

# ── DDP setup ────────────────────────────────────────────────────────────────
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

# ── Data loader ──────────────────────────────────────────────────────────────
data_dir = dataset_dirs.get(dataset, os.path.join('data', dataset)) if 'data_dir' not in globals() else globals()['data_dir']
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

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])


# ── Layer groups ─────────────────────────────────────────────────────────────
def get_layer_groups(gpt):
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
    mapping = {}
    for i, pg in enumerate(optimizer.param_groups):
        name = pg['layer_name']
        if name not in mapping:
            mapping[name] = []
        mapping[name].append(i)
    return mapping


def set_layer_lrs(optimizer, group_param_indices, active_group_name, base_lr, always_trainable_names, prev_group_name=None, step_in_group=0):
    ramp_ratio = min(step_in_group / lr_ramp_steps, 1.0)
    active_lr_mult = lr_ramp_start + (1.0 - lr_ramp_start) * ramp_ratio
    prev_down_ratio = min(step_in_group / prev_ramp_steps, 1.0)
    prev_lr_mult = 1.0 - (1.0 - freeze_lr_mult) * prev_down_ratio
    for name, indices in group_param_indices.items():
        for i in indices:
            if name == prev_group_name:
                optimizer.param_groups[i]['lr'] = base_lr * prev_lr_mult
            elif name in always_trainable_names:
                optimizer.param_groups[i]['lr'] = base_lr
            elif name == active_group_name:
                optimizer.param_groups[i]['lr'] = base_lr * active_lr_mult
            else:
                optimizer.param_groups[i]['lr'] = 0


def freeze_layer_params(model, always_trainable_groups, layer_wise_groups, active_group_name=None, prev_group_name=None):
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


def reset_optimizer_state(optimizer, module):
    for p in module.parameters():
        if p in optimizer.state:
            state = optimizer.state[p]
            state['exp_avg'].zero_()
            state['exp_avg_sq'].zero_()


class EMALayerSync:
    """EMA sync — no VRAM increase. Blends inactive layers toward trained ones."""

    def __init__(self, decay=0.99):
        self.decay = decay
        self.ema_weights = {}

    def snapshot(self, raw_model, layer_wise_groups):
        state = raw_model.state_dict()
        for name, module in layer_wise_groups:
            prefix = f"transformer.h.{name.split('_')[1]}."
            self.ema_weights[name] = {
                k: state[k].clone() for k in state if k.startswith(prefix)
            }

    def sync(self, raw_model, layer_wise_groups, trained_layer_names):
        if not self.ema_weights:
            return
        state = raw_model.state_dict()
        trained_indices = set()
        for tname in trained_layer_names:
            trained_indices.add(int(tname.split('_')[1]))
        for name, module in layer_wise_groups:
            if name in trained_layer_names:
                prefix = f"transformer.h.{name.split('_')[1]}."
                self.ema_weights[name] = {
                    k: state[k].clone() for k in state if k.startswith(prefix)
                }
                continue
            my_idx = int(name.split('_')[1])
            nearest = min(trained_indices, key=lambda x: abs(x - my_idx)) if trained_indices else my_idx
            src_name = f"layer_{nearest}"
            if src_name not in self.ema_weights:
                continue
            prefix = f"transformer.h.{my_idx}."
            with torch.no_grad():
                for k in list(state.keys()):
                    if k.startswith(prefix):
                        state[k].mul_(self.decay).add_(
                            self.ema_weights[src_name][k], alpha=1 - self.decay
                        )
            self.ema_weights[name] = {
                k: state[k].clone() for k in state if k.startswith(prefix)
            }


# ── Loss estimation ──────────────────────────────────────────────────────────
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


def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


# ── CSV logging ──────────────────────────────────────────────────────────────
csv_header_written = False
if master_process:
    with open(CSV_PATH, 'w') as f:
        f.write('step,train_loss,val_loss,lr,layer,wall_time,peak_vram_gb,phase\n')

def log_csv(step, train_loss, val_loss, lr, layer, wall_time, vram_gb, phase):
    if master_process:
        with open(CSV_PATH, 'a') as f:
            f.write(f'{step},{train_loss:.6f},{val_loss:.6f},{lr:.8f},{layer},{wall_time:.1f},{vram_gb:.4f},{phase}\n')


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model
running_mfu = -1.0
peak_vram = 0
layerwise_peak_vram = 0
grad_norm_ema = None

# Data accumulators
all_steps = []
all_train_loss = []
all_val_loss = []
all_lr = []
all_layers = []
all_vram = []
all_wall = []
all_cycle = []
all_step_in_cycle = []

always_trainable_groups, layer_wise_groups = get_layer_groups(raw_model)
always_trainable_names = [name for name, _ in always_trainable_groups]
n_layers = len(layer_wise_groups)

count = iter_num
optimizer = build_optimizer(raw_model, weight_decay, learning_rate, (beta1, beta2), device_type)
ema_sync = EMALayerSync(decay=ema_decay)

if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
    cycle_idx = checkpoint.get('cycle_idx', 0)
    step_in_group = checkpoint.get('step_in_group', 0)
else:
    cycle_idx = 0
    step_in_group = 0

checkpoint = None
group_param_indices = get_group_param_indices(optimizer)

# Initial freeze
prev_idx = (cycle_idx - 1) % n_layers
prev_name = layer_wise_groups[prev_idx][0] if n_layers > 1 else None
freeze_layer_params(raw_model, always_trainable_groups, layer_wise_groups, layer_wise_groups[cycle_idx][0], prev_name)

cycle_num = count // (n_layers * layer_examples)  # which full cycle we're in

print(f"Training {n_layers} layers x {layer_examples} steps each = {n_layers * layer_examples} steps/cycle")
print(f"Stats dir: {STATS_DIR}")

while count < max_iters:

    layer_name, layer_module = layer_wise_groups[cycle_idx]
    prev_idx = (cycle_idx - 1) % n_layers
    prev_name = layer_wise_groups[prev_idx][0] if n_layers > 1 else None

    freeze_layer_params(raw_model, always_trainable_groups, layer_wise_groups, layer_name, prev_name)

    if step_in_group == 0:
        reset_optimizer_state(optimizer, layer_module)

    lr = get_lr(count) if decay_lr else learning_rate
    set_layer_lrs(optimizer, group_param_indices, layer_name, lr, always_trainable_names, prev_name, step_in_group)

    # Evaluate
    if count % eval_interval == 0 and master_process:
        losses = estimate_loss()
        wall = time.time() - t0
        vram_gb = torch.cuda.max_memory_allocated(device) / 1e9 if device_type == 'cuda' else 0
        print(f"step {count}: train {losses['train']:.4f} val {losses['val']:.4f} [{layer_name}] cycle {cycle_num} ({wall:.0f}s)")
        all_steps.append(count)
        all_train_loss.append(losses['train'])
        all_val_loss.append(losses['val'])
        all_lr.append(lr)
        all_layers.append(layer_name)
        all_vram.append(vram_gb)
        all_wall.append(wall)
        all_cycle.append(cycle_num)
        all_step_in_cycle.append(step_in_group)
        log_csv(count, losses['train'], losses['val'], lr, layer_name, wall, vram_gb, 'eval')
        if losses['val'] <= loss_stop_thresh:
            print(f"  >> Val loss {losses['val']:.6f} <= {loss_stop_thresh} — stopping early")
            break
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
                }
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt_eval.pt'))
                if notebook:
                    rclone_upload(os.path.join(out_dir, 'ckpt_eval.pt'), gdrive_remote, gdrive_subdir)
    if count == 0 and eval_only:
        break

    # Skip first step after unfreeze (noisy gradient)
    if step_in_group == 0:
        optimizer.zero_grad(set_to_none=True)
        count += 1
        step_in_group += 1
        local_iter_num += 1
        if count % log_interval == 0 and master_process:
            print(f"  iter {count}: [skip — first step after unfreeze] [{layer_name}]")
        continue

    # Forward-backward
    for micro_step in range(gradient_accumulation_steps):
        X, Y = get_batch('train')
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
        scaler.scale(loss).backward()

    # Gradient norm EMA (scalar — no VRAM increase)
    if grad_ema_decay < 1.0:
        with torch.no_grad():
            total_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in raw_model.parameters() if p.requires_grad],
                float('inf')
            )
            if grad_norm_ema is None:
                grad_norm_ema = total_norm.item()
            else:
                grad_norm_ema = grad_ema_decay * grad_norm_ema + (1 - grad_ema_decay) * total_norm.item()
            if total_norm.item() > 0 and grad_norm_ema > 0:
                scale = min(grad_norm_ema / total_norm.item(), 1.0)
                for p in raw_model.parameters():
                    if p.grad is not None:
                        p.grad.mul_(scale)

    # Dynamic grad_clip: tighter during warmup
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        clip = grad_clip_warmup if step_in_group <= grad_clip_warmup_steps else grad_clip
        torch.nn.utils.clip_grad_norm_(
            [p for p in raw_model.parameters() if p.requires_grad],
            clip
        )

    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    count += 1
    step_in_group += 1

    # Timing + per-step log
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
        print(f"  iter {count}: loss {lossf:.4f} dt {dt*1000:.0f}ms [{layer_name}] cycle {cycle_num}/{step_in_group}")
        log_csv(count, lossf, float('nan'), lr, layer_name, time.time() - t0, peak_vram / 1e9 if device_type == 'cuda' else 0, 'train')

    local_iter_num += 1

    # Periodic checkpoint save
    if count % save_interval == 0 and master_process and count > 0:
        checkpoint = {
            'model': raw_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'model_args': model_args,
            'iter_num': count,
            'best_val_loss': best_val_loss,
            'config': config,
            'cycle_idx': cycle_idx,
            'step_in_group': step_in_group,
        }
        torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
        print(f"  >> Checkpoint saved at step {count}")
        if notebook:
            rclone_upload(os.path.join(out_dir, 'ckpt.pt'), gdrive_remote, gdrive_subdir)

    # Advance layer or EMA sync
    if step_in_group >= layer_examples:
        step_in_group = 0
        cycle_idx += 1
        if cycle_idx >= n_layers:
            if ema_sync_every_cycle and master_process:
                ema_sync.snapshot(raw_model, layer_wise_groups)
                all_names = [n for n, _ in layer_wise_groups]
                ema_sync.sync(raw_model, layer_wise_groups, all_names)
                print(f"  >> EMA sync done at step {count}, cycle {cycle_num} complete")
            cycle_idx = 0
            cycle_num += 1

    if count >= max_iters:
        break


# ══════════════════════════════════════════════════════════════════════════════
#  FINAL EVAL + PLOTS
# ══════════════════════════════════════════════════════════════════════════════
if master_process:
    final_losses = estimate_loss()
    wall = time.time() - t0
    vram_gb = peak_vram / 1e9 if device_type == 'cuda' else 0

    # Append final eval
    all_steps.append(count)
    all_train_loss.append(final_losses['train'])
    all_val_loss.append(final_losses['val'])
    all_lr.append(lr)
    all_layers.append('final')
    all_vram.append(vram_gb)
    all_wall.append(wall)
    all_cycle.append(cycle_num)
    all_step_in_cycle.append(0)
    log_csv(count, final_losses['train'], final_losses['val'], lr, 'final', wall, vram_gb, 'final')

    # ── Amazing plot ─────────────────────────────────────────────────────────
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1], hspace=0.25)

    # Color palette
    c_train = '#2196F3'
    c_val = '#FF9800'
    c_diff = '#4CAF50'
    c_lr = '#9C27B0'
    c_vram = '#F44336'

    # ── Panel 1: Loss curves ─────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(all_steps, all_train_loss, color=c_train, linewidth=1.8, alpha=0.9, label='Train Loss', zorder=3)
    ax1.plot(all_steps, all_val_loss, color=c_val, linewidth=2.2, alpha=1.0, label='Val Loss', zorder=4)

    # Mark cycle boundaries
    cycle_len = n_layers * layer_examples
    for c in range(1, cycle_num + 2):
        boundary = c * cycle_len
        if boundary <= all_steps[-1] + cycle_len:
            ax1.axvline(x=boundary, color='gray', linestyle=':', alpha=0.35, linewidth=0.8)

    # Annotate layer transitions within cycles
    layer_colors = plt.cm.tab20(np.linspace(0, 1, n_layers))
    for i in range(len(all_steps)):
        if all_layers[i] != 'final' and all_layers[i] != (all_layers[i-1] if i > 0 else ''):
            layer_idx = int(all_layers[i].split('_')[1])
            ax1.axvline(x=all_steps[i], color=layer_colors[layer_idx], alpha=0.15, linewidth=2)

    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Layer-Wise Training — Loss Curves', fontsize=14, fontweight='bold', pad=12)
    ax1.legend(fontsize=11, loc='upper right', framealpha=0.9)
    ax1.tick_params(labelsize=10)

    # ── Panel 2: Train-Val diff ──────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    min_len = min(len(all_train_loss), len(all_val_loss))
    if min_len > 0:
        diffs = [all_train_loss[i] - all_val_loss[i] for i in range(min_len)]
        ax2.fill_between(all_steps[:min_len], diffs, alpha=0.3, color=c_diff)
        ax2.plot(all_steps[:min_len], diffs, color=c_diff, linewidth=1.5, label='Train - Val')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax2.set_ylabel('Loss Gap', fontsize=11, fontweight='bold')
    ax2.set_title('Generalization Gap', fontsize=12, fontweight='bold', pad=8)
    ax2.legend(fontsize=10)
    ax2.tick_params(labelsize=10)

    # ── Panel 3: LR + VRAM ───────────────────────────────────────────────────
    ax3a = fig.add_subplot(gs[2], sharex=ax1)
    ax3a.plot(all_steps, all_lr, color=c_lr, linewidth=1.5, alpha=0.9, label='Learning Rate')
    ax3a.set_ylabel('LR', fontsize=11, fontweight='bold', color=c_lr)
    ax3a.set_xlabel('Step', fontsize=12, fontweight='bold')
    ax3a.set_title('Learning Rate & Peak VRAM', fontsize=12, fontweight='bold', pad=8)
    ax3a.tick_params(labelsize=10, labelcolor=c_lr)

    ax3b = ax3a.twinx()
    ax3b.plot(all_steps, all_vram, color=c_vram, linewidth=1.5, alpha=0.8, linestyle='--', label='Peak VRAM')
    ax3b.set_ylabel('VRAM (GB)', fontsize=11, fontweight='bold', color=c_vram)
    ax3b.tick_params(labelsize=10, labelcolor=c_vram)

    lines1, labels1 = ax3a.get_legend_handles_labels()
    lines2, labels2 = ax3b.get_legend_handles_labels()
    ax3a.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc='upper left')

    fig.text(0.99, 0.01, f'{count} steps | {cycle_num} cycles | Peak VRAM: {vram_gb:.2f} GB',
             ha='right', va='bottom', fontsize=9, color='gray', style='italic')

    plt.savefig(PLOT_PATH, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(os.path.join(STATS_DIR, 'loss_plot.svg'), bbox_inches='tight', facecolor='white')
    plt.show()
    plt.close()
    print(f"\nPlot saved: {PLOT_PATH}")

    # ── Run summary ──────────────────────────────────────────────────────────
    with open(SUMMARY_PATH, 'w') as f:
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"CONFIGURATION\n")
        f.write(f"{'-'*40}\n")
        for k, v in sorted(config.items()):
            f.write(f"  {k}: {v}\n")
        f.write(f"\nRESULTS\n")
        f.write(f"{'-'*40}\n")
        f.write(f"  Total steps: {count}\n")
        f.write(f"  Total cycles: {cycle_num}\n")
        f.write(f"  Final train loss: {final_losses['train']:.4f}\n")
        f.write(f"  Final val loss: {final_losses['val']:.4f}\n")
        f.write(f"  Best val loss: {best_val_loss:.4f}\n")
        f.write(f"  Peak VRAM: {vram_gb:.2f} GB\n")
        f.write(f"  Total time: {wall:.0f}s ({wall/3600:.1f}h)\n")
        f.write(f"\nLAYER ARCHITECTURE\n")
        f.write(f"{'-'*40}\n")
        f.write(f"  n_layer: {n_layer}\n")
        f.write(f"  n_head: {n_head}\n")
        f.write(f"  n_embd: {n_embd}\n")
        f.write(f"  block_size: {block_size}\n")
        f.write(f"  vocab_size: {meta_vocab_size}\n")
        f.write(f"  Total params: {sum(p.numel() for p in raw_model.parameters()):,}\n")
        f.write(f"\nTRAINING STRATEGY\n")
        f.write(f"{'-'*40}\n")
        f.write(f"  Method: Layer-wise sequential training\n")
        f.write(f"  Steps per layer: {layer_examples}\n")
        f.write(f"  Steps per cycle: {n_layers * layer_examples}\n")
        f.write(f"  Active layer LR: ramp {lr_ramp_start*100:.0f}%→100% over {lr_ramp_steps} steps ({learning_rate})\n")
        f.write(f"  Prev layer LR: soft freeze ({freeze_lr_mult*100:.0f}% of base, {learning_rate * freeze_lr_mult:.2e})\n")
        f.write(f"  Frozen layers: LR=0, requires_grad=False\n")
        f.write(f"  EMA sync: decay={ema_decay}, every cycle\n")
        f.write(f"  Early stop: val loss <= {loss_stop_thresh}\n")
        f.write(f"\nPREVIOUS LOGIC (for reference)\n")
        f.write(f"{'-'*40}\n")
        f.write(f"  Old approach: soft-freeze ALL inactive layers at 1% LR (0.01x)\n")
        f.write(f"  Old sync: global sync phase — unfreeze ALL layers, full LR, N steps\n")
        f.write(f"  Old VRAM: spiked during global sync (all layers active)\n")
        f.write(f"  New approach: hard-freeze inactive (LR=0), soft-freeze prev (0.1x LR)\n")
        f.write(f"  Active LR ramp: {lr_ramp_start*100:.0f}%→100% over {lr_ramp_steps} steps to avoid loss spike\n")
        f.write(f"  New sync: EMA blend — no unfreeze, no VRAM spike\n")
        f.write(f"  Key diff: prev layer gets momentum continuity, rest stay dead frozen\n")
    print(f"Summary saved: {SUMMARY_PATH}")

    if notebook:
        rclone_upload(STATS_DIR, gdrive_remote, gdrive_subdir)

    print(f"\n{'='*60}")
    print(f"  DONE — {count} steps, {cycle_num} cycles")
    print(f"  Train: {final_losses['train']:.4f} | Val: {final_losses['val']:.4f}")
    print(f"  Peak VRAM: {vram_gb:.2f} GB")
    print(f"  Stats: {STATS_DIR}")
    print(f"{'='*60}")

    # ── Clean up old junk ────────────────────────────────────────────────────
    old_files = [
        'layerTrain/log.txtt',
        'layerTrain/loss_plot.png',
        'layerTrain/loss_diff.png',
        'layerTrain/sample',
        'out/logs/log_gpt2-124m-optimal.txt',
        'out/logs/log_gpt2.txt',
        'plot_loss.py',
    ]
    cleaned = []
    for f in old_files:
        if os.path.exists(f):
            os.remove(f)
            cleaned.append(f)
    if cleaned:
        print(f"Cleaned: {', '.join(cleaned)}")

if ddp:
    destroy_process_group()
