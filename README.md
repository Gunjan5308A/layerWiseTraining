
# nanoGPT

![nanoGPT](assets/nanogpt.jpg)

This is a fork of Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) — the simplest, fastest repository for training/finetuning medium-sized GPTs. The original repo reproduces GPT-2 (124M) on OpenWebText. For full documentation on the original codebase, model architecture, and training setup, see the [original README](https://github.com/karpathy/nanoGPT).

**Our addition:** We implemented a **layer-wise training** approach (`layerWiseTrain.py`) that trains transformer layers sequentially with soft-freezing, plus a 64M parameter model configuration for faster experimentation.

---

## What We Added

### 1. Layer-Wise Training (`layerWiseTrain.py`)

A new training script that trains transformer blocks one at a time in cycles, instead of updating all layers simultaneously every step. This uses **soft-freezing** — inactive layers get a small learning rate multiplier rather than being fully frozen, which preserves AdamW optimizer momentum.

### 2. 64M Model Config (`config/train_gpt2_64M.py`)

A smaller model (12 layers, 8 heads, 512 embed dim) for faster experimentation on a single GPU. Trains for 30K iterations on OpenWebText, reaching ~4.65 validation loss.

### 3. Convenience Script (`startTrain.sh`)

One-command data prep + training:
```sh
bash startTrain.sh
```

---

## Layer-Wise Training: Math and How It Works

### Standard Training vs Layer-Wise

In **standard training**, every forward/backward pass computes gradients for all layers, and the optimizer updates every layer's weights each step:

```
For each step:
    logits, loss = model(x, y)       # all layers active
    loss.backward()                   # gradients for ALL layers
    optimizer.step()                  # ALL layers updated
```

In **layer-wise training**, we cycle through layers, giving each one focused optimization time while the others are "soft-frozen":

```
For each layer i in [0, n_layer):
    For step in [0, layer_examples):
        logits, loss = model(x, y)           # forward pass (all layers active)
        loss.backward()                       # gradients for all layers
        optimizer.step()                      # but only layer i gets meaningful update
```

### Soft-Freezing Math

Instead of fully freezing layers (setting `requires_grad=False`), we reduce their learning rate:

```
lr_j = base_lr               if j == active_layer
lr_j = base_lr * 0.01        otherwise          # freeze_lr_mult = 0.01
```

**Why not hard-freeze?** AdamW maintains two running averages per parameter:

- `m_t = β₁ * m_{t-1} + (1 - β₁) * g_t`  (first moment / momentum)
- `v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²`  (second moment / adaptive LR)

If you hard-freeze a layer (LR=0), `g_t = 0`, so:
- `m_t` decays toward zero (momentum dies)
- `v_t` decays toward zero (adaptive scaling resets)

When you later unfreeze, the optimizer has "forgotten" the gradient history — a **cold restart**. Soft-freezing avoids this by keeping a small `g_t` flowing through the optimizer state.

### Cycle Structure

One full cycle through all layers:

```
Layer 0:  30 steps  (LR for layer 0, 0.01*LR for layers 1-11)
Layer 1:  30 steps  (LR for layer 1, 0.01*LR for layers 0,2-11)
Layer 2:  30 steps
...
Layer 11: 30 steps
Sync:      5 steps  (all layers at full LR)
```

Total steps per cycle: `12 * 30 + 5 = 365`

The **sync phase** runs a few end-to-end steps where all layers train at full LR, allowing them to co-adapt after individual optimization.

### Always-Trainable Groups

Three components never get soft-frozen — they train at full LR throughout:

| Group | Parameters | Why |
|-------|-----------|-----|
| `wte` + `lm_head` | Token embeddings + output projection | Tied weights, needed for vocabulary alignment |
| `wpe` | Position embeddings | Must track sequence position for all layers |
| `ln_f` | Final layer norm | Normalizes output before logits |

### Single Global Optimizer

All layers share **one** AdamW optimizer (not separate optimizers per layer). This is critical because:

1. The optimizer state (momentum, adaptive LR) is preserved across the entire training
2. Layer-specific LR is controlled via `optimizer.param_groups[i]['lr']`
3. No state duplication or synchronization overhead

```python
# Param groups: one per layer (split into decay/no-decay)
param_groups = [
    {'params': layer_0_decay, 'lr': base_lr, 'layer_name': 'layer_0'},
    {'params': layer_0_nodecay, 'lr': base_lr, 'layer_name': 'layer_0'},
    {'params': layer_1_decay, 'lr': base_lr * 0.01, 'layer_name': 'layer_1'},
    ...
]
```

### Learning Rate Schedule

Uses cosine annealing with linear warmup, applied to `base_lr` before scaling by `freeze_lr_mult`:

```
lr(t) =
  t < warmup_iters:     lr_max * (t + 1) / (warmup_iters + 1)
  t > lr_decay_iters:   lr_min
  otherwise:            lr_min + 0.5*(1 + cos(π * decay_ratio)) * (lr_max - lr_min)
```

### Gradient Accumulation

The effective batch size is:

```
tokens_per_step = batch_size * block_size * gradient_accumulation_steps * ddp_world_size
```

For our config: `1 * 256 * 16 * 1 = 4,096 tokens`

Loss is divided by `gradient_accumulation_steps` before backprop to average gradients correctly across micro-batches.

---

## How to Run

### Prerequisites

```sh
pip install torch numpy transformers datasets tiktoken wandb tqdm matplotlib
```

### Layer-Wise Training

```sh
# Single GPU
python layerWiseTrain.py \
    --batch_size=1 \
    --block_size=256 \
    --gradient_accumulation_steps=16 \
    --compile=False

# Multi-GPU (DDP)
torchrun --standalone --nproc_per_node=4 layerWiseTrain.py

# Override any parameter
python layerWiseTrain.py --layer_examples=50 --freeze_lr_mult=0.05 --max_iters=50000
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `layer_examples` | 30 | Optimizer steps per layer per cycle |
| `freeze_lr_mult` | 0.01 | LR multiplier for soft-frozen layers |
| `sync_steps` | 5 | End-to-end steps after each full cycle |
| `compile` | False | Disabled because param group LR changes trigger recompilation |

### Standard Training (for comparison)

```sh
# 64M model
python train.py config/train_gpt2_64M.py

# Shakespeare character-level
python train.py config/train_shakespeare_char.py
```

### Sampling

```sh
python sample.py --out_dir=out
```

---

## Results

Training a 64M parameter model on OpenWebText:

| Approach | Iterations | Val Loss | Notes |
|----------|-----------|----------|-------|
| Standard (`train.py`) | 30K | 4.65 | All layers update every step |
| Layer-wise (`layerWiseTrain.py`) | 11K | 4.96 | Layers trained sequentially, still converging |

The layer-wise approach converges but with different dynamics — each layer gets focused optimization time, and the sync phases re-align them.

---

## File Changes from Original nanoGPT

| File | Status | Description |
|------|--------|-------------|
| `layerWiseTrain.py` | **New** | Layer-wise training script (~520 lines) |
| `config/train_gpt2_64M.py` | **New** | 64M model configuration |
| `startTrain.sh` | **New** | Data prep + training convenience script |
| `n_tokens.py` | **New** | Token counting utility |
| `README.md` | **Modified** | Added layer-wise training docs |
| `model.py` | Unchanged | Original GPT model definition |
| `train.py` | Unchanged | Original training loop |
| `sample.py` | Unchanged | Original sampling script |

---

## License

MIT License — same as original [nanoGPT](https://github.com/karpathy/nanoGPT).
