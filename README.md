
# LayerWiseTraining

This is a fork of Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) — the simplest, fastest repository for training/finetuning medium-sized GPTs. The original repo reproduces GPT-2 (124M) on OpenWebText. For full documentation on the original codebase, model architecture, and training setup, see the [original README](https://github.com/karpathy/nanoGPT).

**Our addition:** We implemented a **layer-wise training** approach (`layerWiseTrain.py`) that trains transformer layers sequentially with a hybrid freeze strategy (hard freeze + soft freeze), plus a 64M parameter model configuration for faster experimentation.

---

## What We Added

### 1. Layer-Wise Training (`layerWiseTrain.py`)

A new training script that trains transformer blocks one at a time in cycles, instead of updating all layers simultaneously every step. Uses a **hybrid freeze strategy** — the active layer and its immediate predecessor train at full LR (`requires_grad=True`), while all other transformer layers are hard-frozen (`requires_grad=False`) to save VRAM. Embeddings, output head, and final layer norm always train at full LR.

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

In **layer-wise training**, we cycle through layers, giving each one focused optimization time while others are hard-frozen (except the prev layer which stays soft-frozen):

```
For each layer i in [0, n_layer):
    freeze_layer_params(active=i, prev=i-1)   # hard freeze all except active+prev
    For step in [0, layer_examples):
        logits, loss = model(x, y)           # forward pass
        loss.backward()                       # gradients only for active+prev+always-trainable
        optimizer.step()                      # only active+prev+always-trainable updated
    sync_steps end-to-end steps               # all layers at full LR
```

### Hybrid Freeze Strategy

We combine hard-freezing (for VRAM savings) with soft-freezing (for momentum continuity):

```
requires_grad = True,  lr = base_lr          if j == active_layer
requires_grad = True,  lr = base_lr          if j == prev_layer     # soft freeze
requires_grad = True,  lr = base_lr          if j in always_trainable
requires_grad = False, lr = base_lr * 0.01   otherwise              # hard freeze
```

**Why hybrid?** Pure soft-freezing (`requires_grad=True` for all) computes gradients for every parameter — no VRAM savings. Pure hard-freezing (`requires_grad=False`) causes cold restarts when layers unfreeze. The hybrid approach gets both:

- **VRAM savings**: 10 of 12 transformer blocks have `requires_grad=False`, so no gradients computed or stored for them (~83% VRAM reduction per step)
- **Momentum continuity**: The previous layer stays soft-frozen (`requires_grad=True`) so AdamW momentum doesn't decay to zero

**AdamW momentum recap:**

- `m_t = β₁ * m_{t-1} + (1 - β₁) * g_t`  (first moment / momentum)
- `v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²`  (second moment / adaptive LR)

If `requires_grad=False`, `g_t = 0`, so momentum decays toward zero — a **cold restart** when unfrozen. The prev layer avoids this.

### Cycle Structure

One full cycle through all layers:

```
Layer 0:  30 steps  (active: layer 0, prev: layer 11, frozen: layers 1-10)
Layer 1:  30 steps  (active: layer 1, prev: layer 0,  frozen: layers 2-11)
Layer 2:  30 steps  (active: layer 2, prev: layer 1,  frozen: layers 0,3-11)
...
Layer 11: 30 steps  (active: layer 11, prev: layer 10, frozen: layers 0-9)
Sync:      5 steps  (all layers at full LR)
```

Total steps per cycle: `12 * 30 + 5 = 365`

The **sync phase** runs a few end-to-end steps where all layers train at full LR, allowing them to co-adapt after individual optimization.

### Always-Trainable Groups

Three components always have `requires_grad=True` and train at full LR — they are never frozen:

| Group | Parameters | Why |
|-------|-----------|-----|
| `wte` + `lm_head` | Token embeddings + output projection | Tied weights, needed for vocabulary alignment |
| `wpe` | Position embeddings | Must track sequence position for all layers |
| `ln_f` | Final layer norm | Normalizes output before logits |

Additionally, the **previous layer** (relative to the active layer) always has `requires_grad=True` at full LR to maintain AdamW momentum continuity.

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
| `freeze_lr_mult` | 0.01 | LR multiplier for soft-frozen layers (used for optimizer state only) |
| `sync_steps` | 5 | End-to-end steps after each full cycle |
| `compile` | False | Disabled because param group LR changes trigger recompilation |

**VRAM note**: Only 2-3 transformer blocks have `requires_grad=True` at any time (active + prev + always-trainable). The other 10 blocks are hard-frozen with no gradients computed, saving ~83% of transformer VRAM per step.

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
