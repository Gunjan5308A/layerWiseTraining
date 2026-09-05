
# LayerWiseTraining

This is a fork of Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) — the simplest, fastest repository for training/finetuning medium-sized GPTs. The original repo reproduces GPT-2 (124M) on OpenWebText. For full documentation on the original codebase, model architecture, and training setup, see the [original README](https://github.com/karpathy/nanoGPT).

**Our addition:** We implemented a **layer-wise training** approach (`layerWiseTrain.py`) that trains transformer layers sequentially with a hybrid freeze strategy (hard freeze + 10% soft freeze), gradual LR ramp, optimizer state reset, and multiple loss spike reduction techniques — all without increasing VRAM usage.

---

## What We Added

### 1. Layer-Wise Training (`layerWiseTrain.py`)

A new training script that trains transformer blocks one at a time in cycles, instead of updating all layers simultaneously every step. Uses a **hybrid freeze strategy** — the active layer trains with a gradual LR ramp (10%→100% over 8 steps), its immediate predecessor ramps down (100%→10% over 3 steps) for smooth transition, and all other transformer layers are hard-frozen (`requires_grad=False`) to save VRAM. Includes 5 loss spike reduction techniques: LR ramp, prev ramp-down, skip first step, gradient norm EMA, and dynamic grad clipping. Embeddings, output head, and final layer norm always train at full LR. Early stopping when validation loss reaches threshold.

### 2. 64M Model Config (`config/train_gpt2_64M.py`)

A smaller model (12 layers, 8 heads, 512 embed dim) for faster experimentation on a single GPU. Trains for 30K iterations on OpenWebText, reaching ~4.65 validation loss.

### 3. FineWeb5B Dataset Support

FineWeb5B is now the **default dataset**. It contains ~1.5B tokens from HuggingFace's FineWeb dataset (sample-10BT), preprocessed into `train.bin` (~3GB) and `val.bin` (~20MB) in `data/fineweb5b/`.

```sh
# Prepare the dataset (if not already present)
python data/fineweb5b/prepare.py
```

### 4. 4GB VRAM Config (`config/gpt2_124m.py`)

Optimized config for GPUs with ~4GB VRAM:
- `batch_size=1`, `gradient_accumulation_steps=32` → 32,768 tokens/step
- `dtype='float16'`, `compile=False`
- Full GPT-2 124M model fits in 4GB

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

In **layer-wise training**, we cycle through layers, giving each one focused optimization time while others are hard-frozen (except the prev layer which ramps down from 100% to 10% LR):

```
For each layer i in [0, n_layer):
    freeze_layer_params(active=i, prev=i-1)   # hard freeze all except active+prev
    reset_optimizer_state(active=i)           # zero m, v for clean slate
    For step in [0, layer_examples):
        if step == 0: skip                    # skip first step (noisy gradient)
        active_lr = 0.1 + 0.9 * min(step / 8, 1)   # ramp 10%→100% over 8 steps
        prev_lr = 1.0 - 0.9 * min(step / 3, 1)     # ramp 100%→10% over 3 steps
        logits, loss = model(x, y)           # forward pass
        loss.backward()                       # gradients for active+prev+always-trainable
        grad_norm_ema = 0.9 * grad_norm_ema + 0.1 * ||grad||  # smooth gradient spikes
        grads *= min(grad_norm_ema / ||grad||, 1)              # scale down spikes
        clip = 0.5 if step <= 3 else 1.0      # tighter clip during warmup
        optimizer.step()                      # update active+prev+always-trainable
```

### Hybrid Freeze Strategy

We combine hard-freezing (for VRAM savings) with soft-freezing (for momentum continuity), gradual LR ramps (to prevent loss spikes), and gradient smoothing:

```
requires_grad = True,  lr = base_lr * ramp_pct    if j == active_layer  (ramps 10%→100%)
requires_grad = True,  lr = base_lr * prev_pct    if j == prev_layer   (ramps 100%→10%)
requires_grad = True,  lr = base_lr                if j in always_trainable
requires_grad = False, lr = 0                      otherwise            (hard freeze)
```

Where:
- `ramp_pct = 0.1 + 0.9 * min(step_in_layer / 8, 1.0)` — linearly increases from 10% to 100% over first 8 steps
- `prev_pct = 1.0 - 0.9 * min(step_in_layer / 3, 1.0)` — linearly decreases from 100% to 10% over first 3 steps

**Why hybrid?** Pure soft-freezing (`requires_grad=True` for all) computes gradients for every parameter — no VRAM savings. Pure hard-freezing (`requires_grad=False`) causes cold restarts when layers unfreeze. The hybrid approach gets both:

- **VRAM savings**: 10 of 12 transformer blocks have `requires_grad=False`, so no gradients computed or stored for them (~83% VRAM reduction per step)
- **Momentum continuity**: The previous layer ramps down from 100% to 10% LR, maintaining AdamW momentum without sudden drops
- **Stability**: Multiple techniques prevent loss spikes (see Loss Spike Reduction section below)

**AdamW momentum recap:**

- `m_t = β₁ * m_{t-1} + (1 - β₁) * g_t`  (first moment / momentum)
- `v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²`  (second moment / adaptive LR)

If `requires_grad=False`, `g_t = 0`, so momentum decays toward zero — a **cold restart** when unfrozen. The prev layer avoids this.

### Cycle Structure

One full cycle through all layers:

```
Layer 0:  10 steps  (active: layer 0, prev: layer 11, frozen: layers 1-10)
          step 0: skip (noisy gradient)
          steps 1-8: LR ramp 10%→100%
          steps 1-3: prev ramp 100%→10%, grad_clip=0.5
          steps 4-10: prev at 10%, grad_clip=1.0
Layer 1:  10 steps  (active: layer 1, prev: layer 0,  frozen: layers 2-11)
          same pattern...
Layer 2:  10 steps  (active: layer 2, prev: layer 1,  frozen: layers 0,3-11)
          same pattern...
...
Layer 11: 10 steps  (active: layer 11, prev: layer 10, frozen: layers 0-9)
          same pattern...
EMA sync: blend inactive layers toward nearest trained (decay=0.99)
```

Total steps per cycle: `12 * 10 = 120`

The **EMA sync** blends each inactive layer toward its nearest trained layer using exponential moving average, maintaining weight coherence without the VRAM spike of a full unfreeze.

### Always-Trainable Groups

Three components always have `requires_grad=True` and train at full LR — they are never frozen:

| Group | Parameters | Why |
|-------|-----------|-----|
| `wte` + `lm_head` | Token embeddings + output projection | Tied weights, needed for vocabulary alignment |
| `wpe` | Position embeddings | Must track sequence position for all layers |
| `ln_f` | Final layer norm | Normalizes output before logits |

Additionally, the **previous layer** (relative to the active layer) always has `requires_grad=True` and ramps down from 100% to 10% LR over 3 steps to maintain AdamW momentum continuity without sudden drops.

### Loss Spike Reduction

Six techniques prevent loss spikes during layer transitions — **all without increasing VRAM**:

| Technique | How It Works | VRAM Cost |
|-----------|--------------|-----------|
| **Active LR ramp** | Linearly increase active layer LR from 10%→100% over 8 steps | None |
| **Prev LR ramp-down** | Linearly decrease prev layer LR from 100%→10% over 3 steps | None |
| **Skip first step** | Don't step optimizer on `step_in_group == 0` (noisy gradient) | None |
| **Optimizer state reset** | Zero Adam `m, v` when layer becomes active (clean slate) | None |
| **Gradient norm EMA** | Scalar EMA of total gradient norm, scale grads by `ema/current` | ~0 (one float) |
| **Dynamic grad_clip** | Use 0.5 for first 3 steps, then 1.0 | None |

**Why these work:**
- Layer transitions cause gradient distribution shifts
- First gradient after unfreeze is often noisy (skip it)
- Stale optimizer state (m, v) from previous training causes instability (reset it)
- Large gradient spikes get smoothed by EMA scaling
- Tighter clipping during warmup prevents overshooting

### Single Global Optimizer

All layers share **one** AdamW optimizer (not separate optimizers per layer). This is critical because:

1. The optimizer state (momentum, adaptive LR) is preserved across the entire training
2. Layer-specific LR is controlled via `optimizer.param_groups[i]['lr']`
3. No state duplication or synchronization overhead

```python
# Param groups: one per layer (split into decay/no-decay)
param_groups = [
    {'params': layer_0_decay, 'lr': base_lr * ramp_pct, 'layer_name': 'layer_0'},  # active, ramps 50%→100%
    {'params': layer_0_nodecay, 'lr': base_lr * ramp_pct, 'layer_name': 'layer_0'},
    {'params': layer_1_decay, 'lr': base_lr * 0.1, 'layer_name': 'layer_1'},       # prev, 10% soft freeze
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

**4GB VRAM config:** `1 * 1024 * 32 * 1 = 32,768 tokens/step`

Loss is divided by `gradient_accumulation_steps` before backprop to average gradients correctly across micro-batches.

### Logging (No wandb Required)

Training logs are written to local CSV files at `out/logs/log_{run_name}.txt`:

```
iter,train_loss,val_loss,lr,phase,tokens_per_iter
0,10.8234,10.8102,0.000006,layer_0,32768
5,9.4521,9.4312,0.000042,layer_0,32768
...
```

This replaces wandb (which requires an account and costs money). Logs can be loaded into any CSV viewer, Jupyter notebook, or plotting tool.

---

## How to Run

### Prerequisites

```sh
pip install torch numpy transformers datasets tiktoken tqdm matplotlib
```

### Layer-Wise Training

```sh
# Default: 4GB VRAM, fineweb5b dataset
python layerWiseTrain.py config/gpt2_124m.py

# Override dataset
python layerWiseTrain.py config/gpt2_124m.py --dataset=openwebtext

# Override any parameter
python layerWiseTrain.py --layer_examples=50 --freeze_lr_mult=0.05 --max_iters=50000

# Multi-GPU (DDP)
torchrun --standalone --nproc_per_node=4 layerWiseTrain.py config/gpt2_124m.py
```

### Standard Training

```sh
# 124M model on fineweb5b
python train.py config/gpt2_124m.py

# Shakespeare character-level
python train.py config/train_shakespeare_char.py
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dataset` | `fineweb5b` | Dataset name: `fineweb5b`, `openwebtext`, `shakespeare`, `shakespeare_char` |
| `data_dir` | auto from dataset | Override full path: `--data_dir=/path/to/data` |
| `layer_examples` | 10 | Optimizer steps per layer per cycle |
| `freeze_lr_mult` | 0.1 | LR multiplier for soft-frozen prev layer (10% of base LR) |
| `lr_ramp_steps` | 8 | Steps to ramp active layer LR from 10% to 100% |
| `lr_ramp_start` | 0.1 | Starting LR multiplier for active layer ramp (10%) |
| `prev_ramp_steps` | 3 | Steps to ramp prev layer LR from 100% to 10% |
| `grad_clip_warmup` | 0.5 | Grad clip value during first 3 steps |
| `grad_clip_warmup_steps` | 3 | Steps to use tighter grad clip |
| `grad_ema_decay` | 0.9 | EMA decay for gradient norm smoothing |
| `loss_stop_thresh` | 1e-4 | Stop training when val loss reaches this threshold |
| `compile` | False | Disabled because param group LR changes trigger recompilation |
| `dtype` | `float16` | `float16` for 4GB VRAM, `bfloat16` for 8GB+ |

### Sampling

```sh
python sample.py --out_dir=out
```

### Visualization

Interactive HTML visualization of the layer-wise training algorithm:
```sh
xdg-open layerwise_viz.html
```

---

## Results

Training a 64M parameter model on OpenWebText:

| Approach | Iterations | Val Loss | Notes |
|----------|-----------|----------|-------|
| Standard (`train.py`) | 30K | 4.65 | All layers update every step |
| Layer-wise (`layerWiseTrain.py`) | 11K | 4.96 | Layers trained sequentially, still converging |

The layer-wise approach converges but with different dynamics — each layer gets focused optimization time, and the sync phases re-align them.

### VRAM Measurement

Both scripts track peak GPU VRAM usage and print a summary at the end of training:

**`train.py`:**
```
Peak VRAM: 4.23 GB
```

**`layerWiseTrain.py`:**
```
Peak VRAM: 2.87 GB (flat — no sync phase spike)
```

The layer-wise approach maintains flat VRAM usage throughout training because there's no global unfreeze phase — only 2 layers (active + prev) have gradients at any time, and EMA sync blends weights without requiring all parameters to be trainable simultaneously.

---

## File Changes from Original nanoGPT

| File | Status | Description |
|------|--------|-------------|
| `layerWiseTrain.py` | **Modified** | Layer-wise training script (~750 lines), local CSV logging, LR ramp, optimizer reset, gradient norm EMA, dynamic grad_clip, early stopping |
| `train.py` | **Modified** | Added `dataset_dirs` mapping, local CSV logging, VRAM tracking |
| `config/gpt2_124m.py` | **Modified** | 4GB VRAM optimized config, fineweb5b default |
| `config/train_gpt2_64M.py` | **New** | 64M model configuration |
| `data/fineweb5b/prepare.py` | **New** | FineWeb 1.5B token extractor |
| `layerwise_viz.html` | **New** | Interactive HTML visualization of layer-wise training |
| `n_tokens.py` | **New** | Token counting utility |
| `model.py` | Unchanged | Original GPT model definition |
| `sample.py` | Unchanged | Original sampling script |

---init

## License

MIT License — same as original [nanoGPT](https://github.com/karpathy/nanoGPT).
